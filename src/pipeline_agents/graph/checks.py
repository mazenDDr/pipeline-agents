"""Deterministic checks between the Executor and the Critic, and at delivery (T6). No model calls.

`step_checks` picks the validation tools that fit the step's kind and the files it wrote. `deliver_checks`
assembles output/pipeline.py from the accepted steps, re-runs it from a clean copy in the sandbox, checks the
deliverables exist, and (for predictive tasks) runs predict.py on a slice of the training data.
"""

import json
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from pipeline_agents.schemas import Finding, PlanStep, RunState, StepAttempt
from pipeline_agents.tools.profile import profile_file
from pipeline_agents.tools.validators import (
    metric_sanity,
    missing_and_sentinels,
    numbers_stored_as_text,
    required_columns,
    row_accounting,
    split_overlap,
    target_leakage,
    temporal_order,
)

MAX_TABLES = 3


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:  # an unreadable table is itself worth a finding, not a crash
        return None


TRAIN_NAMES = ("train",)
VALID_NAMES = ("valid", "val", "test", "holdout", "eval")


def _split_pair(tables: list[Path]) -> tuple[Path, Path] | None:
    """The two files a split step wrote, by name: one training, one validation."""
    train = [p for p in tables if any(n in p.stem.lower() for n in TRAIN_NAMES)]
    valid = [p for p in tables if any(n in p.stem.lower() for n in VALID_NAMES) and p not in train]
    return (train[0], valid[0]) if train and valid else None


def _time_column(train: pd.DataFrame, valid: pd.DataFrame) -> str | None:
    """A column that reads as a date in both halves: then the halves have a time order to check."""
    for col in [c for c in train.columns if c in valid.columns]:
        if train[col].dtype.kind in "biufc":
            continue
        pair = [pd.to_datetime(frame[col], errors="coerce", format="mixed") for frame in (train, valid)]
        if all(times.notna().mean() > 0.9 for times in pair) and all(len(times) for times in pair):
            return col
    return None


def _split_checks(state: RunState, tables: list[Path]) -> list[Finding]:
    pair = _split_pair(tables)
    if pair is None:
        return []
    train, valid = (_read_table(p) for p in pair)
    if train is None or valid is None or train.empty or valid.empty:
        return []
    names = f"{pair[0].name} and {pair[1].name}"
    findings = [
        split_overlap(train, valid, id_column=state.task.id_column).model_copy(
            update={"detail": f"{names}: {split_overlap(train, valid, state.task.id_column).detail}"}
        )
    ]
    col = _time_column(train, valid)
    if col:
        times = [pd.to_datetime(frame[col], errors="coerce", format="mixed") for frame in (train, valid)]
        f = temporal_order(*times)
        findings.append(
            f.model_copy(
                update={
                    "detail": f"{names} by {col}: {f.detail}"
                    + (
                        ""
                        if f.passed
                        else ". A split that mixes the periods measures the wrong thing when "
                        "the task is judged on later data; it is fine when it is not."
                    )
                }
            )
        )
    return findings


def _row_reference(state: RunState, step: PlanStep) -> tuple[int, str, int] | None:
    """What a step's row count is compared with: the largest table from the closest earlier accepted step that
    wrote one, else the largest raw file. Comparing every step with the raw file blamed each later step again
    for a drop an earlier step had already made (and the Critic had accepted)."""
    earlier = [s for s in state.plan.steps[: [p.id for p in state.plan.steps].index(step.id)]]
    output = Path(state.workspace) / "output"
    for prior in reversed(earlier):
        record = state.steps.get(prior.id)
        if record is None or record.status != "accepted" or not record.attempts:
            continue
        tables = [
            output / a for a in record.attempts[-1].artifacts if a.endswith(".csv") and (output / a).exists()
        ]
        if tables:
            biggest = max(tables, key=lambda p: p.stat().st_size)
            df = _read_table(biggest)
            if df is not None:
                return len(df), f"{biggest.name} from {prior.id}", 0
    if not state.raw_rows:
        return None
    name = max(state.raw_rows, key=state.raw_rows.get)
    return state.raw_rows[name], "the largest raw file", state.raw_blank_rows.get(name, 0)


def step_checks(state: RunState, step: PlanStep, attempt: StepAttempt) -> list[Finding]:
    output = Path(state.workspace) / "output"
    findings: list[Finding] = []
    written = [a for a in attempt.artifacts if (output / a).exists()]
    if not written and step.kind not in ("profile", "analyze", "evaluate"):
        findings.append(
            Finding(tool="artifacts", passed=False, detail="the step wrote no files for later steps")
        )
    target, id_column = state.plan.target_column, state.task.id_column
    reference = _row_reference(state, step)

    tables = sorted((a for a in written if a.endswith(".csv")), key=lambda a: -(output / a).stat().st_size)
    for name in tables[:MAX_TABLES]:
        df = _read_table(output / name)
        if df is None:
            findings.append(Finding(tool="read_table", passed=False, detail=f"{name} is not a readable CSV"))
            continue
        f = numbers_stored_as_text(df)
        findings.append(f.model_copy(update={"detail": f"{name}: {f.detail}"}))
        if step.kind in ("clean", "feature"):
            f = missing_and_sentinels(df)
            findings.append(f.model_copy(update={"detail": f"{name}: {f.detail}"}))
            if reference:
                rows, label, blank = reference
                f = row_accounting(rows, len(df), max_drop=0.05, blank_rows=blank)
                findings.append(f.model_copy(update={"detail": f"{name} vs {label}: {f.detail}"}))
            if id_column and state.task.kind == "predictive":
                f = required_columns(df, [id_column])
                findings.append(f.model_copy(update={"detail": f"{name}: {f.detail}"}))
        if target and target in df.columns and step.kind in ("clean", "feature", "split", "train"):
            f = target_leakage(df, target, exclude=[c for c in [id_column] if c])
            findings.append(f.model_copy(update={"detail": f"{name}: {f.detail}"}))

    if step.kind == "split":
        findings += _split_checks(state, [output / a for a in written if a.endswith(".csv")])

    if "metrics.json" in written:
        try:
            metrics = json.loads((output / "metrics.json").read_text())
            for key, value in metrics.items():
                if key.endswith("roc_auc") and isinstance(value, int | float):
                    findings.append(metric_sanity("roc_auc", float(value), 0.5))
        except (ValueError, OSError) as e:
            findings.append(
                Finding(tool="metrics_json", passed=False, detail=f"metrics.json unreadable: {e}")
            )
    if "answer.json" in written:
        try:
            json.loads((output / "answer.json").read_text())
        except (ValueError, OSError) as e:
            findings.append(
                Finding(tool="answer_json", passed=False, detail=f"answer.json is not valid JSON: {e}")
            )
    return findings


# --- delivery --------------------------------------------------------------------------------

PIPELINE = '''"""The accepted steps, in order. Assembled by the orchestrator."""

import runpy

STEPS = {steps!r}

for step in STEPS:
    print(f"== {{step}}", flush=True)
    runpy.run_path(f"output/{{step}}", run_name="__main__")
'''


def assemble_pipeline(state: RunState, files: list[str]) -> None:
    (Path(state.workspace) / "output" / "pipeline.py").write_text(PIPELINE.format(steps=files))


def _smoke_features(state: RunState, fresh: Path, target_column: str | None) -> Path | None:
    """A slice of the data file that holds the id column, in its raw format: header lines plus 200 rows.

    The target column is dropped when the file has one ordinary header row; otherwise (no header, two header
    rows) the raw lines are kept, target included. A smoke test that predict.py runs, not a score.
    """
    data = Path(state.workspace) / "data"
    files = [
        p
        for p in data.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".csv", ".data", ".txt"}
        and not p.name.lower().startswith("readme")
    ]
    if not files:
        return None
    id_column = state.task.id_column or ""
    with_id = [
        p
        for p in files
        if id_column and re.search(rf"\b{re.escape(id_column)}\b", p.read_text(errors="replace")[:2000])
    ]
    source = (with_id or sorted(files, key=lambda p: -p.stat().st_size))[0]
    profile = profile_file(source)
    target_path = fresh / f"smoke_features{source.suffix}"
    if profile.header_rows == 1 and target_column:
        frame = pd.read_csv(source, sep=profile.separator, nrows=200, dtype=str, keep_default_na=False)
        if target_column in frame.columns:
            # Like the real holdout, the features must not contain the target.
            frame.drop(columns=[target_column]).to_csv(target_path, sep=profile.separator, index=False)
            return target_path
    lines = source.read_text(errors="replace").splitlines(keepends=True)
    target_path.write_text("".join(lines[: profile.header_rows + 200]))
    return target_path


def output_tail(result, limit: int = 600) -> str:
    """What a failed script said, wherever it said it: scripts often print their error and exit(1)."""
    parts = [
        f"stderr: {result.stderr[-limit:]}" if result.stderr.strip() else "",
        f"stdout: {result.stdout[-limit:]}" if result.stdout.strip() else "",
    ]
    return " | ".join(p for p in parts if p) or "no output at all"


def _format_note(features: Path) -> str:
    """How the smoke input is laid out, so a failing predict.py can be fixed without guessing: the features
    come in the raw data format, which is often not what the training steps saved."""
    profile = profile_file(features)
    header = {0: "NO header row (columns in the data dictionary's order)", 1: "one header row"}.get(
        profile.header_rows, f"{profile.header_rows} header rows"
    )
    first = features.read_text(errors="replace").splitlines()[0][:160] if features.stat().st_size else ""
    return (
        f"The features file is in the raw data format: {header}, separator {profile.separator!r}, "
        f"no target column; first line: {first!r}"
    )


def _smoke_ids(features: Path, id_column: str | None) -> set[str] | None:
    """The id values in the smoke input, when the file can be read with an ordinary header; None otherwise."""
    if not id_column:
        return None
    profile = profile_file(features)
    if profile.header_rows != 1:
        return None
    frame = pd.read_csv(features, sep=profile.separator, dtype=str, keep_default_na=False)
    return set(frame[id_column].str.strip()) if id_column in frame.columns else None


def deliver_checks(state: RunState, runner, target_column: str | None = None) -> list[Finding]:
    """`target_column` defaults to the plan's; the single-agent baseline has no plan and passes its own."""
    workspace = Path(state.workspace)
    findings: list[Finding] = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "work"
        shutil.copytree(workspace / "data", fresh / "data")
        (fresh / "output").mkdir()
        for py in (workspace / "output").glob("*.py"):
            shutil.copy(py, fresh / "output" / py.name)
        rerun = runner.run(fresh, ["output/pipeline.py"])
        findings.append(
            Finding(
                tool="clean_rerun",
                passed=rerun.ok,
                detail="pipeline.py ran from a clean copy"
                if rerun.ok
                else f"pipeline.py failed from a clean copy ({rerun.reason}): {output_tail(rerun)}",
            )
        )
        if not rerun.ok:
            return findings

        if state.task.kind == "analytical":
            answer = fresh / "output" / "answer.json"
            ok = answer.exists()
            try:
                json.loads(answer.read_text()) if ok else None
            except ValueError:
                ok = False
            findings.append(
                Finding(
                    tool="deliverables",
                    passed=ok,
                    detail="output/answer.json written and valid"
                    if ok
                    else "output/answer.json missing or not valid JSON after the clean re-run",
                )
            )
            return findings

        missing = [f for f in ("predict.py", "metrics.json") if not (fresh / "output" / f).exists()]
        metric_key = f"validation_{state.task.metric}" if state.task.metric else None
        if not missing and metric_key:
            try:
                if metric_key not in json.loads((fresh / "output" / "metrics.json").read_text()):
                    missing.append(f"metrics.json key {metric_key}")
            except ValueError:
                missing.append("valid metrics.json")
        findings.append(
            Finding(
                tool="deliverables",
                passed=not missing,
                detail=f"missing after the clean re-run: {missing}"
                if missing
                else "predict.py and metrics.json written",
            )
        )
        if missing:
            return findings

        target = target_column or (state.plan.target_column if state.plan else None)
        features = _smoke_features(state, fresh, target)
        if features is None:
            return findings
        smoke = runner.run(fresh, ["output/predict.py", features.name, "smoke_predictions.csv"])
        predictions = fresh / "smoke_predictions.csv"
        if not smoke.ok or not predictions.exists():
            findings.append(
                Finding(
                    tool="predict_smoke",
                    passed=False,
                    detail=f"predict.py failed on 200 rows of {features.name} ({smoke.reason}): "
                    f"{output_tail(smoke)}. {_format_note(features)}",
                )
            )
            return findings
        frame = _read_table(predictions)
        needed = {state.task.id_column or "", "prediction"} - {""}
        problems = []
        if frame is None or not needed <= set(frame.columns) or len(frame) == 0:
            columns = [] if frame is None else list(frame.columns)
            problems.append(
                f"predict.py wrote {0 if frame is None else len(frame)} rows with columns {columns}; "
                f"need {sorted(needed)}"
            )
        else:
            expected = _smoke_ids(features, state.task.id_column)
            if expected is not None and set(frame[state.task.id_column].astype(str)) != expected:
                shown = frame[state.task.id_column].astype(str).head(3).tolist()
                problems.append(
                    f"the ids in the predictions ({shown}...) are not the ids in the input file: read with "
                    "the wrong separator or header, or row numbers used instead of the id column?"
                )
            values = pd.to_numeric(frame["prediction"], errors="coerce")
            if len(frame) > 20 and values.nunique() <= 1:
                problems.append(
                    f"all {len(frame)} predictions are identical ({values.iloc[0]}): the features did not "
                    "reach the model (parsing, column names, or a zero-filled reindex)"
                )
        findings.append(
            Finding(
                tool="predict_smoke",
                passed=not problems,
                detail="; ".join(problems) + f". {_format_note(features)}"
                if problems
                else f"{len(frame)} varied predictions, one per id",
            )
        )
    return findings
