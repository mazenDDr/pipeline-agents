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
    target_leakage,
)

MAX_TABLES = 3


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:  # an unreadable table is itself worth a finding, not a crash
        return None


def step_checks(state: RunState, step: PlanStep, attempt: StepAttempt) -> list[Finding]:
    output = Path(state.workspace) / "output"
    findings: list[Finding] = []
    written = [a for a in attempt.artifacts if (output / a).exists()]
    if not written and step.kind not in ("profile", "analyze", "evaluate"):
        findings.append(
            Finding(tool="artifacts", passed=False, detail="the step wrote no files for later steps")
        )
    target, id_column = state.plan.target_column, state.task.id_column
    largest_raw = max(state.raw_rows.values(), default=0)

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
            if largest_raw:
                f = row_accounting(largest_raw, len(df), max_drop=0.05)
                findings.append(
                    f.model_copy(update={"detail": f"{name} vs the largest raw file: {f.detail}"})
                )
            if id_column and state.task.kind == "predictive":
                f = required_columns(df, [id_column])
                findings.append(f.model_copy(update={"detail": f"{name}: {f.detail}"}))
        if target and target in df.columns and step.kind in ("clean", "feature", "split", "train"):
            f = target_leakage(df, target, exclude=[c for c in [id_column] if c])
            findings.append(f.model_copy(update={"detail": f"{name}: {f.detail}"}))

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


def _smoke_features(state: RunState, fresh: Path) -> Path | None:
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
    target_column = state.plan.target_column if state.plan else None
    if profile.header_rows == 1 and target_column:
        frame = pd.read_csv(source, sep=profile.separator, nrows=200, dtype=str, keep_default_na=False)
        if target_column in frame.columns:
            # Like the real holdout, the features must not contain the target.
            frame.drop(columns=[target_column]).to_csv(target_path, sep=profile.separator, index=False)
            return target_path
    lines = source.read_text(errors="replace").splitlines(keepends=True)
    target_path.write_text("".join(lines[: profile.header_rows + 200]))
    return target_path


def _output_tail(result, limit: int = 600) -> str:
    """What a failed script said, wherever it said it: scripts often print their error and exit(1)."""
    parts = [
        f"stderr: {result.stderr[-limit:]}" if result.stderr.strip() else "",
        f"stdout: {result.stdout[-limit:]}" if result.stdout.strip() else "",
    ]
    return " | ".join(p for p in parts if p) or "no output at all"


def deliver_checks(state: RunState, runner) -> list[Finding]:
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
                else f"pipeline.py failed from a clean copy ({rerun.reason}): {_output_tail(rerun)}",
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

        features = _smoke_features(state, fresh)
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
                    f"{_output_tail(smoke)}",
                )
            )
            return findings
        frame = _read_table(predictions)
        needed = {state.task.id_column or "", "prediction"} - {""}
        ok = frame is not None and needed <= set(frame.columns) and len(frame) > 0
        findings.append(
            Finding(
                tool="predict_smoke",
                passed=ok,
                detail=f"predict.py wrote {0 if frame is None else len(frame)} rows with columns "
                f"{[] if frame is None else list(frame.columns)}; need {sorted(needed)}",
            )
        )
    return findings
