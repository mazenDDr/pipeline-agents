"""Run both systems on datasets the benchmark has never seen, and score them (T16).

    python scripts/field_test.py --download                 # once, on the GPU machine
    python scripts/field_test.py --reference                # print the answers this script will score against
    python scripts/field_test.py --tasks wine-quality --systems multi baseline

Each task gets its own workspace built the way the app builds one for an upload: the goal, the data
dictionary and the data, nothing else. Analytical answers are compared with a reference computed here;
the predictive task is scored by running the delivered predict.py on rows held back from the agents.
"""

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

from pipeline_agents.agents.baseline import run_baseline
from pipeline_agents.agents.common import Deps
from pipeline_agents.app.user_run import UserTask, prepare
from pipeline_agents.bench.field import (
    BY_ID,
    RAW,
    REFERENCES,
    TASKS,
    FieldTask,
    load_answer,
    raw_frame,
    score_analytical,
    score_predictions,
    split,
    unzip,
)
from pipeline_agents.config import build_observer, load_run_config
from pipeline_agents.graph.build import run
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.sandbox.runner import default_runner
from pipeline_agents.schemas import Ledger

OUT = Path("outputs/field")
DATA_NAME = {"wine-quality": "wine.csv", "student-grades": "students.csv", "household-power": "power.csv"}


def download(task: FieldTask) -> None:
    dest = RAW / task.id
    if any(dest.rglob(task.member)):
        print(f"{task.id}: already downloaded")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "archive.zip"
    print(f"{task.id}: downloading {task.url}")
    urllib.request.urlretrieve(task.url, archive)
    unzip(archive, dest)
    archive.unlink()
    print(f"{task.id}: {next(dest.rglob(task.member))}")


def build_task(task: FieldTask) -> tuple[UserTask, pd.DataFrame]:
    """The upload the agents get, and the rows held back (empty for an analytical task)."""
    frame = raw_frame(task)
    seen, holdout = split(task, frame)
    name = DATA_NAME[task.id]
    text = seen.to_csv(index=False, sep=task.separator)
    upload = UserTask(
        goal=task.goal,
        kind=task.kind,
        files={name: text.encode()},
        data_readme=task.readme,
        target=task.target,
        id_column=task.id_column,
        metric=task.metric,
    )
    return upload, holdout


def score(task: FieldTask, workspace: Path, holdout: pd.DataFrame, runner) -> dict:
    if task.kind == "analytical":
        answer = load_answer(workspace)
        if answer is None:
            return {"passed": False, "detail": "no valid output/answer.json"}
        reference = REFERENCES[task.id](raw_frame(task))
        passed, problems = score_analytical(task, answer, reference)
        return {"passed": passed, "detail": "; ".join(problems) or "every number matches the reference"}
    features = holdout.drop(columns=[task.target])
    scratch = workspace.parent / "score"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "output").mkdir(parents=True)
    shutil.copytree(workspace / "data", scratch / "data")
    for py in (workspace / "output").glob("*"):
        if py.is_file():
            shutil.copy(py, scratch / "output" / py.name)
    features.to_csv(scratch / "holdout_features.csv", index=False, sep=task.separator)
    result = runner.run(scratch, ["output/predict.py", "holdout_features.csv", "predictions.csv"])
    if not result.ok or not (scratch / "predictions.csv").exists():
        return {"passed": False, "detail": f"predict.py failed ({result.reason}): {result.stderr[-300:]}"}
    predictions = pd.read_csv(scratch / "predictions.csv")
    mae, detail = score_predictions(task, predictions, holdout)
    passed = bool(mae == mae and task.threshold is not None and mae <= task.threshold)
    return {"passed": passed, "detail": detail, "metric": None if mae != mae else mae}


def run_one(task: FieldTask, system: str, cfg, root: Path) -> dict:
    upload, holdout = build_task(task)
    run_dir = root / task.id / system
    shutil.rmtree(run_dir, ignore_errors=True)
    state = prepare(upload, run_dir)
    state = state.model_copy(update={"ledger": Ledger(cap_usd=cfg.budget_usd, degrade_at=cfg.degrade_at)})
    observer = build_observer(cfg, f"{system}_{task.id}", run_dir, cache_dir=run_dir / "cache")
    deps = Deps(observer=observer, registry=PromptRegistry("prompts"), config=cfg, runner=default_runner())
    workspace = Path(state.workspace)
    start = time.perf_counter()
    if system == "baseline":
        final = run_baseline(deps, state.task, workspace, state.run_id)
    else:
        final = run(deps, state, run_dir / "state.sqlite")
    seconds = time.perf_counter() - start
    outcome = score(task, workspace, holdout, deps.runner)
    result = {
        "task": task.id,
        "system": system,
        "kind": task.kind,
        "status": final.status,
        "stop_reason": final.stop_reason,
        "seconds": round(seconds, 1),
        "model_calls": final.model_calls,
        "shadow_usd": round(final.ledger.total.shadow_usd, 5),
        "score": outcome,
        "success": bool(outcome["passed"]) and final.status == "succeeded",
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"{task.id:16s} {system:9s} {result['status']:10s} success={result['success']} "
        f"({result['seconds']:.0f}s, {result['model_calls']} calls) {outcome['detail'][:90]}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--tasks", nargs="*", default=[t.id for t in TASKS])
    parser.add_argument("--systems", nargs="*", default=["multi", "baseline"])
    parser.add_argument("--config", default="configs/run/default.yaml")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    tasks = [BY_ID[t] for t in args.tasks]
    if args.download:
        for task in tasks:
            download(task)
        return
    if args.reference:
        for task in tasks:
            if task.id in REFERENCES:
                print(task.id, json.dumps(REFERENCES[task.id](raw_frame(task)), indent=1)[:1200])
            else:
                seen, holdout = split(task, raw_frame(task))
                print(f"{task.id}: {len(seen):,} rows given, {len(holdout):,} held back")
        return
    cfg = load_run_config(args.config).model_copy(update={"seed": args.seed})
    results = [run_one(task, system, cfg, OUT) for task in tasks for system in args.systems]
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"{sum(r['success'] for r in results)} of {len(results)} runs passed")


if __name__ == "__main__":
    main()
