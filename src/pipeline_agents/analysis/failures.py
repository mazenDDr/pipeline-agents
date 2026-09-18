"""Put every failed run in one class (T12).

A run fails in one of a few ways, and the way is readable from what the run recorded: its stop reason, the
checker stage that failed first, and (for a wrong answer) which key of answer.json was wrong. Answer keys are
mapped to a diagnosis by hand once per key, in `answer_cases.json`, because only reading the code tells you
why a number came out wrong.

    python -m pipeline_agents.analysis.failures outputs/runs/<grid> [...] --cases outputs/trust/t12
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pipeline_agents.bench.families import TASKS

METRICS = {t.id: (t.predictive.metric if t.predictive else None) for t in TASKS}

CATEGORIES = {
    "memory_limit": "the step was killed for using more than the sandbox's memory limit",
    "parse_failure": "a role's reply could not be parsed, twice",
    "loop_exhausted_delivery": "revisions and the re-plan went on the deliverables, usually predict.py",
    "loop_exhausted_tool": "revisions and the re-plan went on a validation tool finding",
    "loop_exhausted_crash": "revisions and the re-plan went on a script that kept crashing",
    "loop_exhausted_plan": "the Reviser sent the run back to the Planner, and the new plan did not help",
    "loop_exhausted_other": "revisions and the re-plan ran out for another reason",
    "repair_exhausted": "the single-agent baseline used all its repair attempts",
    "below_threshold": "delivered, but the holdout score misses the task's threshold",
    "dishonest_estimate": "the score passed, but the run's own estimate was far from the holdout",
    "wrong_answer": "delivered an analytical answer with a wrong number",
    "data_deleted_as_hygiene": "a cleaning habit deleted real rows, and the answer moved",
    "unit_ignored": "a documented unit or code was used raw",
    "benchmark_defect": "the task text and the hidden answer disagree: the benchmark is wrong",
    "other": "none of the above",
}

DELIVERY_CHECKS = ("predict_smoke", "clean_rerun", "deliverables", "delivery check", "predict.py")
TOOLS = (
    "row_accounting",
    "split_overlap",
    "temporal_order",
    "missing_and_sentinels",
    "required_columns",
    "target_leakage",
    "numbers_stored_as_text",
    "validation tool",
)


@dataclass(frozen=True)
class Failure:
    run_id: str
    grid: str
    arm: str
    task: str
    category: str
    detail: str  # the evidence the category was read from
    diagnosis: str = ""  # for wrong answers: what the hand-read code was doing


def _first_failed(result: dict) -> dict | None:
    stages = (result.get("checker") or {}).get("stages", [])
    return next((s for s in stages if not s["passed"]), None)


def _stage(result: dict, name: str) -> dict | None:
    return next((s for s in (result.get("checker") or {}).get("stages", []) if s["name"] == name), None)


def _estimate(result: dict) -> str:
    """Whether the run's own validation estimate flattered it, once the metric's direction is applied."""
    checker = result.get("checker") or {}
    own, holdout = checker.get("validation_metric"), checker.get("holdout_metric")
    if own is None or holdout is None:
        return "no estimate recorded"
    lower_is_better = METRICS.get(result["task"]) != "roc_auc"
    better = own < holdout if lower_is_better else own > holdout
    if abs(own - holdout) <= 1e-9:
        return "its own estimate matched the holdout"
    return (
        f"its own estimate ({own:.4g}) was {'better' if better else 'worse'} than the holdout ({holdout:.4g})"
    )


def classify(result: dict, cases: dict[str, dict] | None = None) -> Failure:
    """`cases` maps "<task>/<answer key>" to {"category", "diagnosis"} for hand-read wrong answers."""
    reason = result.get("stop_reason") or ""
    failed = _first_failed(result)
    arm, grid = result["run_id"].split("_", 1)[0], result.get("grid", "")
    base = {"run_id": result["run_id"], "grid": grid, "arm": arm, "task": result["task"]}
    if result["status"] == "failed":
        if reason.startswith("infra"):
            return Failure(**base, category="memory_limit", detail=reason.split("(")[0].strip())
        if "unusable after a re-ask" in reason:
            return Failure(**base, category="parse_failure", detail=reason)
        if "no passing attempt" in reason:
            return Failure(**base, category="repair_exhausted", detail=reason)
    if result["status"] == "needs_human":
        tail = (reason.split(":", 1)[1] if ":" in reason else reason).strip()[:300]
        if any(k in tail for k in DELIVERY_CHECKS):
            return Failure(**base, category="loop_exhausted_delivery", detail=tail)
        if "the script crashed" in tail:
            return Failure(**base, category="loop_exhausted_crash", detail=tail)
        if any(k in tail for k in TOOLS):
            return Failure(**base, category="loop_exhausted_tool", detail=tail)
        if any(k in tail for k in ("Planner", "plan", "sequence", "step order")):
            return Failure(**base, category="loop_exhausted_plan", detail=tail)
        return Failure(**base, category="loop_exhausted_other", detail=tail)
    if failed and failed["name"] == "score":
        return Failure(**base, category="below_threshold", detail=f"{failed['detail']}; {_estimate(result)}")
    if failed and failed["name"] == "honest_estimate":
        return Failure(
            **base, category="dishonest_estimate", detail=f"{failed['detail']}; {_estimate(result)}"
        )
    if failed and failed["name"] == "answer":
        key = failed["detail"].split(":")[0].strip()
        case = (cases or {}).get(f"{result['task']}/{key}", {})
        return Failure(
            **base,
            category=case.get("category", "wrong_answer"),
            detail=failed["detail"][:200],
            diagnosis=case.get("diagnosis", ""),
        )
    if failed:
        return Failure(**base, category="other", detail=f"{failed['name']}: {failed['detail'][:200]}")
    return Failure(**base, category="other", detail=reason[:200])


def failures(grid_dirs: list[Path], cases: dict[str, dict] | None = None) -> list[Failure]:
    out = []
    for grid_dir in grid_dirs:
        for path in sorted(grid_dir.glob("*/result.json")):
            result = json.loads(path.read_text())
            if result["success"]:
                continue
            out.append(classify({**result, "grid": grid_dir.name}, cases))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grids", nargs="+")
    parser.add_argument("--cases", default="outputs/trust/t12")
    args = parser.parse_args()
    cases_path = Path(args.cases) / "answer_cases.json"
    cases = json.loads(cases_path.read_text()) if cases_path.exists() else {}
    found = failures([Path(g) for g in args.grids], cases)
    counts = Counter(f.category for f in found)
    for category, n in counts.most_common():
        print(f"{n:4d}  {category:26s} {CATEGORIES.get(category, '')}")
    print(f"{len(found):4d}  total failed runs")


if __name__ == "__main__":
    main()
