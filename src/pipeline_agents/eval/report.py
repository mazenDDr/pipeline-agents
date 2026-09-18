"""Summarise a grid (T9): per arm, per task, and paired comparisons between arms.

    python -m pipeline_agents.eval.report outputs/runs/<grid> [outputs/runs/<earlier grid>:<prefix> ...]

Writes summary.json and report.md into the grid directory. Error cells count as failures and are listed.
Arms of an earlier grid on the same tasks and seeds can be added under a prefix (`t9-dev:t9-` turns its
`multi` arm into `t9-multi`) to pair a changed system with the version before the change.
"""

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from pipeline_agents.eval.stats import is_real, nested_bootstrap, paired_difference

METRICS = [
    ("success", "success rate"),
    ("revisions", "revisions"),
    ("model_calls", "model calls"),
    ("shadow_usd", "shadow $"),
    ("seconds", "seconds"),
]


def load_results(grid_dir: Path, prefix: str = "") -> list[dict]:
    results = [json.loads(p.read_text()) for p in sorted(grid_dir.glob("*/result.json"))]
    return [{**r, "arm": prefix + r["run_id"].split("_", 1)[0]} for r in results]


def _label(result: dict) -> str:
    return result.get("arm") or result["run_id"].split("_", 1)[0]


def _by_task(results: list[dict], metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.get(metric) is not None:
            out[r["task"]].append(float(r[metric]))
    return out


def summarize(results: list[dict]) -> dict:
    arms = sorted({_label(r) for r in results}, key=lambda a: [_label(r) for r in results].index(a))
    summary: dict = {"arms": {}, "tasks": {}, "comparisons": [], "errors": []}
    for arm in arms:
        rows = [r for r in results if _label(r) == arm]
        entry = {
            "cells": len(rows),
            "tasks": len({r["task"] for r in rows}),
            "status": dict(Counter(r["status"] for r in rows)),
        }
        for metric, _ in METRICS:
            mean, lo, hi = nested_bootstrap(_by_task(rows, metric))
            entry[metric] = {"mean": mean, "ci": [lo, hi]}
        summary["arms"][arm] = entry
    for task in dict.fromkeys(r["task"] for r in results):
        summary["tasks"][task] = {
            arm: {
                "success": sum(r["success"] for r in results if r["task"] == task and _label(r) == arm),
                "cells": sum(1 for r in results if r["task"] == task and _label(r) == arm),
                "failures": dict(
                    Counter(
                        (r.get("checker") or {}).get("first_failure") or r["status"]
                        for r in results
                        if r["task"] == task and _label(r) == arm and not r["success"]
                    )
                ),
            }
            for arm in arms
        }
    for a, b in combinations(arms, 2):
        for metric, _ in METRICS:
            cells = {arm: defaultdict(dict) for arm in (a, b)}
            for r in results:
                if _label(r) in cells and r.get(metric) is not None:
                    cells[_label(r)][r["task"]][r["seed"]] = float(r[metric])
            mean, lo, hi, n = paired_difference(cells[a], cells[b])
            summary["comparisons"].append(
                {
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "difference": mean,
                    "ci": [lo, hi],
                    "pairs": n,
                    "real": is_real(lo, hi),
                }
            )
    summary["errors"] = [
        {"run_id": r["run_id"], "reason": r["stop_reason"]} for r in results if r["status"] == "error"
    ]
    return summary


def _fmt(metric: str, value: float) -> str:
    if metric == "success":
        return f"{value:.2f}"
    if metric == "shadow_usd":
        return f"{value:.4f}"
    return f"{value:.1f}"


def render(summary: dict, grid_name: str) -> str:
    lines = [
        f"# {grid_name}",
        "",
        "Means over tasks of the mean over seeds; 95% intervals from a nested bootstrap (tasks, then seeds). "
        "Error cells count as failures.",
        "",
        "| arm | cells | " + " | ".join(name for _, name in METRICS) + " | status |",
        "|---|---|" + "---|" * len(METRICS) + "---|",
    ]
    for arm, e in summary["arms"].items():
        cols = [
            f"{_fmt(m, e[m]['mean'])} [{_fmt(m, e[m]['ci'][0])}, {_fmt(m, e[m]['ci'][1])}]"
            for m, _ in METRICS
        ]
        status = ", ".join(f"{k} {v}" for k, v in e["status"].items())
        lines.append(f"| {arm} | {e['cells']} | " + " | ".join(cols) + f" | {status} |")
    arms = list(summary["arms"])
    lines += [
        "",
        "## Per task (successes / cells, and how the others failed)",
        "",
        "| task | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for task, by_arm in summary["tasks"].items():
        cells = []
        for arm in arms:
            e = by_arm[arm]
            why = ", ".join(f"{k} {v}" for k, v in e["failures"].items())
            cells.append(f"{e['success']}/{e['cells']}" + (f" ({why})" if why else ""))
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired differences (a - b, same task and seed)",
        "",
        "| a | b | metric | difference [95% CI] | pairs | real? |",
        "|---|---|---|---|---|---|",
    ]
    for c in summary["comparisons"]:
        m = c["metric"]
        lines.append(
            f"| {c['a']} | {c['b']} | {m} | {_fmt(m, c['difference'])} [{_fmt(m, c['ci'][0])}, "
            f"{_fmt(m, c['ci'][1])}] | {c['pairs']} | {'yes' if c['real'] else 'no'} |"
        )
    if summary["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{e['run_id']}`: {e['reason']}" for e in summary["errors"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    grid_dir = Path(sys.argv[1])
    results = load_results(grid_dir)
    for extra in sys.argv[2:]:
        path, _, prefix = extra.partition(":")
        results += load_results(Path(path), prefix)
    summary = summarize(results)
    (grid_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (grid_dir / "report.md").write_text(render(summary, grid_dir.name))
    print(render(summary, grid_dir.name))


if __name__ == "__main__":
    main()
