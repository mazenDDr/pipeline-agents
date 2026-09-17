"""Summarize a probe run: one row per (model, mode), plus the per-task code matrix.

    python -m pipeline_agents.probe.report outputs/runs/<run_id>

Writes summary.json and report.md into the run dir. Intervals are 95% percentile bootstraps over
tasks (samples of a task are averaged first), so a model is not rewarded for many samples of one
easy task.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap_mean(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), (n_boot, len(arr)))].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _rate(rows: list[dict], field: str) -> float | None:
    return sum(r[field] for r in rows) / len(rows) if rows else None


def summarize(run_dir: Path) -> dict:
    lines = (run_dir / "results.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    servers = {}
    servers_path = run_dir / "servers.jsonl"
    if servers_path.exists():
        for line in servers_path.read_text().splitlines():
            s = json.loads(line)
            servers[(s["model"], s["mode"])] = s  # the last load wins (resumed runs reload)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        model, mode = r["key"].split("|")[:2]
        groups[(model, mode)].append(r)

    out = []
    code_matrix: dict[str, dict[str, float]] = {}
    for (model, mode), rs in groups.items():
        speed = [r for r in rs if r["phase"] == "speed" and not r["error"]]
        plans = [r for r in rs if r["phase"] == "plan"]
        code = [r for r in rs if r["phase"] == "code"]
        per_task: dict[str, list[bool]] = defaultdict(list)
        for r in code:
            per_task[r["task"]].append(r["passed"])
        task_rates = {t: sum(v) / len(v) for t, v in per_task.items()}
        code_matrix[f"{model} ({mode})"] = task_rates
        mean, lo, hi = bootstrap_mean(list(task_rates.values()))
        llm_rows = plans + code
        server = servers.get((model, mode), {})
        out.append(
            {
                "model": model,
                "mode": mode,
                "load_s": server.get("load_s"),
                "vram_mib": (server["peak_vram_mib"] - server["idle_vram_mib"])
                if "peak_vram_mib" in server
                else None,
                "rss_mib": server.get("peak_rss_mib"),
                "ttft_6k_s": statistics.median(r["ttft_s"] for r in speed) if speed else None,
                "prefill_tps": statistics.median(r["prompt_tps"] for r in speed) if speed else None,
                "decode_tps": statistics.median(r["decode_tps"] for r in speed) if speed else None,
                "plan_valid_prompt": _rate([r for r in plans if r["style"] == "prompt"], "valid"),
                "plan_sound_prompt": _rate([r for r in plans if r["style"] == "prompt"], "sound"),
                "plan_valid_enforced": _rate([r for r in plans if r["style"] == "enforced"], "valid"),
                "plan_sound_enforced": _rate([r for r in plans if r["style"] == "enforced"], "sound"),
                "n_plans": len(plans),
                "code_pass": mean,
                "code_pass_ci": [lo, hi],
                "n_code_tasks": len(task_rates),
                "n_code_samples": len(code),
                "mean_call_s": statistics.mean(r["total_s"] for r in llm_rows) if llm_rows else None,
                "mean_completion_tokens": statistics.mean(r["completion_tokens"] or 0 for r in llm_rows)
                if llm_rows
                else None,
                "truncated": sum(r["finish_reason"] == "length" for r in llm_rows),
                "call_errors": sum(bool(r["error"]) for r in llm_rows),
            }
        )
    return {"rows": out, "code_matrix": code_matrix}


def _fmt(v, spec: str = ".2f") -> str:
    return "–" if v is None else format(v, spec)


def render(summary: dict) -> str:
    lines = [
        "| model | mode | VRAM GiB | RSS GiB | TTFT 6k s | prefill tok/s | decode tok/s | plan valid "
        "(prompt / enforced) | plan sound (prompt / enforced) | code pass [95% CI] | s / call "
        "| tokens / call | truncated |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = sorted(summary["rows"], key=lambda r: (-(r["code_pass"] or 0), r["model"], r["mode"]))
    for r in ranked:
        if r["n_code_tasks"] == 0:
            continue  # skipped modes (see skip_modes in the run config) have speed rows only
        vram = None if r["vram_mib"] is None else r["vram_mib"] / 1024
        rss = None if r["rss_mib"] is None else r["rss_mib"] / 1024
        lo, hi = r["code_pass_ci"]
        lines.append(
            f"| {r['model']} | {r['mode']} | {_fmt(vram, '.1f')} | {_fmt(rss, '.1f')} "
            f"| {_fmt(r['ttft_6k_s'])} "
            f"| {_fmt(r['prefill_tps'], '.0f')} | {_fmt(r['decode_tps'], '.0f')} "
            f"| {_fmt(r['plan_valid_prompt'])} / {_fmt(r['plan_valid_enforced'])} "
            f"| {_fmt(r['plan_sound_prompt'])} / {_fmt(r['plan_sound_enforced'])} "
            f"| {_fmt(r['code_pass'])} [{_fmt(lo)}, {_fmt(hi)}] (n={r['n_code_tasks']}) "
            f"| {_fmt(r['mean_call_s'], '.1f')} | {_fmt(r['mean_completion_tokens'], '.0f')} "
            f"| {r['truncated']} |"
        )
    tasks = sorted({t for m in summary["code_matrix"].values() for t in m})
    lines += [
        "",
        "Code pass rate per task:",
        "",
        "| model / mode | " + " | ".join(tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    for name, rates in sorted(summary["code_matrix"].items()):
        lines.append(f"| {name} | " + " | ".join(_fmt(rates.get(t), ".1f") for t in tasks) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    run_dir = Path(sys.argv[1])
    summary = summarize(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "report.md").write_text(render(summary))
    print(render(summary))


if __name__ == "__main__":
    main()
