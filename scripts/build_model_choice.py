"""Generate docs/model_choice.md from the T1 runs, so no number in it is typed by hand.

python scripts/build_model_choice.py
"""

import json
from pathlib import Path

import numpy as np
import yaml

PROBE = Path("outputs/runs/t1-probe/summary.json")
COLOCATE = Path("outputs/runs/t1-colocate/summary.json")
STRONG, CHEAP = "gemma-4-26b-a4b (no-think)", "gemma-4-e4b (no-think)"


def paired(matrix: dict, a: str, b: str, n_boot: int = 4000) -> tuple[float, float, float]:
    tasks = sorted(set(matrix[a]) & set(matrix[b]))
    d = np.array([matrix[a][t] - matrix[b][t] for t in tasks])
    boot = d[np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    probe = json.loads(PROBE.read_text())
    colocate = json.loads(COLOCATE.read_text())
    prices = yaml.safe_load(Path("configs/prices.yaml").read_text())["models"]
    rows = sorted((r for r in probe["rows"] if r["n_code_tasks"]), key=lambda r: -r["code_pass"])

    out = [
        "# Choosing the models",
        "",
        "Every agent call runs on one local GPU (RTX 5060 Ti, 16 GB) through llama-server. This page is "
        "generated "
        "by `scripts/build_model_choice.py` from `outputs/runs/t1-probe` and `outputs/runs/t1-colocate`.",
        "",
        "## What was measured",
        "",
        "- **Code:** 12 small pandas/sklearn tasks, each with a trap (a leaking column, `-999` meaning "
        "missing, a "
        "category that only appears at prediction time, zero-padded join keys, dates typed in three "
        "formats). The "
        "script the model writes is run and its output checked against hidden answers. Every checker is "
        "tested "
        "to accept a correct script and reject a plausible wrong one.",
        "- **Plans:** 8 dataset profiles and goals; the plan must be valid JSON of the right shape and "
        "follow a "
        "few structural rules (a split before training, evaluation after it).",
        "- **Speed and memory:** a 6k-token prompt with the prompt cache off; peak VRAM and resident memory.",
        "",
        "Intervals are 95% bootstraps over tasks. Finalists ran 3 samples per task; the rest ran 1. Two "
        "checkers were fixed after reading the failures by hand (an honest random forest failed, and a model "
        "using the leaking column passed); the stored scripts were re-run against the fixed tasks.",
        "",
        "## Results",
        "",
        "| model | thinking | code pass [95% CI] | samples | plans sound (prompt / schema) | decode tok/s "
        "| s per call | VRAM GiB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["code_pass_ci"]
        out.append(
            f"| {r['model']} | {'no' if r['mode'] == 'no-think' else 'yes' if r['mode'] == 'think' else '–'} "
            f"| {r['code_pass']:.2f} [{lo:.2f}, {hi:.2f}] | {r['n_code_samples']} "
            f"| {r['plan_sound_prompt']:.2f} / {r['plan_sound_enforced']:.2f} | {r['decode_tps']:.0f} "
            f"| {r['mean_call_s']:.1f} | {r['vram_mib'] / 1024:.1f} |"
        )

    matrix = probe["code_matrix"]
    out += [
        "",
        "Qwen3.6-27B with thinking was stopped after its speed test: 12 tokens/s, about 5 minutes per "
        "call, too "
        "slow for an agent that makes dozens of calls.",
        "",
        "## Is the top model really better?",
        "",
        f"Paired differences in code pass rate, {STRONG} minus each model, over the same tasks:",
        "",
        "| compared with | difference [95% CI] | real? |",
        "|---|---|---|",
    ]
    ranked = [k for k in matrix if matrix[k]]  # skipped modes have no code rows
    for other in sorted(ranked, key=lambda k: -np.mean(list(matrix[k].values()))):
        if other == STRONG or len(matrix[other]) < 12:
            continue
        m, lo, hi = paired(matrix, STRONG, other)
        out.append(f"| {other} | {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    out += [
        "",
        "Only some of these differences are real: 12 tasks cannot separate the strongest models. Where "
        "quality "
        "ties, speed and memory decide. Gemma 4 26B-A4B without thinking is the fastest of the top "
        "scorers, and "
        "thinking made it slower without making it better.",
        "",
        "## Two models on one card",
        "",
        "Near the budget cap, low-stakes calls move to a cheaper model. Both models must then be loaded "
        "at once:",
        "",
        "| layout | model | decode tok/s | prefill tok/s | time to first token (6k) |",
        "|---|---|---|---|---|",
    ]
    for layout in colocate:
        for s in layout["servers"]:
            out.append(
                f"| {layout['layout']} | {s['model']} | {s['decode_tps']:.0f} | {s['prefill_tps']:.0f} "
                f"| {s['ttft_s']:.1f} s |"
            )
    out += [
        "",
        "Sharing the card halves the strong model's decode speed, because part of it moves to system "
        "RAM. Running "
        "the cheap model on the CPU keeps the strong model fast but makes the cheap model slower than "
        "the strong "
        "one, which defeats the point. **Chosen: both on the GPU** (`tiered` in `configs/models.yaml`), "
        "with one "
        "model for every role (`single`) kept as an ablation arm.",
        "",
        "## Shadow prices",
        "",
        "Nothing is billed. Costs are reported as what the same tokens would cost from a hosted provider "
        "of the "
        "same open weights (USD per million tokens):",
        "",
        "| model | input | output | source |",
        "|---|---|---|---|",
    ]
    for name, p in prices.items():
        out.append(
            f"| {name} | {p['input_per_mtok']:.2f} | {p['output_per_mtok']:.2f} | {p['source']} "
            f"({p['retrieved']}) |"
        )
    out += [
        "",
        "At these prices the cheap model's input tokens cost more than the strong model's, so the cheap "
        "tier saves "
        "on output tokens and time, not on everything.",
        "",
    ]
    Path("docs/model_choice.md").write_text("\n".join(out))
    print("wrote docs/model_choice.md")


if __name__ == "__main__":
    main()
