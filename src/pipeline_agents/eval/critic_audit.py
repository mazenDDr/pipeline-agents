"""Does the Critic agree with a careful reader? (T10)

Every Critic call in a grid's calls.jsonl becomes a case: the evidence the Critic saw (its user prompt) and
the decision it made. A stratified sample is labelled blind, without the decision, by answering one question:
"can later steps build on this step as it is?" (ok / not ok). Accept maps to ok; revise and escalate map to
not ok. Agreement is Cohen's kappa with a bootstrap interval over cases.

    python -m pipeline_agents.eval.critic_audit sample outputs/runs/t9-dev outputs/trust/t10
    python -m pipeline_agents.eval.critic_audit score outputs/trust/t10
"""

import json
import random
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from pipeline_agents.agents.common import extract_json
from pipeline_agents.eval.stats import N_BOOT

LABELS = ("ok", "not_ok")


class Case(BaseModel):
    case_id: str
    run_id: str
    task: str
    step_id: str
    iteration: int
    decision: str
    confidence: float
    issues: list[str]
    tool_failures: list[str]
    run_success: bool
    evidence: str

    @property
    def critic_label(self) -> str:
        return "ok" if self.decision == "accept" else "not_ok"

    @property
    def stratum(self) -> str:
        return f"{self.decision}/{'tool_fail' if self.tool_failures else 'tools_pass'}"


def load_cases(grid_dir: Path) -> list[Case]:
    cases = []
    for calls in sorted(grid_dir.glob("*/calls.jsonl")):
        result = json.loads((calls.parent / "result.json").read_text())
        n = 0
        for line in calls.read_text().splitlines():
            call = json.loads(line)
            if call["role"] != "critic" or call.get("error"):
                continue
            verdict = extract_json(call["content"])
            evidence = call["messages"][-1]["content"]
            n += 1
            cases.append(
                Case(
                    case_id=f"{result['run_id']}#{n}",
                    run_id=result["run_id"],
                    task=result["task"],
                    step_id=call["step_id"],
                    iteration=call["iteration"],
                    decision=verdict["decision"],
                    confidence=float(verdict["confidence"]),
                    issues=verdict.get("issues", []),
                    tool_failures=sorted(set(re.findall(r"^- \[FAIL\] (\w+)", evidence, re.M))),
                    run_success=result["success"],
                    evidence=evidence,
                )
            )
    return cases


def stratified_sample(cases: Sequence[Case], quotas: dict[str, int], seed: int = 0) -> list[Case]:
    """Up to `quotas[stratum]` random cases per stratum, returned in shuffled order."""
    rng = random.Random(seed)
    picked: list[Case] = []
    for stratum, quota in quotas.items():
        pool = [c for c in cases if c.stratum == stratum]
        picked += rng.sample(pool, min(quota, len(pool)))
    rng.shuffle(picked)
    return picked


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two label lists of the same, non-zero length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def kappa_interval(a: Sequence[str], b: Sequence[str], n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Kappa, a 95% bootstrap interval over cases (resamples with one label only are skipped), and n."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        ra, rb = [a[i] for i in idx], [b[i] for i in idx]
        if len(set(ra) | set(rb)) > 1:
            values.append(cohen_kappa(ra, rb))
    return cohen_kappa(a, b), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), len(a)


def confusion(truth: Sequence[str], critic: Sequence[str]) -> dict[str, int]:
    """Counts keyed "truth->critic"; `ok->not_ok` is the Critic being too strict, `not_ok->ok` too lenient."""
    return dict(Counter(f"{t}->{c}" for t, c in zip(truth, critic, strict=True)))


QUOTAS = {"revise/tool_fail": 10, "revise/tools_pass": 13, "accept/tool_fail": 15, "accept/tools_pass": 12}


def write_sample(grid_dir: Path, out_dir: Path) -> None:
    cases = load_cases(grid_dir)
    sample = stratified_sample(cases, QUOTAS)
    out_dir.mkdir(parents=True, exist_ok=True)
    population = Counter(c.stratum for c in cases)
    (out_dir / "sample.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in sample))
    # Blind copy: what a labeller sees, without the decision, confidence, issues or run outcome.
    blind = [
        {"case_id": c.case_id, "task": c.task, "step_id": c.step_id, "evidence": c.evidence} for c in sample
    ]
    (out_dir / "blind.json").write_text(json.dumps(blind, indent=1))
    (out_dir / "population.json").write_text(
        json.dumps(
            {"cases": len(cases), "strata": population, "sample": Counter(c.stratum for c in sample)},
            indent=1,
        )
    )
    print(f"{len(cases)} critic verdicts; sampled {len(sample)}: {dict(Counter(c.stratum for c in sample))}")


def load_labels(path: Path) -> dict[str, str]:
    """case_id -> label from a jsonl of {"case_id", "label", ...}; "unsure" labels are left out."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["case_id"]: r["label"] for r in rows if r["label"] in LABELS}


def score(sample: Sequence[Case], population: dict[str, int], labels: dict[str, str]) -> dict:
    """Agreement of the Critic with one labeller, on the labelled cases of a stratified sample.

    Besides kappa on the sample, each stratum's too-strict / too-lenient rate is scaled to that stratum's
    size in the population, so the estimate is about all Critic calls, not about the sample's mix.
    """
    cases = [c for c in sample if c.case_id in labels]
    truth, critic = [labels[c.case_id] for c in cases], [c.critic_label for c in cases]
    kappa, lo, hi, n = kappa_interval(truth, critic)
    strata = {}
    for stratum in sorted({c.stratum for c in cases}):
        rows = [(labels[c.case_id], c.critic_label) for c in cases if c.stratum == stratum]
        strict = sum(t == "ok" and k == "not_ok" for t, k in rows)
        lenient = sum(t == "not_ok" and k == "ok" for t, k in rows)
        size = population.get(stratum, 0)
        strata[stratum] = {
            "labelled": len(rows),
            "population": size,
            "agree": len(rows) - strict - lenient,
            "too_strict": strict,
            "too_lenient": lenient,
            "est_too_strict": size * strict / len(rows),
            "est_too_lenient": size * lenient / len(rows),
        }
    return {
        "labelled": n,
        "kappa": kappa,
        "kappa_ci": [lo, hi],
        "agreement": sum(t == k for t, k in zip(truth, critic, strict=True)) / n,
        "confusion": confusion(truth, critic),
        "strata": strata,
        "est_too_strict": sum(s["est_too_strict"] for s in strata.values()),
        "est_too_lenient": sum(s["est_too_lenient"] for s in strata.values()),
        "population": sum(population.values()),
    }


def labeller_agreement(a: dict[str, str], b: dict[str, str]) -> dict:
    shared = sorted(set(a) & set(b))
    kappa, lo, hi, n = kappa_interval([a[k] for k in shared], [b[k] for k in shared])
    return {"shared": n, "kappa": kappa, "kappa_ci": [lo, hi]}


def main() -> None:
    command, *args = sys.argv[1:]
    if command == "sample":
        write_sample(Path(args[0]), Path(args[1]))
    elif command == "score":
        out = Path(args[0])
        sample = [Case.model_validate_json(line) for line in (out / "sample.jsonl").read_text().splitlines()]
        population = json.loads((out / "population.json").read_text())["strata"]
        result = {}
        for path in sorted(out.glob("labels_*.jsonl")):
            result[path.stem.removeprefix("labels_")] = score(sample, population, load_labels(path))
        names = sorted(result)
        result["between_labellers"] = {
            f"{a}~{b}": labeller_agreement(
                load_labels(out / f"labels_{a}.jsonl"), load_labels(out / f"labels_{b}.jsonl")
            )
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        }
        (out / "scores.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
