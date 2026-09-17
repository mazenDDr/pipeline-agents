"""Critic audit: loading verdicts from call logs, sampling, kappa and population-weighted estimates."""

import json
from pathlib import Path

import pytest

from pipeline_agents.eval.critic_audit import (
    Case,
    cohen_kappa,
    kappa_interval,
    labeller_agreement,
    load_cases,
    score,
    stratified_sample,
)

EVIDENCE = "# Task\n\n# The step\n\n- id: s1 [load]\n\n# Validation tool findings\n\n{findings}"


def _case(n: int, decision: str, fails: list[str]) -> Case:
    return Case(
        case_id=f"r#{n}",
        run_id="r",
        task="t",
        step_id="s1",
        iteration=1,
        decision=decision,
        confidence=1.0,
        issues=[],
        tool_failures=fails,
        run_success=False,
        evidence="",
    )


def test_kappa_known_values() -> None:
    assert cohen_kappa(["ok", "not_ok"], ["ok", "not_ok"]) == 1.0
    assert cohen_kappa(["ok", "ok", "not_ok", "not_ok"], ["ok", "not_ok", "ok", "not_ok"]) == 0.0
    # observed 0.8, expected 0.6*0.6 + 0.4*0.4 = 0.52 -> (0.8 - 0.52) / 0.48
    a = ["ok"] * 6 + ["not_ok"] * 4
    b = ["ok"] * 5 + ["not_ok"] + ["ok"] + ["not_ok"] * 3
    assert cohen_kappa(a, b) == pytest.approx(0.28 / 0.48)
    kappa, lo, hi, n = kappa_interval(a, b, n_boot=500)
    assert n == 10 and lo <= kappa <= hi


def test_load_cases_reads_critic_calls_and_tool_failures(tmp_path: Path) -> None:
    run = tmp_path / "multi_t_s1"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"run_id": "multi_t_s1", "task": "t", "success": True}))
    findings = "- [pass] row_accounting: fine\n- [FAIL] missing_and_sentinels: 3 missing"
    calls = [
        {"role": "executor", "content": "x", "messages": [], "step_id": "s1", "iteration": 1},
        {
            "role": "critic",
            "content": '```json\n{"decision": "revise", "issues": ["a"], "confidence": 0.9}\n```',
            "messages": [{"role": "user", "content": EVIDENCE.format(findings=findings)}],
            "step_id": "s1",
            "iteration": 1,
        },
        {
            "role": "critic",
            "content": "",
            "error": "timeout",
            "messages": [],
            "step_id": "s1",
            "iteration": 2,
        },
    ]
    (run / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    [case] = load_cases(tmp_path)
    assert (case.case_id, case.decision, case.confidence) == ("multi_t_s1#1", "revise", 0.9)
    assert case.tool_failures == ["missing_and_sentinels"] and case.stratum == "revise/tool_fail"
    assert case.critic_label == "not_ok"


def test_stratified_sample_respects_quotas() -> None:
    cases = [_case(i, "accept", []) for i in range(20)] + [_case(100 + i, "revise", ["x"]) for i in range(3)]
    picked = stratified_sample(cases, {"accept/tools_pass": 5, "revise/tool_fail": 10}, seed=3)
    strata = [c.stratum for c in picked]
    assert strata.count("accept/tools_pass") == 5 and strata.count("revise/tool_fail") == 3
    assert picked == stratified_sample(cases, {"accept/tools_pass": 5, "revise/tool_fail": 10}, seed=3)


def test_score_scales_errors_to_the_population() -> None:
    sample = [_case(i, "accept", []) for i in range(4)] + [_case(10 + i, "revise", ["x"]) for i in range(2)]
    labels = {"r#0": "not_ok", "r#1": "ok", "r#2": "ok", "r#3": "ok", "r#10": "ok", "r#11": "not_ok"}
    result = score(sample, {"accept/tools_pass": 100, "revise/tool_fail": 2}, labels)
    assert result["strata"]["accept/tools_pass"]["est_too_lenient"] == 25.0
    assert result["strata"]["revise/tool_fail"]["est_too_strict"] == 1.0
    assert result["confusion"] == {"not_ok->ok": 1, "ok->ok": 3, "ok->not_ok": 1, "not_ok->not_ok": 1}


def test_unlabelled_and_unsure_cases_are_skipped(tmp_path: Path) -> None:
    from pipeline_agents.eval.critic_audit import load_labels

    path = tmp_path / "labels_x.jsonl"
    path.write_text('{"case_id": "a", "label": "ok"}\n{"case_id": "b", "label": "unsure"}\n')
    assert load_labels(path) == {"a": "ok"}
    both = labeller_agreement({"a": "ok", "b": "not_ok", "c": "ok"}, {"a": "ok", "b": "not_ok"})
    assert both["shared"] == 2 and both["kappa"] == 1.0
