"""Failure classes: each one is read from what a run recorded, with the answer cases mapped by hand."""

from pipeline_agents.analysis.failures import classify

CASES = {"bike-2-weather/mean_temp_celsius": {"category": "unit_ignored", "diagnosis": "normalized temp"}}


def _result(**kw) -> dict:
    base = {
        "run_id": "multi_bike-1-forecast_s1",
        "task": "bike-1-forecast",
        "status": "succeeded",
        "success": False,
        "stop_reason": "delivered",
        "checker": {"stages": [], "holdout_metric": None, "validation_metric": None},
    }
    return {**base, **kw}


def _stages(*names_passed) -> dict:
    return [{"name": n, "passed": p, "detail": f"{n} detail"} for n, p in names_passed]


def test_stops_are_read_from_the_stop_reason() -> None:
    infra = classify(_result(status="failed", stop_reason="infra: s3 failed 4 times (unshare: ...)"))
    assert infra.category == "memory_limit" and infra.arm == "multi"
    parse = classify(_result(status="failed", stop_reason="reviser output unusable after a re-ask on s7: x"))
    assert parse.category == "parse_failure"
    baseline = classify(
        _result(run_id="baseline_bike-1-forecast_s1", status="failed", stop_reason="no passing attempt in 6")
    )
    assert baseline.category == "repair_exhausted" and baseline.arm == "baseline"


def test_a_stop_for_a_human_is_split_by_what_it_kept_arguing_about() -> None:
    def human(tail: str) -> str:
        return classify(_result(status="needs_human", stop_reason=f"re-plans used up (1): {tail}")).category

    assert (
        human("s7 still rejected: ['delivery check predict_smoke failed: ...']") == "loop_exhausted_delivery"
    )
    assert human("s3 still rejected: ['the script crashed (1): Traceback']") == "loop_exhausted_crash"
    assert human("s3 still rejected: ['row_accounting: 21% dropped']") == "loop_exhausted_tool"
    assert human("s3: the plan sequence is flawed, the Planner must reorder") == "loop_exhausted_plan"


def test_score_and_honesty_carry_the_direction_of_the_estimate() -> None:
    mae = classify(
        _result(
            checker={
                "stages": _stages(("score", False), ("honest_estimate", True)),
                "validation_metric": 54.0,
                "holdout_metric": 78.0,
            }
        )
    )
    assert mae.category == "below_threshold" and "(54) was better than the holdout (78)" in mae.detail
    auc = classify(
        _result(
            run_id="multi_adult-1-income_s1",
            task="adult-1-income",
            checker={
                "stages": _stages(("score", True), ("honest_estimate", False)),
                "validation_metric": 0.93,
                "holdout_metric": 0.84,
            },
        )
    )
    assert auc.category == "dishonest_estimate" and "was better than the holdout" in auc.detail


def test_a_wrong_answer_takes_its_class_from_the_hand_read_case() -> None:
    known = classify(
        _result(
            run_id="multi_bike-2-weather_s3",
            task="bike-2-weather",
            checker={"stages": [{"name": "answer", "passed": False, "detail": "mean_temp_celsius: got 0.5"}]},
        ),
        CASES,
    )
    assert known.category == "unit_ignored" and known.diagnosis == "normalized temp"
    unknown = classify(
        _result(
            run_id="multi_bike-2-weather_s3",
            task="bike-2-weather",
            checker={"stages": [{"name": "answer", "passed": False, "detail": "other_key: got 1"}]},
            status="succeeded",
        ),
        CASES,
    )
    assert unknown.category == "wrong_answer" and unknown.diagnosis == ""
