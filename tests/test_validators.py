"""Critic tools on planted problems: each must catch its problem and stay quiet on clean data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_agents.sandbox.runner import Limits, UnsandboxedRunner
from pipeline_agents.tools.validators import (
    clean_rerun,
    metric_sanity,
    missing_and_sentinels,
    numbers_stored_as_text,
    predictions_frame,
    required_columns,
    row_accounting,
    split_overlap,
    target_leakage,
    temporal_order,
)

RNG = np.random.default_rng(0)


def _churn(n: int = 2000) -> pd.DataFrame:
    tenure = RNG.integers(1, 72, n)
    tickets = RNG.poisson(1.5, n)
    churned = (RNG.random(n) < 1 / (1 + np.exp(-(0.5 - 0.08 * tenure + 0.8 * tickets)))).astype(int)
    return pd.DataFrame({"customer_id": range(n), "tenure": tenure, "tickets": tickets, "churned": churned})


def _hourly(n: int = 3000) -> pd.DataFrame:
    hr = RNG.integers(0, 24, n)
    casual = RNG.poisson(5 + 10 * (hr > 10), n)
    registered = RNG.poisson(20 + 40 * (hr > 7), n)
    return pd.DataFrame(
        {
            "hr": hr,
            "temp": RNG.random(n),
            "casual": casual,
            "registered": registered,
            "cnt": casual + registered,
        }
    )


# --- rows and columns ------------------------------------------------------------------------


def test_row_accounting() -> None:
    assert row_accounting(1000, 995).passed
    silent_drop = row_accounting(250, 84)
    assert not silent_drop.passed and "66.4% dropped" in silent_drop.detail
    assert not row_accounting(1000, 1400).passed  # a join that duplicated rows
    assert row_accounting(1000, 1400, allow_added=True).passed
    kept_blank_lines = row_accounting(7854, 7968, blank_rows=114)
    assert kept_blank_lines.passed and "empty lines" in kept_blank_lines.detail
    assert not row_accounting(7854, 7969, blank_rows=114).passed


def test_required_columns() -> None:
    df = pd.DataFrame({"hr": [1], "cnt": [2]})
    assert not required_columns(df, ["instant", "cnt"]).passed
    assert required_columns(df, ["cnt"]).passed


def test_missing_placeholders_and_sentinels() -> None:
    clean = pd.DataFrame(
        {
            "age": RNG.integers(18, 90, 500),
            "job": RNG.choice(["a", "b"], 500),
            "flag": RNG.integers(0, 2, 500),
        }
    )
    assert missing_and_sentinels(clean).passed
    dirty = clean.assign(
        age=clean["age"].where(RNG.random(500) > 0.1, -999),
        job=clean["job"].where(RNG.random(500) > 0.05, "unknown"),
    )
    finding = missing_and_sentinels(dirty)
    assert not finding.passed and "-999" in finding.detail and "unknown" in finding.detail
    assert not missing_and_sentinels(
        clean.assign(age=clean["age"].astype(float).where(RNG.random(500) > 0.02))
    ).passed


def test_numbers_stored_as_text() -> None:
    assert (
        numbers_stored_as_text(
            pd.DataFrame({"co": ["2,6", "1,3", "0,9"] * 20, "name": ["x", "y", "z"] * 20})
        ).passed
        is False
    )
    assert numbers_stored_as_text(pd.DataFrame({"co": [2.6, 1.3], "name": ["x", "y"]})).passed


# --- leakage ---------------------------------------------------------------------------------


def test_clean_features_are_not_leaks() -> None:
    assert target_leakage(_churn(), "churned", exclude=["customer_id"]).passed
    assert target_leakage(_hourly().drop(columns=["casual", "registered"]), "cnt").passed


def test_sum_of_components_is_a_leak() -> None:
    finding = target_leakage(_hourly(), "cnt")
    assert not finding.passed and "linear combination" in finding.detail
    assert "casual" in finding.detail and "registered" in finding.detail


def test_field_filled_only_after_the_outcome_is_a_leak() -> None:
    df = _churn()
    df["cancel_reason"] = np.where(df["churned"] == 1, RNG.choice(["price", "moved"], len(df)), None)
    finding = target_leakage(df, "churned", exclude=["customer_id"])
    assert not finding.passed and "cancel_reason" in finding.detail and "empty" in finding.detail


def test_near_copy_of_the_target_is_a_leak() -> None:
    df = _churn()
    df["days_until_cancel"] = np.where(df["churned"] == 1, RNG.integers(1, 30, len(df)), 999)
    finding = target_leakage(df, "churned", exclude=["customer_id"])
    assert not finding.passed and "days_until_cancel" in finding.detail


def test_pure_category_is_a_leak() -> None:
    df = _churn()
    df["status"] = np.where(df["churned"] == 1, "closed", RNG.choice(["active", "paused"], len(df)))
    finding = target_leakage(df, "churned", exclude=["customer_id"])
    assert not finding.passed and "status" in finding.detail


# --- splits, metrics, predictions --------------------------------------------------------------


def test_split_overlap() -> None:
    df = _churn()
    train, valid = df.iloc[:1500], df.iloc[1500:]
    assert split_overlap(train, valid, id_column="customer_id").passed
    leaky = pd.concat([valid, train.iloc[:10]])
    finding = split_overlap(train, leaky, id_column="customer_id")
    assert not finding.passed and "10 ids" in finding.detail
    # A couple of records that happen to match is not leakage; the same rows in both halves is.
    rows = pd.DataFrame({"x": range(2000)})
    assert split_overlap(rows.iloc[:1000], rows.iloc[998:]).passed  # 2 rows in both: coincidence
    assert not split_overlap(rows.iloc[:1000], rows.iloc[900:]).passed  # 100 rows in both: the same records
    patients = pd.DataFrame({"patient": [1, 1, 2, 3], "x": [1, 2, 3, 4]})
    assert not split_overlap(patients.iloc[:1], patients.iloc[1:], group_column="patient").passed


def test_temporal_order() -> None:
    days = pd.Series(pd.date_range("2012-01-01", periods=100))
    assert temporal_order(days[:80], days[80:]).passed
    assert not temporal_order(
        days.sample(80, random_state=0), days.drop(days.sample(80, random_state=0).index)
    ).passed


@pytest.mark.parametrize(
    ("metric", "value", "dummy", "passed"),
    [
        ("roc_auc", 0.81, 0.5, True),
        ("roc_auc", 0.999, 0.5, False),
        ("roc_auc", 0.51, 0.5, False),
        ("mae", 45.0, 140.0, True),
        ("mae", 139.0, 140.0, False),
        ("mae", 0.0, 140.0, False),
        ("mae", float("nan"), 140.0, False),
    ],
)
def test_metric_sanity(metric, value, dummy, passed) -> None:
    assert metric_sanity(metric, value, dummy).passed is passed


def test_predictions_frame() -> None:
    ids = pd.Series(range(100))
    good = pd.DataFrame({"id": ids, "prediction": RNG.random(100)})
    assert predictions_frame(good, "id", ids, probability=True).passed
    bad = pd.concat([good.iloc[:90], good.iloc[:2]]).assign(prediction=lambda d: d["prediction"] * 2)
    detail = predictions_frame(bad, "id", ids, probability=True).detail
    assert "2 duplicated ids" in detail and "10 of 100 ids" in detail and "outside [0, 1]" in detail


# --- reproducibility -------------------------------------------------------------------------


def _step(tmp_path: Path, code: str) -> Path:
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    (work / "output").mkdir()
    (work / "data" / "in.csv").write_text("a\n3\n1\n2\n")
    (work / "output" / "step.py").write_text(code)
    return work


def test_clean_rerun_reproduces_a_deterministic_step(tmp_path: Path) -> None:
    code = (
        "import pandas as pd\n"
        "pd.read_csv('data/in.csv').sort_values('a').to_csv('output/out.csv', index=False)\n"
    )
    work = _step(tmp_path, code)
    runner = UnsandboxedRunner(Limits(timeout_s=60))
    assert runner.run(work, ["output/step.py"]).ok
    assert clean_rerun(runner, work, "output/step.py", ["output/out.csv"]).passed


def test_clean_rerun_catches_hand_made_outputs(tmp_path: Path) -> None:
    """The Executor claims out.csv came from the step, but the step writes something else."""
    work = _step(tmp_path, "open('output/other.csv', 'w').write('x')\n")
    (work / "output" / "out.csv").write_text("a\n1\n2\n3\n")
    finding = clean_rerun(UnsandboxedRunner(Limits(timeout_s=60)), work, "output/step.py", ["output/out.csv"])
    assert not finding.passed and "out.csv not produced" in finding.detail


def test_a_known_code_inside_the_normal_range_is_data() -> None:
    """Regression from the first real run: 99 rentals in an hour was flagged as a missing-value code, and the
    Critic had valid rows deleted."""
    counts = pd.DataFrame({"cnt": RNG.poisson(120, 5000), "casual": RNG.integers(0, 367, 5000)})
    assert (counts == 99).any().all()
    assert missing_and_sentinels(counts).passed


def test_leakage_check_survives_categories_whose_targets_are_all_missing() -> None:
    """Regression from the dev grid: a category with only missing targets crashed the purity computation."""
    df = pd.DataFrame(
        {
            "site": ["a"] * 50 + ["b"] * 50 + ["c"] * 20,
            "value": RNG.normal(size=120),
            "target": list(RNG.normal(size=100)) + [np.nan] * 20,
        }
    )
    finding = target_leakage(df, "target")
    assert finding.passed, finding.detail
