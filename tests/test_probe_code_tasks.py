"""The probe's checkers must accept a correct script and reject a plausible wrong one.

Otherwise a model's pass rate measures the checker, not the model.
"""

from pathlib import Path

import pytest

from pipeline_agents.probe.code_tasks import TASKS
from pipeline_agents.probe.sandbox import run_script


def _run(task, code: str, tmp_path: Path, seed: int = 0):
    work, hidden = tmp_path / "work", tmp_path / "hidden"
    work.mkdir()
    hidden.mkdir()
    task.build(work, hidden, seed)
    result = run_script(code, work, timeout_s=60)
    assert result.exit_code is not None, "script timed out"
    return result, task.check(work, hidden)


def test_task_ids_unique() -> None:
    assert len({t.id for t in TASKS}) == len(TASKS)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
@pytest.mark.parametrize("seed", [0, 1])
def test_reference_passes(task, seed, tmp_path: Path) -> None:
    result, check = _run(task, task.reference, tmp_path, seed)
    assert result.exit_code == 0, result.stderr
    assert check.passed, check.detail


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
@pytest.mark.parametrize("seed", [0, 1])
def test_broken_fails(task, seed, tmp_path: Path) -> None:
    _, check = _run(task, task.broken, tmp_path, seed)
    assert not check.passed, f"checker accepted the broken script: {check.detail}"


def test_no_output_fails(tmp_path: Path) -> None:
    _, check = _run(TASKS[0], "pass\n", tmp_path)
    assert not check.passed


HONEST_FOREST = {
    "leaky_column": (
        "import pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\n"
        "X = ['tenure_months', 'monthly_fee', 'support_tickets']\n"
        "m = RandomForestClassifier(random_state=0).fit(tr[X], tr['churned'])\n"
        "pd.DataFrame({'customer_id': te['customer_id'], 'churn_probability': m.predict_proba(te[X])[:, 1]})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
    "unseen_category": (
        "import pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\n"
        "Xtr = pd.get_dummies(tr[['plan_type', 'weekly_hours']])\n"
        "Xte = pd.get_dummies(te[['plan_type', 'weekly_hours']]).reindex(columns=Xtr.columns, fill_value=0)\n"
        "m = RandomForestClassifier(random_state=0).fit(Xtr, tr['upgraded'])\n"
        "pd.DataFrame({'account_id': te['account_id'], 'upgrade_probability': m.predict_proba(Xte)[:, 1]})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
}


@pytest.mark.parametrize("task_id", sorted(HONEST_FOREST))
@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_honest_random_forest_passes(task_id, seed, tmp_path: Path) -> None:
    """An AUC threshold must not depend on the model family: a leak-free random forest passes."""
    task = next(t for t in TASKS if t.id == task_id)
    result, check = _run(task, HONEST_FOREST[task_id], tmp_path, seed)
    assert result.exit_code == 0, result.stderr
    assert check.passed, check.detail


@pytest.mark.parametrize("script", sorted((Path(__file__).parent / "fixtures" / "probe").glob("leak_*.py")))
@pytest.mark.parametrize("seed", [0, 7])
def test_real_leaky_scripts_fail(script: Path, seed, tmp_path: Path) -> None:
    """Scripts from the probe that used the leaking column must fail, whatever their model family."""
    task = next(t for t in TASKS if t.id == "leaky_column")
    _, check = _run(task, script.read_text(), tmp_path, seed)
    assert not check.passed, check.detail
