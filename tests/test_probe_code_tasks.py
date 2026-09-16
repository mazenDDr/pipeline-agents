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
