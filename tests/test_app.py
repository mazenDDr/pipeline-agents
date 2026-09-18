"""The app's library part: building a workspace from an upload, and reading a run back as rows."""

import json
from pathlib import Path

import pytest

from pipeline_agents.app.user_run import UserTask, prepare, read_calls, spend, trace
from pipeline_agents.schemas import (
    Ledger,
    Plan,
    Revision,
    RunState,
    Spend,
    StepAttempt,
    StepRecord,
    TaskContext,
    Verdict,
)

CSV = b"id,amount\n1,10\n2,20\n"


def _task(**kw) -> UserTask:
    return UserTask(
        **{"goal": " Count the orders. ", "kind": "analytical", "files": {"sales.csv": CSV}, **kw}
    )


def test_an_upload_becomes_a_workspace(tmp_path: Path) -> None:
    state = prepare(_task(), tmp_path / "run")
    workspace = tmp_path / "run" / "workspace"
    assert (workspace / "data" / "sales.csv").read_bytes() == CSV
    assert (workspace / "output").is_dir()
    assert "Count the orders." in (workspace / "task.md").read_text()
    assert "answer.json" in (workspace / "task.md").read_text()
    assert "No data dictionary" in (workspace / "data" / "README.md").read_text()
    assert state.goal == "Count the orders." and state.task.kind == "analytical"


def test_a_predictive_task_names_its_columns_and_metric(tmp_path: Path) -> None:
    task = _task(
        kind="predictive", target="amount", id_column="id", metric="roc_auc", data_readme="id: the order"
    )
    state = prepare(task, tmp_path / "run")
    text = (tmp_path / "run" / "workspace" / "task.md").read_text()
    assert "`id`" in text and "predicted `amount`" in text and '{"validation_roc_auc": <number>}' in text
    assert state.task.metric == "roc_auc"
    assert (tmp_path / "run" / "workspace" / "data" / "README.md").read_text() == "id: the order"


def test_a_run_needs_files_and_a_free_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one data file"):
        prepare(_task(files={}), tmp_path / "run")
    prepare(_task(), tmp_path / "run")
    with pytest.raises(FileExistsError):
        prepare(_task(), tmp_path / "run")


def _state_with_history() -> RunState:
    plan = Plan(
        goal_type="analytical",
        target_column=None,
        steps=[{"id": "s1", "kind": "clean", "intent": "clean it", "acceptance_checks": ["no nulls"]}],
    )
    record = StepRecord(
        step_id="s1",
        status="accepted",
        attempts=[
            StepAttempt(attempt=1, code="a", exit_code=1, failure="logic", artifacts=[]),
            StepAttempt(attempt=2, code="b", exit_code=0, artifacts=["clean.csv"]),
        ],
        verdicts=[
            Verdict(decision="revise", confidence=1.0, issues=["it crashed"], findings=[]),
            Verdict(decision="accept", confidence=0.9),
        ],
        revisions=[Revision(action="fix", instructions="read the file with the right separator")],
    )
    ledger = Ledger(cap_usd=0.05, by_role={"planner": Spend(calls=1, input_tokens=10, shadow_usd=0.001)})
    return RunState(
        run_id="r",
        task_id="r",
        goal="g",
        task=TaskContext(task_id="r", task_md="t", kind="analytical"),
        workspace="/tmp/ws",
        plan=plan,
        steps={"s1": record},
        ledger=ledger,
    )


def test_the_trace_is_one_row_per_attempt_with_its_verdict() -> None:
    rows = trace(_state_with_history())
    assert [r["attempt"] for r in rows] == [1, 2]
    assert rows[0]["decision"] == "revise" and rows[0]["failure"] == "logic"
    assert rows[0]["revision"].startswith("read the file")
    assert rows[1]["decision"] == "accept" and rows[1]["artifacts"] == "clean.csv"


def test_spend_lists_each_role() -> None:
    assert spend(_state_with_history()) == [
        {
            "role": "planner",
            "calls": 1,
            "input tokens": 10,
            "output tokens": 0,
            "shadow $": 0.001,
            "seconds": 0.0,
        }
    ]


def test_calls_are_read_up_to_the_last_whole_line(tmp_path: Path) -> None:
    (tmp_path / "calls.jsonl").write_text(json.dumps({"role": "planner"}) + "\n" + '{"role": "exec')
    assert read_calls(tmp_path) == [{"role": "planner"}]
    assert read_calls(tmp_path / "missing") == []
