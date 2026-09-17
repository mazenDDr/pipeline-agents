"""Delivery checks: what a failed script said reaches the Reviser, and the smoke features drop the target."""

from pathlib import Path

from pipeline_agents.graph.checks import assemble_pipeline, deliver_checks
from pipeline_agents.sandbox.runner import Limits, UnsandboxedRunner
from pipeline_agents.schemas import Ledger, Plan, RunState, TaskContext

STEP = """
import json, pandas as pd
df = pd.read_csv("data/sales.csv")
json.dump({"mean": float(df["amount"].mean())}, open("output/model.json", "w"))
json.dump({"validation_mae": 1.0}, open("output/metrics.json", "w"))
"""
PREDICT_OK = """
import json, sys, pandas as pd
X = pd.read_csv(sys.argv[1])
if "amount" in X.columns:
    print("refusing: the features contain the target column amount")
    sys.exit(1)
mean = json.load(open("output/model.json"))["mean"]
prediction = mean + X["order_id"] * 0.01
pd.DataFrame({"order_id": X["order_id"], "prediction": prediction}).to_csv(sys.argv[2], index=False)
"""
PREDICT_SILENT_FAIL = "import sys\nprint('model file has the wrong version, cannot predict')\nsys.exit(1)\n"


def _state(tmp_path: Path, predict: str) -> RunState:
    ws = tmp_path / "workspace"
    (ws / "data").mkdir(parents=True)
    (ws / "output").mkdir()
    (ws / "data" / "sales.csv").write_text(
        "order_id,region,amount\n" + "".join(f"{i},r{i % 3},{i}\n" for i in range(300))
    )
    (ws / "output" / "step_01_train.py").write_text(STEP)
    (ws / "output" / "predict.py").write_text(predict)
    plan = Plan(
        goal_type="predictive",
        target_column="amount",
        steps=[{"id": "s1", "kind": "train", "intent": "fit the mean", "acceptance_checks": ["metrics"]}],
    )
    task = TaskContext(
        task_id="toy", task_md="predict amount", kind="predictive", id_column="order_id", metric="mae"
    )
    state = RunState(
        run_id="r", task_id="toy", goal="g", task=task, workspace=str(ws), plan=plan, ledger=Ledger(cap_usd=1)
    )
    assemble_pipeline(state, ["step_01_train.py"])
    return state


def test_smoke_features_have_no_target_column(tmp_path: Path) -> None:
    findings = deliver_checks(_state(tmp_path, PREDICT_OK), UnsandboxedRunner(Limits(timeout_s=60)))
    assert all(f.passed for f in findings), [f.detail for f in findings]
    assert [f.tool for f in findings] == ["clean_rerun", "deliverables", "predict_smoke"]


def test_a_failure_printed_to_stdout_reaches_the_finding(tmp_path: Path) -> None:
    findings = deliver_checks(_state(tmp_path, PREDICT_SILENT_FAIL), UnsandboxedRunner(Limits(timeout_s=60)))
    smoke = findings[-1]
    assert smoke.tool == "predict_smoke" and not smoke.passed
    assert "model file has the wrong version" in smoke.detail
    assert "raw data format: one header row, separator ','" in smoke.detail


PREDICT_ROW_NUMBERS = PREDICT_OK.replace('"order_id": X["order_id"]', '"order_id": range(len(X))')
PREDICT_CONSTANT = PREDICT_OK.replace('mean + X["order_id"] * 0.01', "mean")


def test_row_numbers_instead_of_ids_fail_the_smoke_test(tmp_path: Path) -> None:
    state = _state(tmp_path, PREDICT_ROW_NUMBERS)
    sales = Path(state.workspace) / "data" / "sales.csv"
    sales.write_text("order_id,region,amount\n" + "".join(f"{1000 + i},r{i % 3},{i}\n" for i in range(300)))
    smoke = deliver_checks(state, UnsandboxedRunner(Limits(timeout_s=60)))[-1]
    assert not smoke.passed and "not the ids in the input" in smoke.detail


def test_identical_predictions_fail_the_smoke_test(tmp_path: Path) -> None:
    smoke = deliver_checks(_state(tmp_path, PREDICT_CONSTANT), UnsandboxedRunner(Limits(timeout_s=60)))[-1]
    assert not smoke.passed and "all 200 predictions are identical" in smoke.detail


def _two_steps(tmp_path: Path, s2_rows: int) -> tuple[RunState, Plan]:
    """s1 filters 300 raw rows to 150 on purpose and was accepted; s2 then writes `s2_rows` rows."""
    from pipeline_agents.graph.checks import step_checks
    from pipeline_agents.schemas import StepAttempt, StepRecord

    ws = tmp_path / "workspace"
    (ws / "output").mkdir(parents=True)
    table = "order_id,amount\n" + "".join(f"{i},{i}\n" for i in range(300))
    (ws / "output" / "filtered.csv").write_text("\n".join(table.splitlines()[:151]) + "\n")
    (ws / "output" / "features.csv").write_text("\n".join(table.splitlines()[: s2_rows + 1]) + "\n")
    plan = Plan(
        goal_type="predictive",
        target_column="amount",
        steps=[
            {"id": "s1", "kind": "clean", "intent": "keep one region", "acceptance_checks": ["x"]},
            {"id": "s2", "kind": "feature", "intent": "add features", "acceptance_checks": ["x"]},
        ],
    )
    task = TaskContext(task_id="toy", task_md="t", kind="predictive", id_column="order_id", metric="mae")
    state = RunState(
        run_id="r",
        task_id="toy",
        goal="g",
        task=task,
        workspace=str(ws),
        plan=plan,
        ledger=Ledger(cap_usd=1),
        raw_rows={"sales.csv": 300},
        steps={
            "s1": StepRecord(
                step_id="s1",
                status="accepted",
                attempts=[StepAttempt(attempt=1, code="", exit_code=0, artifacts=["filtered.csv"])],
            )
        },
    )
    attempt = StepAttempt(attempt=1, code="", exit_code=0, artifacts=["features.csv"])
    return state, step_checks(state, plan.steps[1], attempt)


def test_row_accounting_compares_with_the_previous_accepted_step(tmp_path: Path) -> None:
    _, findings = _two_steps(tmp_path / "a", s2_rows=150)
    [rows] = [f for f in findings if f.tool == "row_accounting"]
    assert rows.passed and "filtered.csv from s1" in rows.detail  # s1's accepted drop is not blamed on s2

    _, findings = _two_steps(tmp_path / "b", s2_rows=100)
    [rows] = [f for f in findings if f.tool == "row_accounting"]
    assert not rows.passed and "150 rows in, 100 out" in rows.detail
