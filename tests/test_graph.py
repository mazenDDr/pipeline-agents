"""The orchestration graph, branch by branch, with a scripted model per role and real step scripts.

The toy task: data/sales.csv, where amount uses -999 for missing, and the deliverable is output/answer.json
  with
the mean known amount. Every test checks where the run ends up and why, not just that it ran.
"""

import json
from pathlib import Path

import pytest

from pipeline_agents.agents.common import Deps
from pipeline_agents.config import LoopConfig, RoleConfig, RunConfig
from pipeline_agents.graph.build import run
from pipeline_agents.harness.budget import Price
from pipeline_agents.harness.client import Completion, FakeModelClient
from pipeline_agents.harness.observer import Observer
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.sandbox.runner import Limits, RunResult, UnsandboxedRunner
from pipeline_agents.schemas import Ledger, RunState, TaskContext

PRICE = Price(input_per_mtok=0.1, output_per_mtok=0.3, source="test", retrieved="2026-09-17")

PLAN = {
    "goal_type": "analytical",
    "target_column": None,
    "rationale": "amount uses -999 for missing.",
    "steps": [
        {
            "id": "s1",
            "kind": "clean",
            "intent": "load sales and mark -999 as missing",
            "acceptance_checks": ["no -999 left"],
        },
        {
            "id": "s2",
            "kind": "report",
            "intent": "write the mean known amount to answer.json",
            "depends_on": ["s1"],
            "acceptance_checks": ["answer.json written"],
        },
    ],
}
CLEAN = """```python
import pandas as pd
df = pd.read_csv("data/sales.csv")
df["amount"] = df["amount"].where(df["amount"] != -999)
print("rows", len(df), "missing", int(df["amount"].isna().sum()))
df.to_csv("output/clean.csv", index=False)
```
CLAIMS: marks -999 as missing and writes output/clean.csv"""
CRASH = "```python\nimport pandas as pd\npd.read_csv('data/sales.csv')['nope']\n```\nCLAIMS: nothing"
REPORT = """```python
import json, pandas as pd
df = pd.read_csv("output/clean.csv")
json.dump({"mean_amount": float(df["amount"].mean())}, open("output/answer.json", "w"))
print("mean", df["amount"].mean())
```
CLAIMS: writes output/answer.json"""
BAD_REPORT = REPORT.replace(
    'json.dump({"mean_amount": float(df["amount"].mean())}, open("output/answer.json", "w"))',
    'open("output/answer.json", "w").write("not json")',
)
ACCEPT = json.dumps({"decision": "accept", "issues": [], "confidence": 0.9})
REVISE = json.dumps({"decision": "revise", "issues": ["amount still contains -999"], "confidence": 0.8})
UNSURE = json.dumps({"decision": "revise", "issues": ["cannot tell"], "confidence": 0.2})
FIX = json.dumps(
    {"action": "fix", "instructions": "replace -999 with missing before saving", "reason": "sentinel"}
)


class ScriptedModel:
    """One queue of replies per role, chosen from the system prompt; records the order of roles called."""

    ROLES = {
        "You are the Planner": "planner",
        "You are the Executor": "executor",
        "You are the Critic": "critic",
        "You are the Reviser": "reviser",
    }

    def __init__(self, **queues: list) -> None:
        self.queues = {role: list(replies) for role, replies in queues.items()}
        self.log: list[str] = []
        self.client = FakeModelClient([self._reply] * 200, model="m")

    def _reply(self, messages, params):
        role = next(r for prefix, r in self.ROLES.items() if messages[0]["content"].startswith(prefix))
        self.log.append(role)
        if not self.queues.get(role):
            raise AssertionError(f"no scripted reply left for {role}; calls so far: {self.log}")
        reply = self.queues[role].pop(0)
        if isinstance(reply, BaseException):  # KeyboardInterrupt simulates the process dying
            raise reply
        return reply


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "run" / "workspace"
    (ws / "data").mkdir(parents=True)
    (ws / "output").mkdir()
    (ws / "data" / "sales.csv").write_text(
        "order_id,amount\n" + "".join(f"{i},{-999 if i % 5 == 0 else 10 + i}\n" for i in range(1, 41))
    )
    (ws / "data" / "README.md").write_text("sales.csv: amount in dollars; -999 means not recorded.")
    return ws


def _setup(
    tmp_path: Path, model: ScriptedModel, loop: LoopConfig | None = None, budget: float = 10.0, runner=None
) -> tuple[Deps, RunState, Path]:
    ws = _workspace(tmp_path)
    cfg = RunConfig(
        name="test",
        budget_usd=budget,
        loop=loop or LoopConfig(),
        roles={r: RoleConfig(max_tokens=512) for r in ("planner", "executor", "critic", "reviser")},
    )
    observer = Observer(
        "r1",
        tmp_path / "run",
        Ledger(cap_usd=budget),
        {"strong": model.client, "cheap": model.client},
        {"m": PRICE},
    )
    deps = Deps(
        observer=observer,
        registry=PromptRegistry("prompts"),
        config=cfg,
        runner=runner or UnsandboxedRunner(Limits(timeout_s=60)),
    )
    task = TaskContext(
        task_id="toy",
        task_md="Report the mean known amount in output/answer.json.",
        data_readme=(ws / "data" / "README.md").read_text(),
        kind="analytical",
    )
    state = RunState(
        run_id="r1",
        task_id="toy",
        goal=task.task_md,
        task=task,
        workspace=str(ws),
        ledger=Ledger(cap_usd=budget),
    )
    return deps, state, tmp_path / "run" / "state.sqlite"


def test_happy_path_delivers(tmp_path: Path) -> None:
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, ACCEPT])
    deps, state, db = _setup(tmp_path, model)
    final = run(deps, state, db)
    assert final.status == "succeeded" and final.stop_reason == "delivered", final.stop_reason
    assert model.log == ["planner", "executor", "critic", "executor", "critic"]
    answer = json.loads((Path(final.workspace) / "output" / "answer.json").read_text())
    assert answer["mean_amount"] == pytest.approx(sum(10 + i for i in range(1, 41) if i % 5) / 32)
    assert all(f.passed for f in final.deliverable_findings)
    assert (Path(final.workspace) / "output" / "pipeline.py").exists() and db.exists()
    assert final.model_calls == 5 and final.ledger.total.calls == 5


def test_crash_skips_the_critic_and_is_fixed(tmp_path: Path) -> None:
    model = ScriptedModel(
        planner=[json.dumps(PLAN)], executor=[CRASH, CLEAN, REPORT], critic=[ACCEPT, ACCEPT], reviser=[FIX]
    )
    deps, state, db = _setup(tmp_path, model)
    final = run(deps, state, db)
    assert final.status == "succeeded"
    assert model.log[:4] == ["planner", "executor", "reviser", "executor"]  # no critic call on the crash
    s1 = final.steps["s1"]
    assert len(s1.attempts) == 2 and s1.attempts[0].failure == "logic" and s1.revision_count == 1
    assert "KeyError" in s1.verdicts[0].issues[0]


def test_revisions_used_up_escalate_to_a_replan_that_succeeds(tmp_path: Path) -> None:
    model = ScriptedModel(
        planner=[json.dumps(PLAN), json.dumps(PLAN)],
        executor=[CLEAN, CLEAN, CLEAN, CLEAN, REPORT],
        critic=[REVISE, REVISE, REVISE, ACCEPT, ACCEPT],
        reviser=[FIX, FIX],
    )
    deps, state, db = _setup(tmp_path, model, LoopConfig(max_revisions_per_step=2, max_replans=1))
    final = run(deps, state, db)
    assert final.status == "succeeded", final.stop_reason
    assert final.replans == 1 and len(final.plan_history) == 1 and final.plan.version == 2
    assert (
        final.archived_steps[0]["s1"].revision_count == 2
        and final.archived_steps[0]["s1"].status == "escalated"
    )
    assert model.log.count("reviser") == 2  # the third rejection escalates without asking the Reviser
    replan_prompt = json.loads((tmp_path / "run" / "calls.jsonl").read_text().splitlines()[-5])
    assert (
        replan_prompt["role"] == "planner"
        and "The previous plan failed" in replan_prompt["messages"][1]["content"]
    )


def test_replans_used_up_stop_for_a_human(tmp_path: Path) -> None:
    escalate = json.dumps(
        {"action": "escalate", "instructions": "a different approach", "reason": "impossible"}
    )
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN], critic=[REVISE], reviser=[escalate])
    deps, state, db = _setup(tmp_path, model, LoopConfig(max_replans=0))
    final = run(deps, state, db)
    assert final.status == "needs_human" and "re-plans used up" in final.stop_reason


def test_low_confidence_stops_for_a_human_instead_of_looping(tmp_path: Path) -> None:
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN], critic=[UNSURE])
    deps, state, db = _setup(tmp_path, model, LoopConfig(human_review_threshold=0.5))
    final = run(deps, state, db)
    assert final.status == "needs_human" and "low critic confidence (0.20)" in final.stop_reason
    assert model.log == ["planner", "executor", "critic"]


class FlakyRunner(UnsandboxedRunner):
    """Times out on its first `failures` runs, then behaves."""

    def __init__(self, failures: int) -> None:
        super().__init__(Limits(timeout_s=60))
        self.failures = failures

    def run(self, workdir, argv):
        if self.failures > 0:
            self.failures -= 1
            return RunResult(None, "", "timeout", 60.0, "timeout", sandboxed=False)
        return super().run(workdir, argv)


def test_infra_failure_retries_without_spending_a_revision(tmp_path: Path) -> None:
    model = ScriptedModel(
        planner=[json.dumps(PLAN)], executor=[CLEAN, CLEAN, REPORT], critic=[ACCEPT, ACCEPT]
    )
    deps, state, db = _setup(tmp_path, model, runner=FlakyRunner(failures=1))
    final = run(deps, state, db)
    assert final.status == "succeeded"
    s1 = final.steps["s1"]
    assert [a.failure for a in s1.attempts] == ["infra", None] and s1.revision_count == 0


def test_infra_failures_used_up_stop_the_run(tmp_path: Path) -> None:
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN] * 3)
    deps, state, db = _setup(tmp_path, model, LoopConfig(infra_retries=2), runner=FlakyRunner(failures=10))
    final = run(deps, state, db)
    assert final.status == "failed" and final.stop_reason.startswith("infra: s1 failed 3 times")


def test_model_endpoint_failure_retries_the_node(tmp_path: Path) -> None:
    down = Completion(model="m", content="", error="APIConnectionError: refused", infra_error=True)
    model = ScriptedModel(planner=[down, json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, ACCEPT])
    deps, state, db = _setup(tmp_path, model)
    final = run(deps, state, db)
    assert final.status == "succeeded" and model.log[:2] == ["planner", "planner"]


def test_unparseable_reply_gets_one_reask(tmp_path: Path) -> None:
    model = ScriptedModel(
        planner=["Here is my plan: step one, load.", json.dumps(PLAN)],
        executor=[CLEAN, REPORT],
        critic=[ACCEPT, ACCEPT],
    )
    deps, state, db = _setup(tmp_path, model)
    final = run(deps, state, db)
    assert final.status == "succeeded"
    reask = json.loads((tmp_path / "run" / "calls.jsonl").read_text().splitlines()[1])
    assert "could not be used" in reask["messages"][-1]["content"]


def test_failed_delivery_check_sends_the_last_step_back(tmp_path: Path) -> None:
    model = ScriptedModel(
        planner=[json.dumps(PLAN)],
        executor=[CLEAN, BAD_REPORT, REPORT],
        critic=[ACCEPT, ACCEPT, ACCEPT],
        reviser=[FIX],
    )
    deps, state, db = _setup(tmp_path, model)
    final = run(deps, state, db)
    assert final.status == "succeeded"
    s2 = final.steps["s2"]
    assert len(s2.attempts) == 2 and any("answer.json" in i for i in s2.verdicts[1].issues)


def test_budget_cap_stops_the_run(tmp_path: Path) -> None:
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, ACCEPT])
    deps, state, db = _setup(tmp_path, model, budget=0.00001)
    final = run(deps, state, db)
    assert final.status == "failed" and final.stop_reason.startswith("budget:")


def test_node_visit_cap_stops_the_run(tmp_path: Path) -> None:
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, ACCEPT])
    deps, state, db = _setup(tmp_path, model, LoopConfig(max_node_visits=3))
    final = run(deps, state, db)
    assert final.status == "failed" and final.stop_reason.startswith("cap: 3 node visits")


def test_resume_from_checkpoint_after_a_crash(tmp_path: Path) -> None:
    """The process dies during s2's critique; a new process resumes from the last checkpoint without
    re-planning or re-running s1."""
    first = ScriptedModel(
        planner=[json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, KeyboardInterrupt()]
    )
    deps, state, db = _setup(tmp_path, first)
    with pytest.raises(KeyboardInterrupt):
        run(deps, state, db)
    second = ScriptedModel(critic=[ACCEPT])
    deps2, _, _ = _setup(tmp_path / "unused", second)
    deps2.observer = Observer(
        "r1",
        tmp_path / "run",
        Ledger(cap_usd=10.0),
        {"strong": second.client, "cheap": second.client},
        {"m": PRICE},
    )
    final = run(deps2, None, db)
    assert final.status == "succeeded" and second.log == ["critic"]
    assert final.steps["s1"].status == "accepted" and len(final.steps["s2"].attempts) == 1
    assert (
        final.ledger.total.calls == 5
    )  # 4 before the crash, restored from the checkpoint, plus the resumed critic
