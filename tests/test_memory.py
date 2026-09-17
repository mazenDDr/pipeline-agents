"""Memory: each store's retrieval rule, the leakage rules, and hits reaching the right prompts."""

import json
from pathlib import Path

import pytest

from pipeline_agents.agents.common import Deps
from pipeline_agents.config import LoopConfig, MemoryConfig, RoleConfig, RunConfig
from pipeline_agents.graph.build import run
from pipeline_agents.harness.budget import Price
from pipeline_agents.harness.client import FakeModelClient
from pipeline_agents.harness.observer import Observer
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.memory.store import HashEmbedder, MemoryItem, item_id
from pipeline_agents.memory.system import MemorySystem, Outcome, fingerprint
from pipeline_agents.sandbox.runner import Limits, UnsandboxedRunner
from pipeline_agents.schemas import Ledger, PlanStep, RunState, TaskContext
from tests.test_graph import ACCEPT, CLEAN, PLAN, REPORT, ScriptedModel, _workspace

PRICE = Price(input_per_mtok=0.1, output_per_mtok=0.3, source="test", retrieved="2026-09-17")
ALL_ON = MemoryConfig(procedural=True, semantic=True, episodic=True, top_k=2)


def _memory(tmp_path: Path, split: str = "dev", config: MemoryConfig = ALL_ON) -> MemorySystem:
    return MemorySystem(tmp_path / "memory", HashEmbedder(), config, split)


def _item(store: str, key: str, text: str, task: str = "other-task", split: str = "dev") -> MemoryItem:
    return MemoryItem(
        id=item_id(store, key, text),
        store=store,
        key=key,
        text=text,
        source_run="r0",
        source_task=task,
        source_split=split,
    )


def test_semantic_facts_follow_the_file_not_the_task(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    memory = _memory(tmp_path)
    key = fingerprint(ws / "data" / "sales.csv")
    memory.semantic.add(
        [
            _item("semantic", key, "sales.csv: amount uses -999 for missing"),
            _item("semantic", "someotherfile", "orders.csv: dates are month first"),
        ],
        HashEmbedder(),
    )
    hits = memory.run_hits("toy", ws, "any goal", "profile")
    assert [h.text for h in hits if h.store == "semantic"] == ["sales.csv: amount uses -999 for missing"]


def test_fingerprint_ignores_rows_but_not_header(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    (a / "x.csv").write_text("id,v\n1,2\n")
    (b / "x.csv").write_text("id,v\n9,9\n8,8\n")
    (c / "x.csv").write_text("id,w\n1,2\n")
    assert fingerprint(a / "x.csv") == fingerprint(b / "x.csv") != fingerprint(c / "x.csv")


def test_episodic_never_returns_the_same_task(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.episodic.add(
        [
            _item("episodic", "toy", "Task toy: mean amount with -999 sentinels", task="toy"),
            _item("episodic", "air-1", "Task air-1: daily means with -200 sentinels", task="air-1"),
        ],
        HashEmbedder(),
    )
    hits = memory.run_hits("toy", _workspace(tmp_path), "mean amount with sentinels", "")
    assert [h.key for h in hits if h.store == "episodic"] == ["air-1"]


def test_procedural_retrieval_is_ranked_and_dev_only(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.procedural.add(
        [
            _item(
                "procedural",
                "clean",
                "When: a numeric column uses a sentinel for missing. Do: replace with NaN",
            ),
            _item("procedural", "split", "When: forecasting later periods. Do: chronological split"),
            _item(
                "procedural",
                "clean",
                "When: sentinel values for missing numbers. Do: NaN (test run)",
                split="test",
            ),
        ],
        HashEmbedder(),
    )
    step = PlanStep(
        id="s1", kind="clean", intent="replace the sentinel -999 with NaN", acceptance_checks=["no -999"]
    )
    hits = memory.step_hits(step)
    assert hits[0].text.startswith("When: a numeric column uses a sentinel")
    assert all("test run" not in h.text for h in hits) and all(h.step_id == "s1" for h in hits)


def test_switches_turn_each_store_off(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    memory = _memory(tmp_path, config=MemoryConfig())
    memory.semantic.add([_item("semantic", fingerprint(ws / "data" / "sales.csv"), "a fact")], HashEmbedder())
    memory.episodic.add([_item("episodic", "air-1", "an episode", task="air-1")], HashEmbedder())
    step = PlanStep(id="s1", kind="clean", intent="clean it", acceptance_checks=["x"])
    assert memory.run_hits("toy", ws, "goal", "") == [] and memory.step_hits(step) == []


def test_skills_are_mined_from_dev_runs_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dev runs only"):
        _memory(tmp_path, split="test").mine_skills(
            None, None, Outcome(status="succeeded", checker_passed=True)
        )


def test_outcome_carries_no_checker_details() -> None:
    text = Outcome(status="succeeded", checker_passed=False, first_failure="answer").describe()
    assert text == "The run ended with status succeeded. The hidden check failed at: answer."


# --- integration with the graph --------------------------------------------------------------


def _deps(tmp_path: Path, model: ScriptedModel, memory: MemorySystem) -> tuple[Deps, RunState, Path]:
    ws = _workspace(tmp_path)
    cfg = RunConfig(
        name="t",
        budget_usd=10,
        loop=LoopConfig(),
        memory=memory.config,
        roles={r: RoleConfig(max_tokens=512) for r in ("planner", "executor", "critic", "reviser")},
    )
    clients = {"strong": model.client, "cheap": model.client}
    observer = Observer("r1", tmp_path / "run", Ledger(cap_usd=10), clients, {"m": PRICE})
    deps = Deps(
        observer=observer,
        registry=PromptRegistry("prompts"),
        config=cfg,
        runner=UnsandboxedRunner(Limits(timeout_s=60)),
        memory=memory,
    )
    task = TaskContext(
        task_id="toy",
        task_md="Report the mean known amount in output/answer.json.",
        data_readme="amount: -999 means not recorded",
        kind="analytical",
    )
    state = RunState(
        run_id="r1", task_id="toy", goal=task.task_md, task=task, workspace=str(ws), ledger=Ledger(cap_usd=10)
    )
    return deps, state, tmp_path / "run" / "state.sqlite"


def test_hits_reach_the_planner_and_executor_prompts(tmp_path: Path) -> None:
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, ACCEPT])
    memory = _memory(tmp_path)
    deps, state, db = _deps(tmp_path, model, memory)
    ws = Path(state.workspace)
    embedder = HashEmbedder()
    memory.semantic.add(
        [_item("semantic", fingerprint(ws / "data" / "sales.csv"), "FACT: amount uses -999")], embedder
    )
    memory.episodic.add(
        [_item("episodic", "air-1", "EPISODE: sentinel -200 in air quality", task="air-1")], embedder
    )
    memory.procedural.add(
        [_item("procedural", "clean", "SKILL: replace a missing-value sentinel with NaN")], embedder
    )
    final = run(deps, state, db)
    assert final.status == "succeeded"
    calls = [json.loads(line) for line in (tmp_path / "run" / "calls.jsonl").read_text().splitlines()]
    planner = calls[0]["messages"][1]["content"]
    executor_s1 = next(c for c in calls if c["role"] == "executor" and c["step_id"] == "s1")
    executor_text = executor_s1["messages"][1]["content"]
    assert "FACT" in planner and "EPISODE" in planner and "SKILL" not in planner
    assert "FACT" in executor_text and "SKILL" in executor_text and "EPISODE" not in executor_text
    assert {h.store for h in final.memory_hits} == {"semantic", "episodic", "procedural"}


def test_recording_writes_an_episode_and_facts_without_checker_details(tmp_path: Path) -> None:
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, ACCEPT])
    memory = _memory(tmp_path)
    deps, state, db = _deps(tmp_path, model, memory)
    final = run(deps, state, db)
    writer = FakeModelClient([json.dumps({"facts": ["sales.csv: amount uses -999 for missing values"]})], "m")
    deps.observer.clients = {"strong": writer, "cheap": writer}
    outcome = Outcome(status=final.status, checker_passed=True)
    assert memory.record_episode(final, outcome) == 1
    assert memory.extract_facts(deps, final, outcome) == 1
    episode = memory.episodic.items()[0].text
    assert "Task toy" in episode and "hidden check passed" in episode
    prompt = writer.calls[0][0][1]["content"]
    assert "hidden check passed" in prompt and "want" not in prompt
    assert memory.semantic.items()[0].key == fingerprint(Path(final.workspace) / "data" / "sales.csv")


def test_memory_writers_ask_for_schema_enforced_json(tmp_path: Path) -> None:
    """A free-form miner reply broke its JSON twice in the first real run; writers now constrain it."""
    model = ScriptedModel(planner=[json.dumps(PLAN)], executor=[CLEAN, REPORT], critic=[ACCEPT, ACCEPT])
    memory = _memory(tmp_path)
    deps, state, db = _deps(tmp_path, model, memory)
    final = run(deps, state, db)
    writer = FakeModelClient([json.dumps({"facts": []}), json.dumps({"skills": []})], "m")
    deps.observer.clients = {"strong": writer, "cheap": writer}
    outcome = Outcome(status="succeeded", checker_passed=True)
    assert memory.extract_facts(deps, final, outcome) == 0 and memory.mine_skills(deps, final, outcome) == 0
    formats = [params.get("response_format", {}) for _, params in writer.calls]
    assert [f["json_schema"]["name"] for f in formats] == ["memory_writer", "skill_miner"]
    assert "skills" in formats[1]["json_schema"]["schema"]["properties"]
