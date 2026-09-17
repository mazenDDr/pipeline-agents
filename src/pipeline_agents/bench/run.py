"""Run one (task, system, config, seed) cell and score it with the hidden checker (T9).

The grid runner and scripts/run_task.py both call `run_one`. A cell writes outputs/runs/<run_id>/ with the
agents' workspace, calls.jsonl, cache/, state.sqlite (multi) or baseline.json (baseline), and result.json.
"""

import json
import shutil
import time
import traceback
from pathlib import Path

from pipeline_agents.agents.baseline import run_baseline
from pipeline_agents.agents.common import Deps, ParseError
from pipeline_agents.bench.checker import check
from pipeline_agents.bench.families import TASKS
from pipeline_agents.bench.spec import TaskSpec
from pipeline_agents.config import RunConfig, build_observer
from pipeline_agents.graph.build import run
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.memory.store import SentenceEmbedder
from pipeline_agents.memory.system import MemorySystem, Outcome
from pipeline_agents.sandbox.runner import default_runner
from pipeline_agents.schemas import Ledger, RunState, TaskContext

BENCH = Path("data/benchmark")


def task_context(spec: TaskSpec, workspace: Path) -> TaskContext:
    return TaskContext(
        task_id=spec.id,
        task_md=(workspace / "task.md").read_text(),
        data_readme=(workspace / "data" / "README.md").read_text(),
        kind=spec.kind,
        id_column=spec.predictive.id_column if spec.predictive else None,
        metric=spec.predictive.metric if spec.predictive else None,
    )


def revisions(final, system: str) -> int:
    """Rework a run needed: revisions of every step (archived plans included) plus re-plans; for the baseline,
    attempts after the first."""
    if system == "baseline":
        return max(len(final.attempts) - 1, 0)
    records = [*final.steps.values(), *(r for plan in final.archived_steps for r in plan.values())]
    return sum(r.revision_count for r in records) + final.replans


def run_one(
    task_id: str,
    system: str,
    cfg: RunConfig,
    run_dir: Path,
    memory_dir: Path | None = None,
    record_memory: bool = False,
    resume: bool = False,
) -> dict:
    spec = TaskSpec.model_validate_json((BENCH / task_id / "spec.json").read_text())
    workspace = run_dir / "workspace"
    run_id = run_dir.name
    state = None
    if not resume:
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} exists; resume it or choose another run id")
        shutil.copytree(BENCH / task_id / "workspace", workspace)
        (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
        state = RunState(
            run_id=run_id,
            task_id=spec.id,
            goal=spec.goal,
            task=task_context(spec, workspace),
            workspace=str(workspace.resolve()),
            ledger=Ledger(cap_usd=cfg.budget_usd, degrade_at=cfg.degrade_at),
        )

    memory = None
    memory_on = any((cfg.memory.semantic, cfg.memory.procedural, cfg.memory.episodic))
    if memory_dir is not None and (memory_on or record_memory):
        memory = MemorySystem(memory_dir, SentenceEmbedder(), cfg.memory, spec.split)
    observer = build_observer(cfg, run_id, run_dir, cache_dir=run_dir / "cache")
    deps = Deps(
        observer=observer,
        registry=PromptRegistry("prompts"),
        config=cfg,
        runner=default_runner(),
        memory=memory,
    )
    store_before = memory.digest() if memory else None

    start = time.perf_counter()
    if system == "baseline":
        final = run_baseline(deps, task_context(spec, workspace), workspace.resolve(), run_id)
        (run_dir / "baseline.json").write_text(final.model_dump_json(indent=2))
        steps, replans, hits = {"attempts": len(final.attempts)}, 0, []
    else:
        final = run(deps, state, run_dir / "state.sqlite")
        steps = {
            sid: {"attempts": len(r.attempts), "revisions": r.revision_count, "status": r.status}
            for sid, r in final.steps.items()
        }
        replans, hits = final.replans, [h.model_dump() for h in final.memory_hits]
    seconds = time.perf_counter() - start

    report = check(spec, workspace, BENCH / task_id / "hidden", run_dir / "check_scratch")
    shutil.rmtree(run_dir / "check_scratch", ignore_errors=True)
    result = {
        "run_id": run_id,
        "task": spec.id,
        "family": spec.family,
        "split": spec.split,
        "kind": spec.kind,
        "system": system,
        "config": cfg.name,
        "seed": cfg.seed,
        "status": final.status,
        "stop_reason": final.stop_reason,
        "success": final.status == "succeeded" and report.passed,
        "checker": report.to_dict(),
        "revisions": revisions(final, system),
        "replans": replans,
        "seconds": round(seconds, 1),
        "model_calls": final.model_calls,
        "shadow_usd": final.ledger.total.shadow_usd,
        "by_role": {k: v.model_dump() for k, v in final.ledger.by_role.items()},
        "steps": steps,
        "sandboxed": deps.runner.sandboxed,
        "memory": {"config": cfg.memory.model_dump(), "store_before": store_before, "hits": hits},
    }
    if record_memory and memory is not None and system == "multi":
        outcome = Outcome(
            status=final.status, checker_passed=report.passed, first_failure=report.first_failure
        )
        recorded: dict = {"episodes": memory.record_episode(final, outcome)}
        try:
            recorded["facts"] = memory.extract_facts(deps, final, outcome)
        except ParseError as e:
            recorded["facts_error"] = str(e)
        result["memory"]["recorded"] = recorded
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def error_result(task_id: str, system: str, cfg: RunConfig, run_dir: Path, error: BaseException) -> dict:
    """A cell that crashed still gets a result, so the grid goes on and the crash is counted, not hidden."""
    # From the task definitions in code, not data/benchmark: recording a crash must not depend on the data.
    spec = next((t for t in TASKS if t.id == task_id), None)
    result = {
        "run_id": run_dir.name,
        "task": task_id,
        "family": spec.family if spec else None,
        "split": spec.split if spec else None,
        "kind": spec.kind if spec else None,
        "system": system,
        "config": cfg.name,
        "seed": cfg.seed,
        "status": "error",
        "success": False,
        "stop_reason": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc()[-4000:],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
