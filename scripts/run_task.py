"""Run the agents (or the single-agent baseline) on one benchmark task, then score what they delivered with
the hidden checker.

    python scripts/run_task.py --task bike-2-weather [--system multi|baseline] [--config FILE] [--run-id ID]
                               [--resume]

On the GPU machine, with the models serving (bash scripts/serve_models.sh tiered). Writes to
outputs/runs/<run_id>/: workspace/ (the agents' copy), calls.jsonl, cache/, result.json, and state.sqlite
(multi) or baseline.json (baseline).
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from pipeline_agents.agents.baseline import run_baseline
from pipeline_agents.agents.common import Deps
from pipeline_agents.bench.checker import check
from pipeline_agents.bench.spec import TaskSpec
from pipeline_agents.config import build_observer, load_run_config
from pipeline_agents.graph.build import run
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.sandbox.runner import default_runner
from pipeline_agents.schemas import Ledger, RunState, TaskContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--system", choices=["multi", "baseline"], default="multi")
    parser.add_argument("--config", default="configs/run/default.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.system == "baseline":
        raise SystemExit("the baseline has no checkpoints to resume")

    cfg = load_run_config(args.config)
    bench = Path("data/benchmark") / args.task
    spec = TaskSpec.model_validate_json((bench / "spec.json").read_text())
    label = cfg.name if args.system == "multi" else f"baseline-{cfg.name}"
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M}_{label}_{args.task}"
    run_dir = Path("outputs/runs") / run_id
    workspace = run_dir / "workspace"

    state = None
    if not args.resume:
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists; pass --resume or another --run-id")
        shutil.copytree(bench / "workspace", workspace)
        (run_dir / "config.yaml").write_text(Path(args.config).read_text())
        task = TaskContext(
            task_id=spec.id,
            task_md=(workspace / "task.md").read_text(),
            data_readme=(workspace / "data" / "README.md").read_text(),
            kind=spec.kind,
            id_column=spec.predictive.id_column if spec.predictive else None,
            metric=spec.predictive.metric if spec.predictive else None,
        )
        state = RunState(
            run_id=run_id,
            task_id=spec.id,
            goal=spec.goal,
            task=task,
            workspace=str(workspace.resolve()),
            ledger=Ledger(cap_usd=cfg.budget_usd, degrade_at=cfg.degrade_at),
        )

    observer = build_observer(cfg, run_id, run_dir, cache_dir=run_dir / "cache")
    deps = Deps(observer=observer, registry=PromptRegistry("prompts"), config=cfg, runner=default_runner())
    start = time.perf_counter()
    if args.system == "baseline":
        final = run_baseline(deps, state.task, workspace.resolve(), run_id)
        (run_dir / "baseline.json").write_text(final.model_dump_json(indent=2))
        replans, steps = 0, {"attempts": len(final.attempts)}
    else:
        final = run(deps, state, run_dir / "state.sqlite")
        replans = final.replans
        steps = {
            sid: {"attempts": len(r.attempts), "revisions": r.revision_count, "status": r.status}
            for sid, r in final.steps.items()
        }
    seconds = time.perf_counter() - start

    report = check(spec, workspace, bench / "hidden", run_dir / "check_scratch")
    shutil.rmtree(run_dir / "check_scratch", ignore_errors=True)
    result = {
        "run_id": run_id,
        "task": spec.id,
        "system": args.system,
        "status": final.status,
        "stop_reason": final.stop_reason,
        "checker": report.to_dict(),
        "success": final.status == "succeeded" and report.passed,
        "seconds": round(seconds, 1),
        "model_calls": final.model_calls,
        "shadow_usd": final.ledger.total.shadow_usd,
        "by_role": {k: v.model_dump() for k, v in final.ledger.by_role.items()},
        "replans": replans,
        "steps": steps,
        "sandboxed": deps.runner.sandboxed,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "by_role"}, indent=2))


if __name__ == "__main__":
    main()
