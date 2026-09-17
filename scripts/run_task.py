"""Run the agents (or the single-agent baseline) on one benchmark task and score it with the hidden checker.

    python scripts/run_task.py --task bike-2-weather [--system multi|baseline] [--config FILE] [--seed N]
                               [--run-id ID] [--resume] [--memory-dir DIR] [--record-memory]

On the GPU machine, with the models serving (bash scripts/serve_models.sh tiered). For many cells, use
scripts/run_grid.py. Memory: the config's switches decide what is retrieved from --memory-dir; --record-memory
writes the run's episode and data facts there afterwards.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from pipeline_agents.bench.run import run_one
from pipeline_agents.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--system", choices=["multi", "baseline"], default="multi")
    parser.add_argument("--config", default="configs/run/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--memory-dir", default="outputs/memory/default")
    parser.add_argument("--record-memory", action="store_true")
    args = parser.parse_args()
    if args.resume and args.system == "baseline":
        raise SystemExit("the baseline has no checkpoints to resume")
    cfg = load_run_config(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    label = cfg.name if args.system == "multi" else f"baseline-{cfg.name}"
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M}_{label}_{args.task}"
    result = run_one(
        args.task,
        args.system,
        cfg,
        Path("outputs/runs") / run_id,
        Path(args.memory_dir),
        args.record_memory,
        args.resume,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("by_role", "memory")}, indent=2))


if __name__ == "__main__":
    main()
