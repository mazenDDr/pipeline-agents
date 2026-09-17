"""Mine procedural skills from finished dev-split runs whose hidden check passed (T8).

    python scripts/mine_skills.py outputs/runs/<run_id> [...] [--memory-dir outputs/memory/default]

Reads each run's result.json (task, split via the benchmark spec, checker outcome) and its last checkpoint.
Test-split runs are refused by MemorySystem.mine_skills. Needs the models serving.
"""

import argparse
import json
from pathlib import Path

from pipeline_agents.agents.common import Deps, ParseError
from pipeline_agents.bench.spec import TaskSpec
from pipeline_agents.config import MemoryConfig, build_observer, load_run_config
from pipeline_agents.graph.build import load_state
from pipeline_agents.harness.registry import PromptRegistry
from pipeline_agents.memory.store import SentenceEmbedder
from pipeline_agents.memory.system import MemorySystem, Outcome
from pipeline_agents.sandbox.runner import default_runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--memory-dir", default="outputs/memory/default")
    parser.add_argument("--config", default="configs/run/default.yaml")
    args = parser.parse_args()
    cfg = load_run_config(args.config)
    embedder = SentenceEmbedder()
    for run in map(Path, args.runs):
        result = json.loads((run / "result.json").read_text())
        spec = TaskSpec.model_validate_json(
            (Path("data/benchmark") / result["task"] / "spec.json").read_text()
        )
        if spec.split != "dev":
            print(f"skip {run.name}: {spec.split} split")
            continue
        state = load_state(run / "state.sqlite", result["run_id"])
        memory = MemorySystem(Path(args.memory_dir), embedder, MemoryConfig(), "dev")
        observer = build_observer(cfg, f"mine-{result['run_id']}", run / "mining")
        deps = Deps(
            observer=observer, registry=PromptRegistry("prompts"), config=cfg, runner=default_runner()
        )
        checker = result["checker"]
        outcome = Outcome(
            status=result["status"], checker_passed=checker["passed"], first_failure=checker["first_failure"]
        )
        try:
            print(f"{run.name}: {memory.mine_skills(deps, state, outcome)} new skills")
        except ParseError as e:
            print(f"{run.name}: FAILED, the miner's reply was unusable: {e}")


if __name__ == "__main__":
    main()
