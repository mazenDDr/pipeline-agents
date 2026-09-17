"""Run a grid of (task, system, config, seed) cells, resumably (T9).

A grid file (configs/grids/*.yaml) names the tasks, the arms and the seeds:

    name: t9-dev
    tasks: dev                    # dev | test | a list of task ids
    seeds: [1, 2, 3]
    arms:
      - {label: multi, system: multi, config: configs/run/default.yaml}
      - {label: baseline, system: baseline, config: configs/run/default.yaml}
    memory: none                  # none | sequence

Cells run seed by seed, arm by arm, and within a dataset family the warm-up before the follow-up. With
`memory: sequence`, each (arm, seed) keeps its own memory directory, starting empty, and records every run
into it, so a follow-up can learn from its warm-up and nothing crosses between arms or seeds.

Every cell writes outputs/runs/<grid>/<label>_<task>_s<seed>/result.json. A cell with a result is skipped
on re-run (unless it errored and --retry-errors is given); a crashing cell records an error result and the
grid goes on. Before each cell the model servers must answer, or the grid waits.
"""

import json
import shutil
import statistics
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from pipeline_agents.bench.families import FAMILIES, TASKS
from pipeline_agents.bench.run import error_result, run_one
from pipeline_agents.config import load_models, load_run_config


class Arm(BaseModel):
    label: str
    system: Literal["multi", "baseline"]
    config: str


class Grid(BaseModel):
    name: str
    tasks: Literal["dev", "test"] | list[str]
    seeds: list[int]
    arms: list[Arm]
    memory: Literal["none", "sequence"] = "none"

    def task_ids(self) -> list[str]:
        wanted = [t.id for t in TASKS if t.split == self.tasks] if isinstance(self.tasks, str) else self.tasks
        by_family = {f: [t.id for t in m.TASKS] for f, m in FAMILIES.items()}
        # Warm-up before follow-up in each family.
        ordered = [tid for tasks in by_family.values() for tid in tasks]
        return [tid for tid in ordered if tid in wanted]


@dataclass(frozen=True)
class Cell:
    arm: Arm
    task_id: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.arm.label}_{self.task_id}_s{self.seed}"


def cells(grid: Grid) -> list[Cell]:
    return [
        Cell(arm, task_id, seed) for seed in grid.seeds for arm in grid.arms for task_id in grid.task_ids()
    ]


def load_grid(path: Path | str) -> Grid:
    return Grid.model_validate(yaml.safe_load(Path(path).read_text()))


def estimate_seconds(todo: list[Cell], history: list[dict]) -> float:
    """Median seconds of earlier runs with the same system and task kind; 600 s when there are none."""
    kinds = {t.id: t.kind for t in TASKS}
    medians: dict[tuple[str, str], float] = {}
    for key in {(c.arm.system, kinds[c.task_id]) for c in todo}:
        seen = [
            r["seconds"]
            for r in history
            if (r.get("system", "multi"), r.get("kind")) == key and r.get("seconds")
        ]
        medians[key] = statistics.median(seen) if seen else 600.0
    return sum(medians[(c.arm.system, kinds[c.task_id])] for c in todo)


def servers_ready(timeout_s: float = 5.0) -> bool:
    layout = load_models()["layouts"]["tiered"]
    try:
        for tier in ("strong", "cheap"):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{layout[tier]['port']}/health", timeout=timeout_s
            ) as r:
                if r.status != 200:
                    return False
    except OSError:
        return False
    return True


def run_grid(
    grid: Grid,
    root: Path = Path("outputs/runs"),
    retry_errors: bool = False,
    runner: Callable[..., dict] = run_one,
    wait_for_servers: Callable[[], bool] = servers_ready,
    max_wait_s: float = 1800,
    log: Callable[[str], None] = print,
) -> list[dict]:
    grid_dir = root / grid.name
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "grid.json").write_text(grid.model_dump_json(indent=2))
    all_cells = cells(grid)

    def done(cell: Cell) -> dict | None:
        path = grid_dir / cell.run_id / "result.json"
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        return None if (retry_errors and result.get("status") == "error") else result

    todo = [c for c in all_cells if done(c) is None]
    history = [json.loads(p.read_text()) for p in root.glob("**/result.json")]
    log(
        f"{grid.name}: {len(all_cells)} cells, {len(all_cells) - len(todo)} done, {len(todo)} to run, "
        f"estimated {estimate_seconds(todo, history) / 3600:.1f} h"
    )

    results = []
    for n, cell in enumerate(all_cells, 1):
        if (finished := done(cell)) is not None:
            results.append(finished)
            continue
        waited = 0.0
        while not wait_for_servers():
            if waited >= max_wait_s:
                raise RuntimeError(
                    f"model servers not answering after {max_wait_s:.0f} s; grid stopped before {cell.run_id}"
                )
            time.sleep(30)
            waited += 30
        cfg = load_run_config(cell.arm.config).model_copy(update={"seed": cell.seed})
        run_dir = grid_dir / cell.run_id
        memory_dir = (
            grid_dir / "memory" / f"{cell.arm.label}_s{cell.seed}" if grid.memory == "sequence" else None
        )
        start = time.perf_counter()
        try:
            if run_dir.exists():  # an interrupted cell (no result): start it again from scratch
                shutil.rmtree(run_dir)
            result = runner(
                cell.task_id, cell.arm.system, cfg, run_dir, memory_dir, grid.memory == "sequence"
            )
        except Exception as e:  # noqa: BLE001 - recorded, not raised: one broken cell must not stop the grid
            result = error_result(cell.task_id, cell.arm.system, cfg, run_dir, e)
        results.append(result)
        log(
            f"[{n}/{len(all_cells)}] {cell.run_id}: {result['status']} success={result['success']} "
            f"({time.perf_counter() - start:.0f} s, {result.get('model_calls', '?')} calls)"
        )
    return results
