"""Run a grid of benchmark cells, then summarise it (T9).

    python scripts/run_grid.py configs/grids/t9-dev.yaml [--retry-errors] [--dry-run]

On the GPU machine with the models serving. Resumable: finished cells are skipped. Writes
outputs/runs/<grid>/<cell>/result.json per cell and outputs/runs/<grid>/{summary.json, report.md}.
"""

import argparse
import json
from pathlib import Path

from pipeline_agents.bench.grid import cells, estimate_seconds, load_grid, run_grid
from pipeline_agents.eval.report import load_results, render, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the cells and the time estimate only")
    args = parser.parse_args()
    grid = load_grid(args.grid)
    if args.dry_run:
        history = [json.loads(p.read_text()) for p in Path("outputs/runs").glob("**/result.json")]
        todo = cells(grid)
        for cell in todo:
            print(cell.run_id)
        hours = estimate_seconds(todo, history) / 3600
        print(f"{len(todo)} cells, estimated {hours:.1f} h from {len(history)} past runs")
        return
    run_grid(grid, retry_errors=args.retry_errors)
    grid_dir = Path("outputs/runs") / grid.name
    summary = summarize(load_results(grid_dir))
    (grid_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (grid_dir / "report.md").write_text(render(summary, grid.name))
    print(render(summary, grid.name))


if __name__ == "__main__":
    main()
