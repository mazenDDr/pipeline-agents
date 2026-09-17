"""Build the benchmark and prove each checker right before any agent runs (T2).

For every task: build its workspace and hidden answers from data/raw, then run every solution in
benchmark/solutions/<task_id>/ through the checker. The reference and honest variants must pass; each
broken variant must fail the stage named in expected.yaml (first or not). Run on the GPU machine:

    python scripts/validate_benchmark.py [--families bike,...] [--tasks id,...]

Appends to outputs/runs/t2-validate/results.jsonl; scripts/build_benchmark_doc.py summarises it.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from pipeline_agents.bench.checker import check
from pipeline_agents.bench.families import FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    raw, bench = Path("data/raw"), Path("data/benchmark")
    run_dir = Path("outputs/runs/t2-validate")
    run_dir.mkdir(parents=True, exist_ok=True)
    families = args.families.split(",") if args.families else list(FAMILIES)
    wanted = set(args.tasks.split(",")) if args.tasks else None

    rows = []
    for name in families:
        module = FAMILIES[name]
        tasks = [t for t in module.TASKS if not wanted or t.id in wanted]
        if not args.no_build:
            module.build(raw, bench, module.TASKS)
        for task in tasks:
            solutions = Path("benchmark/solutions") / task.id
            expected = yaml.safe_load((solutions / "expected.yaml").read_text())
            for variant, want in expected.items():
                with tempfile.TemporaryDirectory() as tmp:
                    workspace = Path(tmp) / "workspace"
                    shutil.copytree(bench / task.id / "workspace", workspace)
                    for py in (solutions / variant).glob("*.py"):
                        shutil.copy(py, workspace / "output" / py.name)
                    report = check(task, workspace, bench / task.id / "hidden", Path(tmp) / "scratch")
                failed = [stage for stage in report.stages if not stage.passed]
                got = "pass" if report.passed else "+".join(stage.name for stage in failed)
                row = {
                    "task": task.id,
                    "variant": variant,
                    "expected": want,
                    "got": got,
                    # A broken solution proves a stage catches its mistake when that stage fails, first or
                    # not.
                    "ok": report.passed if want == "pass" else want in {stage.name for stage in failed},
                    **report.to_dict(),
                }
                rows.append(row)
                metric = "" if report.holdout_metric is None else f" holdout={report.holdout_metric:.4f}"
                if report.validation_metric is not None:
                    metric += f" validation={report.validation_metric:.4f}"
                detail = f" | {failed[0].detail[-160:]}" if failed else ""
                print(
                    f"{'OK ' if row['ok'] else 'BAD'} {task.id:28s} {variant:24s} want={want:16s} got={got}"
                    f"{metric}{detail}",
                    flush=True,
                )

    with (run_dir / "results.jsonl").open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows) - len(bad)}/{len(rows)} solutions behaved as expected")


if __name__ == "__main__":
    main()
