"""Re-score stored probe code after a checker or task-data fix, without calling the models again.

    python -m pipeline_agents.probe.rescore outputs/runs/<run_id> --tasks leaky_column,unseen_category

Re-runs each stored script for the named tasks against freshly built task data and replaces those
rows in results.jsonl. The first rescore keeps the original file as results.before-rescore.jsonl.
Caveat, recorded in each rescored row: the model saw a preview of the *old* files in its prompt.
Scripts only depend on column names and types, which did not change.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from pipeline_agents.probe import code_tasks
from pipeline_agents.probe.run import extract_code
from pipeline_agents.probe.sandbox import run_script


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    task_ids = set(args.tasks.split(","))
    tasks = {t.id: t for t in code_tasks.TASKS if t.id in task_ids}
    assert set(tasks) == task_ids, f"unknown tasks: {task_ids - set(tasks)}"
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

    backup = run_dir / "results.before-rescore.jsonl"
    if not backup.exists():
        shutil.copy(run_dir / "results.jsonl", backup)

    calls = {}
    for line in (run_dir / "calls.jsonl").read_text().splitlines():
        call = json.loads(line)
        calls[call["key"]] = call  # the last attempt wins, as in results.jsonl

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if row["phase"] != "code" or row["task"] not in tasks:
            continue
        task = tasks[row["task"]]
        with tempfile.TemporaryDirectory() as tmp:
            work, hidden = Path(tmp) / "work", Path(tmp) / "hidden"
            work.mkdir()
            hidden.mkdir()
            task.build(work, hidden, cfg["data_seed"])
            run = run_script(
                extract_code(calls[row["key"]]["content"]), work, timeout_s=cfg["code_timeout_s"]
            )
            check = task.check(work, hidden) if run.exit_code == 0 else code_tasks.CheckResult(False, "")
        if check.passed != row["passed"]:
            changed += 1
            print(f"{row['key']}: {row['passed']} -> {check.passed} ({check.detail or run.stderr[-120:]})")
        row.update(
            passed=check.passed,
            detail=check.detail or run.stderr[-300:],
            exit_code=run.exit_code,
            rescored="task data regenerated after a checker fix; prompt preview showed the old files",
        )

    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"rescored {sum(r.get('rescored') is not None for r in rows)} rows, {changed} changed")


if __name__ == "__main__":
    main()
