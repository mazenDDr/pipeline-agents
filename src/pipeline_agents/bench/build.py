"""Shared helpers for building task directories (T2).

Layout per task, under data/benchmark/<task_id>/ on the GPU machine (never committed):
    workspace/task.md, workspace/data/..., workspace/output/   what the agent sees
    hidden/holdout_features.csv + holdout_labels.csv (predictive) or hidden/answer.json (analytical)
    spec.json                                                  the TaskSpec, for the checker
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from pipeline_agents.bench.spec import TaskSpec

CONTRACT = """## Deliverables

Write `output/pipeline.py`. It must run from this directory with `python output/pipeline.py`, read its
inputs only from `data/`, and write everything it produces into `output/`. It will be re-run from a clean
copy of `data/` and your `.py` files, so do not rely on files created by hand.

{deliverable}
"""

PREDICTIVE_CONTRACT = """For this predictive task the pipeline must also:
- write `output/metrics.json` containing your own validation estimate, `{{"validation_{metric}": <number>}}`,
  measured on data the model was not trained on, in a way that reflects how the model will be used;
- leave a `output/predict.py` that runs as `python output/predict.py <features.csv> <predictions.csv>` and
  writes a CSV with a header and columns `{id_column},prediction` for every row of the features file. The
  features file has the same format and columns as the training data, minus the target and anything that
  would not be known at prediction time.
"""


def task_dir(root: Path, task: TaskSpec, readme: str) -> tuple[Path, Path]:
    base = root / task.id
    if base.exists():
        shutil.rmtree(base)
    workspace, hidden = base / "workspace", base / "hidden"
    (workspace / "data").mkdir(parents=True)
    (workspace / "output").mkdir()
    hidden.mkdir()
    deliverable = task.deliverable
    if task.kind == "predictive":
        deliverable += "\n\n" + PREDICTIVE_CONTRACT.format(
            metric=task.predictive.metric, id_column=task.predictive.id_column
        )
    (workspace / "task.md").write_text(
        f"# Task\n\n{task.goal}\n\n" + CONTRACT.format(deliverable=deliverable)
    )
    (workspace / "data" / "README.md").write_text(readme)
    (base / "spec.json").write_text(task.model_dump_json(indent=2))
    return workspace, hidden


def write_holdout(
    hidden: Path, features: pd.DataFrame, id_column: str, target: pd.Series, features_text: str | None = None
) -> None:
    """`features_text`, if given, is written verbatim instead of a CSV (raw formats without a header)."""
    if features_text is None:
        features.to_csv(hidden / "holdout_features.csv", index=False)
    else:
        (hidden / "holdout_features.csv").write_text(features_text)
    pd.DataFrame({id_column: features[id_column].to_numpy(), "target": target.to_numpy()}).to_csv(
        hidden / "holdout_labels.csv", index=False
    )


def write_answer(hidden: Path, answer: dict) -> None:
    (hidden / "answer.json").write_text(json.dumps(answer, indent=2))
