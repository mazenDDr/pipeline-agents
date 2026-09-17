"""Benchmark task contract (T2). Frozen before any agent run; changing it invalidates earlier runs.

A task is a raw dataset plus a goal. The agent sees a workspace:

    task.md          the goal and the deliverables
    data/            input files and README.md (a data dictionary)
    output/          everything the agent delivers

and must deliver `output/pipeline.py`, which runs from the workspace root with
`python output/pipeline.py`, reads only `data/`, and writes into `output/`:

- predictive tasks: `output/metrics.json` with the agent's own validation estimate
  (`{"validation_<metric>": <number>}`), and `output/predict.py`, run as
  `python output/predict.py <features.csv> <predictions.csv>`, writing `<id>,prediction`;
- analytical tasks: `output/answer.json`, in the shape task.md describes.

The checker copies `data/` and the agent's `.py` files to a fresh directory, re-runs the pipeline,
and scores it against answers the agent never sees. Every trap must be discoverable from the data or
its dictionary: the benchmark measures judgement, not guessing.
"""

from typing import Literal

from pydantic import BaseModel, Field

Metric = Literal["roc_auc", "mae", "rmse"]


class Trap(BaseModel):
    id: str  # e.g. "leak_casual_registered"
    kind: Literal[
        "target_leak",
        "sentinel",
        "placeholder",
        "duplicates",
        "int_coded",
        "group_leak",
        "temporal",
        "format",
        "semantics",
    ]
    description: str  # what a careful analyst would notice, and where it is documented
    # Which checker stage catches it, or "score only" when falling for it merely lowers the metric.
    # Recorded so the failure analysis never claims a trap was tested when it was not.
    caught_by: Literal["predict", "score", "honest_estimate", "answer", "score only"]


class PredictiveCheck(BaseModel):
    id_column: str
    metric: Metric
    threshold: float  # holdout metric must be >= (roc_auc) or <= (mae, rmse) this
    # |validation estimate - holdout metric| must be <= this: absolute for roc_auc, relative to the holdout
    # value for error metrics (mae, rmse), whose scale depends on the task.
    honesty_tolerance: float


class AnalyticalCheck(BaseModel):
    numeric_tolerance: float = 1e-3  # relative tolerance for numbers in answer.json


class TaskSpec(BaseModel):
    id: str  # "{family}-{n}-{slug}"
    family: str  # one dataset; the dev/test split is by family
    split: Literal["dev", "test"]
    role: Literal["warmup", "followup"]  # the follow-up is a similar task on the same data (episodic memory)
    kind: Literal["predictive", "analytical"]
    difficulty: Literal["easy", "medium", "hard"]
    goal: str  # shown to the agent in task.md
    deliverable: str  # the exact output contract, shown in task.md
    traps: list[Trap] = Field(min_length=1)
    predictive: PredictiveCheck | None = None
    analytical: AnalyticalCheck | None = None
    timeout_s: int = 900  # clean re-run of pipeline.py
