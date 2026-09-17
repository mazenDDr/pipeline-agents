"""Checker mechanics on a toy task (no real data): each stage fails for the right reason."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_agents.bench.checker import answers_match, check
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

TRAP = [Trap(id="t", kind="semantics", description="toy", caught_by="score only")]
PREDICTIVE = TaskSpec(
    id="toy-1",
    family="toy",
    split="dev",
    role="warmup",
    kind="predictive",
    difficulty="easy",
    goal="g",
    deliverable="d",
    traps=TRAP,
    timeout_s=60,
    predictive=PredictiveCheck(id_column="id", metric="roc_auc", threshold=0.8, honesty_tolerance=0.05),
)
ANALYTICAL = PREDICTIVE.model_copy(
    update={"kind": "analytical", "predictive": None, "analytical": AnalyticalCheck(numeric_tolerance=1e-3)}
)

PIPELINE = """
import json, pickle, pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("data/train.csv")
m = LogisticRegression().fit(df[["x"]], df["y"])
pickle.dump(m, open("output/model.pkl", "wb"))
json.dump({{"validation_roc_auc": {auc}}}, open("output/metrics.json", "w"))
"""
PREDICT = """
import pickle, sys, pandas as pd
X = pd.read_csv(sys.argv[1]); m = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame({"id": X["id"], "prediction": m.predict_proba(X[["x"]])[:, 1]}).to_csv(sys.argv[2], index=False)
"""


def _task_dirs(tmp: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    y = (x + rng.normal(scale=0.5, size=600) > 0).astype(int)
    workspace, hidden = tmp / "workspace", tmp / "hidden"
    (workspace / "data").mkdir(parents=True)
    (workspace / "output").mkdir()
    hidden.mkdir()
    pd.DataFrame({"id": range(400), "x": x[:400], "y": y[:400]}).to_csv(
        workspace / "data" / "train.csv", index=False
    )
    pd.DataFrame({"id": range(400, 600), "x": x[400:]}).to_csv(hidden / "holdout_features.csv", index=False)
    pd.DataFrame({"id": range(400, 600), "target": y[400:]}).to_csv(
        hidden / "holdout_labels.csv", index=False
    )
    return workspace, hidden


def _deliver(workspace: Path, pipeline: str | None, predict: str | None) -> None:
    if pipeline is not None:
        (workspace / "output" / "pipeline.py").write_text(pipeline)
    if predict is not None:
        (workspace / "output" / "predict.py").write_text(predict)


@pytest.mark.parametrize(
    ("pipeline", "predict", "first_failure"),
    [
        (PIPELINE.format(auc=0.92), PREDICT, None),
        (None, PREDICT, "delivered"),
        ("raise SystemExit(3)", PREDICT, "clean_rerun"),
        (PIPELINE.format(auc=0.92), "raise KeyError('duration')", "predict"),
        (
            PIPELINE.format(auc=0.92),
            PREDICT.replace(".to_csv(sys.argv[2]", ".head(100).to_csv(sys.argv[2]"),
            "coverage",
        ),
        (PIPELINE.format(auc=0.99 + 0.009), PREDICT, "honest_estimate"),
        (PIPELINE.format(auc=0.5), PREDICT.replace("[:, 1]", "[:, 0]"), "score"),
    ],
    ids=[
        "pass",
        "not-delivered",
        "pipeline-crash",
        "predict-crash",
        "missing-rows",
        "overclaimed",
        "bad-model",
    ],
)
def test_predictive_stages(pipeline, predict, first_failure, tmp_path: Path) -> None:
    workspace, hidden = _task_dirs(tmp_path)
    _deliver(workspace, pipeline, predict)
    report = check(PREDICTIVE, workspace, hidden, tmp_path / "scratch")
    assert report.first_failure == first_failure, report.stages
    assert report.passed == (first_failure is None)


def test_rerun_ignores_leftover_artifacts(tmp_path: Path) -> None:
    """A model file left in output/ by hand must not rescue a pipeline that no longer writes it."""
    workspace, hidden = _task_dirs(tmp_path)
    _deliver(workspace, PIPELINE.format(auc=0.92), PREDICT)
    assert check(PREDICTIVE, workspace, hidden, tmp_path / "scratch1").passed
    (workspace / "output" / "model.pkl").write_bytes(
        (tmp_path / "scratch1" / "output" / "model.pkl").read_bytes()
    )
    _deliver(
        workspace,
        "import json; json.dump({'validation_roc_auc': 0.92}, open('output/metrics.json', 'w'))",
        None,
    )
    report = check(PREDICTIVE, workspace, hidden, tmp_path / "scratch2")
    assert report.first_failure == "predict"


def test_analytical_answer(tmp_path: Path) -> None:
    workspace, hidden = _task_dirs(tmp_path)
    (hidden / "answer.json").write_text(json.dumps({"rate": 0.25, "top": ["a", "b"]}))
    _deliver(
        workspace,
        "import json; json.dump({'rate': 0.25001, 'top': ['A', 'b']}, open('output/answer.json', 'w'))",
        None,
    )
    assert check(ANALYTICAL, workspace, hidden, tmp_path / "s1").passed
    _deliver(
        workspace,
        "import json; json.dump({'rate': 0.25, 'top': ['b', 'a']}, open('output/answer.json', 'w'))",
        None,
    )
    assert check(ANALYTICAL, workspace, hidden, tmp_path / "s2").first_failure == "answer"


@pytest.mark.parametrize(
    ("got", "want", "ok"),
    [
        ({"a": 1.0001}, {"a": 1.0}, True),
        ({"a": 1.1}, {"a": 1.0}, False),
        ({"a": "x"}, {"a": 1.0}, False),
        ({}, {"a": 1}, False),
        ({"a": [1, 2]}, {"a": [1, 2, 3]}, False),
        ({"a": " Mist "}, {"a": "mist"}, True),
        ({"a": 0.0}, {"a": 0.0}, True),
    ],
)
def test_answers_match(got, want, ok) -> None:
    assert (answers_match(got, want, 1e-3) is None) == ok


def test_perfect_error_metric_does_not_crash_honesty(tmp_path: Path) -> None:
    """MAE 0 on the holdout (e.g. predictions that equal the labels) must be scored, not divide by zero."""
    workspace, hidden = _task_dirs(tmp_path)
    labels = pd.read_csv(hidden / "holdout_labels.csv")
    task = PREDICTIVE.model_copy(
        update={
            "predictive": PredictiveCheck(id_column="id", metric="mae", threshold=0.5, honesty_tolerance=0.2)
        }
    )
    (workspace / "data" / "answers.csv").write_text(labels.to_csv(index=False))
    _deliver(
        workspace,
        "import json; json.dump({'validation_mae': 0.0}, open('output/metrics.json', 'w'))",
        "import sys, pandas as pd\n"
        "pd.read_csv('data/answers.csv')"
        ".rename(columns={'target': 'prediction'}).to_csv(sys.argv[2], index=False)\n",
    )
    report = check(task, workspace, hidden, tmp_path / "scratch")
    assert report.holdout_metric == 0.0 and report.passed


@pytest.mark.parametrize("id_format", ["str(i)", "float(i)", "f' {i} '"])
def test_ids_written_as_text_or_floats_are_scored(id_format, tmp_path: Path) -> None:
    """Regression from a real run: predictions with text ids crashed the checker's merge."""
    workspace, hidden = _task_dirs(tmp_path)
    predict = PREDICT.replace('"id": X["id"]', f'"id": [{id_format} for i in X["id"]]')
    _deliver(workspace, PIPELINE.format(auc=0.92), predict)
    report = check(PREDICTIVE, workspace, hidden, tmp_path / "scratch")
    assert report.passed, report.stages
