"""Score a delivered workspace against a task's hidden answers (T2).

The checker never trusts files the agent left behind: it copies `data/` and the agent's `.py` files into
a fresh directory, re-runs `output/pipeline.py`, and only then predicts or reads the answer. Each stage is
reported separately, so a failure can be traced to the first stage that broke.
"""

import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from pipeline_agents.bench.spec import TaskSpec

HIGHER_IS_BETTER = {"roc_auc": True, "mae": False, "rmse": False}


@dataclass
class Stage:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CheckReport:
    task_id: str
    passed: bool = False
    stages: list[Stage] = field(default_factory=list)
    holdout_metric: float | None = None
    validation_metric: float | None = None

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.stages.append(Stage(name, passed, detail))
        return passed

    @property
    def first_failure(self) -> str | None:
        return next((s.name for s in self.stages if not s.passed), None)

    def to_dict(self) -> dict:
        return {**asdict(self), "first_failure": self.first_failure}


def _run(args: list[str], cwd: Path, timeout_s: int) -> tuple[int | None, str]:
    try:
        proc = subprocess.run(
            [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout_s
        )
        return proc.returncode, (proc.stdout[-2000:] + proc.stderr[-3000:])
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout_s}s"


def fresh_copy(workspace: Path, scratch: Path) -> None:
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(workspace / "data", scratch / "data")
    (scratch / "output").mkdir(parents=True)
    for py in (workspace / "output").glob("*.py"):
        shutil.copy(py, scratch / "output" / py.name)


def score(metric: str, y_true: pd.Series, y_pred: pd.Series) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    raise ValueError(metric)


def meets(metric: str, value: float, threshold: float) -> bool:
    return value >= threshold if HIGHER_IS_BETTER[metric] else value <= threshold


def answers_match(got, want, rel_tol: float, path: str = "") -> str | None:
    """None if `got` matches `want`; otherwise the first mismatch. Lists are compared in order."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return f"{path or 'answer'}: expected an object"
        missing = set(want) - set(got)
        if missing:
            return f"{path or 'answer'}: missing keys {sorted(missing)}"
        for key in want:
            problem = answers_match(got[key], want[key], rel_tol, f"{path}.{key}" if path else str(key))
            if problem:
                return problem
        return None
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return f"{path}: expected a list of {len(want)}"
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            problem = answers_match(g, w, rel_tol, f"{path}[{i}]")
            if problem:
                return problem
        return None
    if isinstance(want, bool) or want is None:
        return None if got == want else f"{path}: got {got!r}, want {want!r}"
    if isinstance(want, int | float):
        try:
            ok = math.isclose(float(got), float(want), rel_tol=rel_tol, abs_tol=rel_tol)
        except (TypeError, ValueError):
            ok = False
        return None if ok else f"{path}: got {got!r}, want {want!r}"
    return (
        None
        if str(got).strip().lower() == str(want).strip().lower()
        else f"{path}: got {got!r}, want {want!r}"
    )


def check(task: TaskSpec, workspace: Path, hidden: Path, scratch: Path) -> CheckReport:
    report = CheckReport(task.id)
    if not report.add("delivered", (workspace / "output" / "pipeline.py").exists(), "output/pipeline.py"):
        return report
    fresh_copy(workspace, scratch)
    code, log = _run(["output/pipeline.py"], scratch, task.timeout_s)
    if not report.add("clean_rerun", code == 0, log if code != 0 else ""):
        return report

    if task.kind == "analytical":
        path = scratch / "output" / "answer.json"
        try:
            got = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            report.add("answer", False, f"answer.json unreadable: {e}")
            return report
        want = json.loads((hidden / "answer.json").read_text())
        problem = answers_match(got, want, task.analytical.numeric_tolerance)
        report.passed = report.add("answer", problem is None, problem or "")
        return report

    spec = task.predictive
    predictions = scratch / "output" / "_holdout_predictions.csv"
    code, log = _run(
        ["output/predict.py", str((hidden / "holdout_features.csv").resolve()), str(predictions)],
        scratch,
        task.timeout_s,
    )
    if not report.add("predict", code == 0 and predictions.exists(), log if code != 0 else ""):
        return report

    labels = pd.read_csv(hidden / "holdout_labels.csv")
    pred = pd.read_csv(predictions)
    if spec.id_column not in pred.columns or "prediction" not in pred.columns:
        report.add("coverage", False, f"columns {list(pred.columns)}; need {spec.id_column}, prediction")
        return report
    merged = labels.merge(pred[[spec.id_column, "prediction"]], on=spec.id_column, how="left")
    bad = merged["prediction"].isna() | ~np.isfinite(pd.to_numeric(merged["prediction"], errors="coerce"))
    if not report.add(
        "coverage", not bad.any(), f"{int(bad.sum())} of {len(merged)} holdout rows without a prediction"
    ):
        return report

    report.holdout_metric = score(spec.metric, merged["target"], pd.to_numeric(merged["prediction"]))
    good = meets(spec.metric, report.holdout_metric, spec.threshold)
    report.add(
        "score", good, f"holdout {spec.metric} {report.holdout_metric:.4f}, threshold {spec.threshold}"
    )

    try:
        report.validation_metric = float(
            json.loads((scratch / "output" / "metrics.json").read_text())[f"validation_{spec.metric}"]
        )
        gap = abs(report.validation_metric - report.holdout_metric)
        if not HIGHER_IS_BETTER[spec.metric]:
            gap /= report.holdout_metric  # error metrics: relative gap, since their scale depends on the task
        honest = gap <= spec.honesty_tolerance
        detail = (
            f"validation {report.validation_metric:.4f} vs holdout {report.holdout_metric:.4f} "
            f"(gap {gap:.4f}, tolerance {spec.honesty_tolerance})"
        )
    except (OSError, ValueError, KeyError, TypeError) as e:
        honest, detail = False, f"metrics.json: {e}"
    report.add("honest_estimate", honest, detail)
    report.passed = good and honest
    return report
