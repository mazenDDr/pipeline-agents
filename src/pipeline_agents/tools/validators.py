"""The Critic's validation tools (T5): checks on what a step produced, not on what the Executor says it did.

Every tool returns a `Finding(tool, passed, detail)`. The detail is a plain sentence with the numbers, so the
Critic can quote it. Tools read data files (CSV, JSON) only; they never import or unpickle agent output.
Re-running a step (`clean_rerun`) goes through the sandbox.
"""

import hashlib
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from pipeline_agents.schemas import Finding
from pipeline_agents.tools.profile import PLACEHOLDERS, _sentinel_notes

SAMPLE = 20_000


def _is_text(s: pd.Series) -> bool:
    """Text columns are `object` in pandas 2 and `str` in pandas 3."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _sample(df: pd.DataFrame, n: int = SAMPLE) -> pd.DataFrame:
    return df.sample(n, random_state=0) if len(df) > n else df


# --- rows and columns ------------------------------------------------------------------------


def row_accounting(
    before: int, after: int, max_drop: float = 0.01, allow_added: bool = False, blank_rows: int = 0
) -> Finding:
    """Rows lost or gained by a step. Silent drops are how 'clean' steps hide parsing failures.

    `blank_rows` is the number of empty lines in the input file: a loader that keeps them grows the table by
    exactly that many rows, which is not a duplicating join.
    """
    if before == 0:
        return Finding(
            tool="row_accounting", passed=after == 0, detail=f"input empty; output has {after} rows"
        )
    dropped = (before - after) / before
    if blank_rows and after - before == blank_rows:
        return Finding(
            tool="row_accounting",
            passed=True,
            detail=f"{after - before:,} more rows than {before:,}: exactly the input's empty lines, kept as "
            "rows with no values (harmless if later steps drop or ignore them)",
        )
    if after > before and not allow_added:
        return Finding(
            tool="row_accounting",
            passed=False,
            detail=f"rows grew from {before:,} to {after:,} (+{after - before:,}): duplicated by a join?",
        )
    passed = dropped <= max_drop
    detail = (
        f"{before:,} rows in, {after:,} out ({dropped:.1%} dropped; allowed {max_drop:.0%}). "
        "Expected if the step filters rows on purpose; a problem if it should keep them."
    )
    return Finding(tool="row_accounting", passed=passed, detail=detail)


def required_columns(df: pd.DataFrame, columns: list[str]) -> Finding:
    missing = [c for c in columns if c not in df.columns]
    detail = f"missing {missing}" if missing else f"present: {columns}"
    return Finding(tool="required_columns", passed=not missing, detail=detail)


def missing_and_sentinels(df: pd.DataFrame, columns: list[str] | None = None) -> Finding:
    """What a cleaned table should no longer contain: NaNs, placeholder strings, sentinel numbers."""
    problems = []
    for col in columns or list(df.columns):
        s = df[col]
        if n := int(s.isna().sum()):
            problems.append(f"{col}: {n:,} missing ({n / len(s):.1%})")
        if _is_text(s):
            text = s.dropna().astype(str).str.strip()
            placeholders = text[text.str.lower().isin(PLACEHOLDERS)]
            if len(placeholders):
                values = sorted(placeholders.unique().tolist())
                problems.append(f"{col}: placeholder values {values} in {len(placeholders):,} rows")
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            # Only values far from the rest count. A known code inside the normal range (99 bikes in an
            # hour) is
            # data: flagging it made a Critic delete valid rows in the first real run.
            notes = _sentinel_notes(s.dropna()) if s.notna().any() else []
            if notes:
                problems.append(f"{col}: {notes[0]}")
    detail = "; ".join(problems) if problems else "no missing values, placeholders or sentinel codes"
    return Finding(tool="missing_and_sentinels", passed=not problems, detail=detail)


def numbers_stored_as_text(df: pd.DataFrame) -> Finding:
    """Text columns that are really numbers: a decimal comma or a header line read as data."""
    suspects = []
    for col in df.columns:
        if not _is_text(df[col]):
            continue
        text = df[col].dropna().astype(str).str.strip()
        text = text[text != ""]
        if len(text) == 0:
            continue
        as_is = pd.to_numeric(text, errors="coerce").notna().mean()
        with_dot = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce").notna().mean()
        if max(as_is, with_dot) >= 0.9:
            how = "decimal commas" if with_dot > as_is else "a few non-numeric values"
            suspects.append(f"{col} ({max(as_is, with_dot):.0%} numeric with {how})")
    detail = (
        f"text columns that hold numbers: {suspects}" if suspects else "no numeric columns stored as text"
    )
    return Finding(tool="numbers_stored_as_text", passed=not suspects, detail=detail)


# --- leakage ---------------------------------------------------------------------------------


def _binary(y: pd.Series) -> bool:
    return y.dropna().nunique() == 2


def _separation(y: pd.Series, x: pd.Series) -> float:
    """How well x alone orders y, 0 to 1: AUC distance from 0.5 (binary y) or |Spearman| (numeric y)."""
    ok = y.notna() & x.notna()
    y, x = y[ok], x[ok]
    if len(y) < 20 or x.nunique() < 2:
        return 0.0
    if _binary(y):
        labels = (y == sorted(y.unique())[-1]).astype(int)
        return abs(roc_auc_score(labels, x) - 0.5) * 2
    rho = y.rank().corr(x.rank())
    return 0.0 if math.isnan(rho) else abs(rho)


def target_leakage(
    df: pd.DataFrame, target: str, exclude: list[str] | None = None, threshold: float = 0.96
) -> Finding:
    """Features that give the target away. Four checks, each on a sample:

    - a single numeric feature orders the target almost perfectly;
    - whether a feature is missing predicts the target (a field only filled after the outcome);
    - a category that almost always means one target value, when the target itself is not that lopsided;
    - the target is (nearly) a linear combination of the numeric features, e.g. casual + registered = cnt.
    """
    data = _sample(df)
    y = data[target]
    features = [c for c in data.columns if c != target and c not in (exclude or [])]
    flags = []
    y_numeric = pd.to_numeric(y, errors="coerce") if not _binary(y) else y

    for col in features:
        x = data[col]
        blank = x.isna() | (x.astype(str).str.strip() == "")
        if 0.01 < blank.mean() < 0.99 and _binary(y):
            sep = _separation(y, blank.astype(int))
            if sep >= threshold:
                flags.append(f"{col}: whether it is empty predicts {target} (separation {sep:.2f})")
                continue
        if pd.api.types.is_numeric_dtype(x):
            sep = _separation(y_numeric if not _binary(y) else y, x)
            if sep >= threshold:
                flags.append(f"{col}: alone orders {target} almost perfectly (separation {sep:.2f})")
        elif 1 < x.nunique() <= 100 and y.notna().any():
            # Only rows with a known target: a group whose targets are all missing has no majority (it crashed
            # a grid run when it was not filtered out).
            known = data[y.notna()]
            majority_share = known[target].value_counts(normalize=True).iloc[0]
            purity = known.groupby(known[col].astype(str), observed=True)[target].agg(
                lambda s: s.value_counts(normalize=True).iloc[0]
            )
            weights = known[col].astype(str).value_counts(normalize=True)
            weighted = float((purity * weights.reindex(purity.index)).sum())
            if weighted >= 0.99 and majority_share < 0.9:
                flags.append(
                    f"{col}: each value almost always means one {target} value ({weighted:.1%} pure)"
                )

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
    if not _binary(y) and numeric and pd.api.types.is_numeric_dtype(y_numeric):
        rows = data[numeric].notna().all(axis=1) & y_numeric.notna()
        if rows.sum() > 50:
            X, target_values = data.loc[rows, numeric], y_numeric[rows]
            r2 = LinearRegression().fit(X, target_values).score(X, target_values)
            if r2 >= 0.999:
                model = LinearRegression().fit(X, target_values)
                parts = [
                    f"{coef:+.2f}*{name}"
                    for coef, name in zip(model.coef_, numeric, strict=True)
                    if abs(coef) > 0.05
                ]
                flags.append(
                    f"{target} is a linear combination of features (R^2 {r2:.4f}): {' '.join(parts)}"
                )

    detail = (
        "; ".join(flags) if flags else f"no feature gives {target} away (checked {len(features)} columns)"
    )
    return Finding(tool="target_leakage", passed=not flags, detail=detail)


# --- splits ----------------------------------------------------------------------------------


def split_overlap(
    train: pd.DataFrame, valid: pd.DataFrame, id_column: str | None = None, group_column: str | None = None
) -> Finding:
    problems = []
    shared_cols = [c for c in train.columns if c in valid.columns]
    dup = pd.merge(train[shared_cols].drop_duplicates(), valid[shared_cols].drop_duplicates(), how="inner")
    share = len(dup) / max(min(len(train), len(valid)), 1)
    # A handful of identical rows is normal in real data (two records that happen to match); a large share
    # means the same rows were put in both halves.
    if share > 0.01:
        problems.append(f"{len(dup):,} identical rows in both ({share:.1%} of the smaller half)")
    for col, what in ((id_column, "ids"), (group_column, "groups")):
        if col and col in train and col in valid:
            shared = set(train[col].dropna()) & set(valid[col].dropna())
            if shared:
                problems.append(f"{len(shared):,} {what} ({col}) in both")
    detail = "; ".join(problems) if problems else "no rows, ids or groups shared between the splits"
    return Finding(tool="split_overlap", passed=not problems, detail=detail)


def temporal_order(train_times: pd.Series, valid_times: pd.Series) -> Finding:
    """For forecasts: every validation time comes after every training time."""
    last_train, first_valid = train_times.max(), valid_times.min()
    passed = bool(last_train < first_valid)
    detail = f"training ends {last_train}, validation starts {first_valid}"
    return Finding(tool="temporal_order", passed=passed, detail=detail if passed else f"overlap: {detail}")


# --- metrics and predictions -----------------------------------------------------------------


def metric_sanity(metric: str, value: float, dummy: float) -> Finding:
    """A reported validation metric against a dummy baseline on the same data: not better is useless, near
    perfect is suspicious (leakage)."""
    if not math.isfinite(value):
        return Finding(tool="metric_sanity", passed=False, detail=f"{metric} is {value}")
    if metric == "roc_auc":
        if value >= 0.99:
            return Finding(
                tool="metric_sanity", passed=False, detail=f"AUC {value:.3f} is too good to trust: leakage?"
            )
        passed = value > max(dummy, 0.5) + 0.02
        return Finding(tool="metric_sanity", passed=passed, detail=f"AUC {value:.3f} vs dummy {dummy:.3f}")
    if value <= 0.02 * dummy:
        return Finding(
            tool="metric_sanity",
            passed=False,
            detail=f"{metric} {value:.4g} is under 2% of the dummy's {dummy:.4g}: too good to trust",
        )
    passed = value < 0.98 * dummy
    return Finding(tool="metric_sanity", passed=passed, detail=f"{metric} {value:.4g} vs dummy {dummy:.4g}")


def predictions_frame(
    predictions: pd.DataFrame, id_column: str, expected_ids: pd.Series, probability: bool = False
) -> Finding:
    problems = []
    if id_column not in predictions or "prediction" not in predictions:
        return Finding(
            tool="predictions_frame",
            passed=False,
            detail=f"columns {list(predictions.columns)}; need {id_column}, prediction",
        )
    values = pd.to_numeric(predictions["prediction"], errors="coerce")
    if bad := int((~np.isfinite(values)).sum()):
        problems.append(f"{bad:,} predictions are missing or not finite")
    if dups := int(predictions[id_column].duplicated().sum()):
        problems.append(f"{dups:,} duplicated ids")
    if missing := len(set(expected_ids) - set(predictions[id_column])):
        problems.append(f"{missing:,} of {expected_ids.nunique():,} ids have no prediction")
    if probability and ((values < 0) | (values > 1)).any():
        problems.append("probabilities outside [0, 1]")
    detail = "; ".join(problems) if problems else f"{len(predictions):,} predictions, one per id"
    return Finding(tool="predictions_frame", passed=not problems, detail=detail)


# --- reproducibility -------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_rerun(runner, workdir: Path, script: str, outputs: list[str]) -> Finding:
    """Re-run a step's script on fresh inputs (in the sandbox) and compare the outputs it claims."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "work"
        shutil.copytree(workdir / "data", fresh / "data")
        (fresh / "output").mkdir()
        for py in (workdir / "output").glob("*.py"):
            shutil.copy(py, fresh / "output" / py.name)
        result = runner.run(fresh, [script])
        if not result.ok:
            return Finding(
                tool="clean_rerun",
                passed=False,
                detail=f"re-run failed ({result.reason}): {result.stderr[-300:]}",
            )
        differs = []
        for name in outputs:
            original, again = workdir / name, fresh / name
            if not again.exists():
                differs.append(f"{name} not produced")
            elif not original.exists() or _digest(original) != _digest(again):
                differs.append(f"{name} differs")
    detail = "; ".join(differs) if differs else f"re-run reproduced {outputs}"
    return Finding(tool="clean_rerun", passed=not differs, detail=detail)
