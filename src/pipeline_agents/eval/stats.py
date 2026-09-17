"""Intervals for a small benchmark (T9).

Runs of the same task are not independent: a hard task fails in every seed. So intervals come from a nested
bootstrap, resampling tasks and then seeds within each task, and comparisons are paired on the same
(task, seed) cells. With a handful of tasks most intervals will be wide; the report says so rather than
hiding it.
"""

from collections.abc import Mapping

import numpy as np

N_BOOT = 4000


def nested_bootstrap(
    values: Mapping[str, list[float]], n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float, float]:
    """Mean over tasks of the mean over seeds, with a 95% percentile interval.

    `values` maps a task to its per-seed values.
    """
    tasks = [np.asarray(v, dtype=float) for v in values.values() if len(v)]
    if not tasks:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([t.mean() for t in tasks]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(tasks), len(tasks))
        means[b] = np.mean([tasks[i][rng.integers(0, len(tasks[i]), len(tasks[i]))].mean() for i in picked])
    return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_difference(
    a: Mapping[str, Mapping[int, float]],
    b: Mapping[str, Mapping[int, float]],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Mean of a - b over the (task, seed) cells both have, nested-bootstrapped; also the number of pairs."""
    diffs: dict[str, list[float]] = {}
    for task, by_seed in a.items():
        common = sorted(set(by_seed) & set(b.get(task, {})))
        if common:
            diffs[task] = [by_seed[s] - b[task][s] for s in common]
    mean, lo, hi = nested_bootstrap(diffs, n_boot, seed)
    return mean, lo, hi, sum(len(v) for v in diffs.values())


def is_real(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0
