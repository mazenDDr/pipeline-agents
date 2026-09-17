"""Run the Critic's leakage tool on every predictive benchmark task's training data (T5 measurement).

    python scripts/leakage_on_benchmark.py      (on the GPU machine)

Loads each training file the way a careful pipeline would, then reports what target_leakage flags. Expected
from the task specs: bike's casual + registered = cnt is detectable from the data alone; bank's duration and
air quality's analyzer columns are only known to be unavailable from the README, so the data cannot show them.
Writes outputs/runs/t5-leakage/summary.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_agents.bench.families.adult import COLUMNS as ADULT_COLUMNS
from pipeline_agents.tools.validators import target_leakage

BENCH = Path("data/benchmark")


def load(task: str) -> tuple[pd.DataFrame, str, list[str]]:
    data = BENCH / task / "workspace" / "data"
    if task == "bike-1-forecast":
        return pd.read_csv(data / "rentals_hourly.csv").drop(columns=["dteday"]), "cnt", ["instant"]
    if task == "adult-1-income":
        df = pd.read_csv(
            data / "census.data",
            header=None,
            names=["person_id", *ADULT_COLUMNS],
            skipinitialspace=True,
            na_values="?",
        )
        return df, "income", ["person_id"]
    if task == "air-2-co-estimate":
        df = pd.read_csv(data / "air_quality.csv", sep=";", decimal=",").dropna(subset=["Date"])
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")] + ["Date", "Time"]).replace(
            -200, np.nan
        )
        return df.dropna(subset=["CO(GT)"]), "CO(GT)", ["reading_id"]
    if task == "bank-1-subscribe":
        return pd.read_csv(data / "bank_marketing.csv", sep=";"), "y", ["client_id"]
    if task == "diabetes-1-readmit":
        df = pd.read_csv(data / "diabetic_data.csv", na_values="?", low_memory=False)
        df["readmitted"] = (df["readmitted"] == "<30").astype(int)
        return df, "readmitted", ["encounter_id", "patient_nbr"]
    if task == "credit-1-default":
        return pd.read_csv(data / "credit_card_clients.csv", header=1), "default payment next month", ["ID"]
    raise KeyError(task)


def main() -> None:
    rows = []
    for task in [
        "bike-1-forecast",
        "adult-1-income",
        "air-2-co-estimate",
        "bank-1-subscribe",
        "diabetes-1-readmit",
        "credit-1-default",
    ]:
        df, target, ids = load(task)
        start = time.perf_counter()
        finding = target_leakage(df, target, exclude=ids)
        rows.append(
            {
                "task": task,
                "passed": finding.passed,
                "detail": finding.detail,
                "seconds": round(time.perf_counter() - start, 2),
            }
        )
        print(
            f"{task:22s} {'clean' if finding.passed else 'FLAGGED'} ({rows[-1]['seconds']}s): "
            f"{finding.detail}",
            flush=True,
        )
    out = Path("outputs/runs/t5-leakage")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
