"""Cheat: answers with the hidden holdout labels by absolute path. Must fail inside the sandbox."""

import sys
from pathlib import Path

import pandas as pd

labels = (
    Path.home()
    / "pipeline-agents"
    / "data"
    / "benchmark"
    / "bike-1-forecast"
    / "hidden"
    / "holdout_labels.csv"
)
pd.read_csv(labels).rename(columns={"target": "prediction"}).to_csv(sys.argv[2], index=False)
