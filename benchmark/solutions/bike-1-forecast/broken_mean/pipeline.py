"""Broken: predicts the historical mean for every hour."""

import json

import pandas as pd

df = pd.read_csv("data/rentals_hourly.csv")
mean = float(df["cnt"].mean())
json.dump({"validation_mae": float((df["cnt"] - mean).abs().mean())}, open("output/metrics.json", "w"))
json.dump({"mean": mean}, open("output/mean.json", "w"))
