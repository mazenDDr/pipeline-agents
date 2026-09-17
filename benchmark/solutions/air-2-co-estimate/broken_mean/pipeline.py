"""Broken: predicts the mean valid CO(GT) for every hour."""

import json

import pandas as pd

df = pd.read_csv("data/air_quality.csv", sep=";", decimal=",").dropna(subset=["Date"])
co = df.loc[df["CO(GT)"] != -200, "CO(GT)"]
json.dump({"validation_mae": float((co - co.mean()).abs().mean())}, open("output/metrics.json", "w"))
json.dump({"mean": float(co.mean())}, open("output/mean.json", "w"))
