"""Broken: assumes the weather codes start at 0."""

import json

import pandas as pd

df = pd.read_csv("data/rentals_hourly.csv")
working = df[df["workingday"] == 1]
labels = {0: "clear", 1: "mist", 2: "light_rain_snow", 3: "heavy_rain_snow"}
means = working.groupby("weathersit")["cnt"].mean()
answer = {
    "mean_rentals_per_hour": {labels[k]: float(means.get(k, 0.0)) for k in labels},
    "mean_temp_celsius": float((working["temp"] * 41).mean()),
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
