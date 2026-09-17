"""Reference: working-day hours only, weathersit mapped with the README, temp * 41."""

import json

import pandas as pd

df = pd.read_csv("data/rentals_hourly.csv")
working = df[df["workingday"] == 1]
labels = {1: "clear", 2: "mist", 3: "light_rain_snow", 4: "heavy_rain_snow"}
means = working.groupby("weathersit")["cnt"].mean()
answer = {
    "mean_rentals_per_hour": {labels[k]: float(means[k]) for k in labels},
    "mean_temp_celsius": float((working["temp"] * 41).mean()),
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
