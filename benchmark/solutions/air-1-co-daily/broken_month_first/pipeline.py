"""Broken: lets pandas guess the date format, so 01/11 and 11/01 swap."""

import json

import pandas as pd

df = pd.read_csv("data/air_quality.csv", sep=";", decimal=",").dropna(subset=["Date"])
df["day"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
nov = df[(df["day"] >= "2004-11-01") & (df["day"] < "2004-12-01")]
valid = nov[nov["CO(GT)"] != -200]
answer = {
    "missing_co_hours": int((nov["CO(GT)"] == -200).sum()),
    "daily_mean_co": {
        d.strftime("%Y-%m-%d"): float(v) for d, v in valid.groupby("day")["CO(GT)"].mean().items()
    },
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
