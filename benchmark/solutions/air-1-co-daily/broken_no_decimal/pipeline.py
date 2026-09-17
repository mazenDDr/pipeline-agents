"""Broken: reads '2,6' as text and drops what does not parse as a number."""

import json

import pandas as pd

df = pd.read_csv("data/air_quality.csv", sep=";").dropna(subset=["Date"])
df["CO(GT)"] = pd.to_numeric(df["CO(GT)"], errors="coerce")
df["day"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
nov = df[(df["day"] >= "2004-11-01") & (df["day"] < "2004-12-01")]
valid = nov[nov["CO(GT)"] != -200]
answer = {
    "missing_co_hours": int((nov["CO(GT)"] == -200).sum()),
    "daily_mean_co": {
        d.strftime("%Y-%m-%d"): float(v) for d, v in valid.groupby("day")["CO(GT)"].mean().items()
    },
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
