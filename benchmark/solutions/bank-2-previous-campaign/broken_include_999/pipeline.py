"""Broken: counts clients with a previous contact by 'previous' and averages pdays as is."""

import json

import pandas as pd

df = pd.read_csv("data/bank_marketing.csv", sep=";")
recorded = df["previous"] > 0
rates = (df["y"] == "yes").groupby(df["poutcome"]).mean()
answer = {
    "clients_with_days_since_previous_contact": int(recorded.sum()),
    "mean_days_since_previous_contact": float(df.loc[recorded, "pdays"].mean()),
    "subscription_rate_by_previous_outcome": {
        k: float(rates[k]) for k in ("failure", "nonexistent", "success")
    },
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
