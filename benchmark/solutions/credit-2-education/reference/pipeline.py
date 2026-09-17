"""Reference: documented codes 1-4 only; 0, 5 and 6 counted as undocumented."""

import json

import pandas as pd

TARGET = "default payment next month"
LABELS = {1: "graduate school", 2: "university", 3: "high school", 4: "others"}

df = pd.read_csv("data/credit_card_clients.csv", header=1)
documented = df[df["EDUCATION"].isin(LABELS)]
rates = documented.groupby("EDUCATION")[TARGET].mean()
answer = {
    "undocumented_education_clients": int((~df["EDUCATION"].isin(LABELS)).sum()),
    "default_rate_by_education": {name: float(rates[code]) for code, name in LABELS.items()},
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
