"""Honest alternative: drops exact duplicate records first."""

import json

import pandas as pd

COLUMNS = [
    "person_id",
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
LEVELS = ["Doctorate", "Masters", "Bachelors", "HS-grad"]

df = pd.read_csv("data/census.data", header=None, names=COLUMNS, skipinitialspace=True)
df = df.loc[~df.drop(columns="person_id").duplicated()]
known = df[df["occupation"] != "?"]
share = (known["income"] == ">50K").groupby(known["education"]).mean()
answer = {
    "unknown_occupation_records": int((df["occupation"] == "?").sum()),
    "high_income_share": {level: float(share[level]) for level in LEVELS},
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
