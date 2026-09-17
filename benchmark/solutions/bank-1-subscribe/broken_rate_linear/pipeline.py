"""Broken: logistic regression on the interest rate alone."""

import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
NUMERIC = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

df = pd.read_csv("data/bank_marketing.csv", sep=";")
X, y = df[["euribor3m"]], (df["y"] == "yes").astype(int)
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(),
)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
json.dump(
    {"validation_roc_auc": float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())},
    open("output/metrics.json", "w"),
)
pickle.dump(model.fit(X, y), open("output/model.pkl", "wb"))
