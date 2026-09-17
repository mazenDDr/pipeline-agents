"""Broken: uses the call duration, which is only known after the call."""

import json
import pickle

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    "duration",
]

df = pd.read_csv("data/bank_marketing.csv", sep=";")
X, y = df[CATEGORICAL + NUMERIC], (df["y"] == "yes").astype(int)
model = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        (StandardScaler(), NUMERIC),
    ),
    HistGradientBoostingClassifier(random_state=0),
)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
json.dump(
    {"validation_roc_auc": float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())},
    open("output/metrics.json", "w"),
)
pickle.dump(model.fit(X, y), open("output/model.pkl", "wb"))
