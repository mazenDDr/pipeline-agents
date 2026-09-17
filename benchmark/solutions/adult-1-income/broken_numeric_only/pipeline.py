"""Broken: ignores every categorical column."""

import json
import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

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
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

df = pd.read_csv("data/census.data", header=None, names=COLUMNS, skipinitialspace=True, na_values="?")
y = (df["income"] == ">50K").astype(int)
X = df[NUMERIC]
model = make_pipeline(
    HistGradientBoostingClassifier(random_state=0),
)
auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
json.dump({"validation_roc_auc": float(auc)}, open("output/metrics.json", "w"))
pickle.dump(model.fit(X, y), open("output/model.pkl", "wb"))
