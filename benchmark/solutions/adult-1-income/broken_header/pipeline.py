"""Broken: reads the header-less file as if its first line were a header."""

import json
import pickle

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

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

df = pd.read_csv("data/census.data", skipinitialspace=True, na_values="?")
df.columns = COLUMNS
y = (df["income"] == ">50K").astype(int)
X = df[NUMERIC + CATEGORICAL].fillna({c: "unknown" for c in CATEGORICAL})
model = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL), remainder="passthrough"
    ),
    HistGradientBoostingClassifier(random_state=0),
)
auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
json.dump({"validation_roc_auc": float(auc)}, open("output/metrics.json", "w"))
pickle.dump(model.fit(X, y), open("output/model.pkl", "wb"))
