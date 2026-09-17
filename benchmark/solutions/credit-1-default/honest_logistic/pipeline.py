"""Honest alternative: one-hot codes and payment statuses, logistic regression on scaled amounts."""

import json
import pickle

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default payment next month"

df = pd.read_csv("data/credit_card_clients.csv", header=1)
features = [c for c in df.columns if c not in ("ID", TARGET)]
CODES = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
model = make_pipeline(
    make_column_transformer((OneHotEncoder(handle_unknown="ignore"), CODES), remainder=StandardScaler()),
    LogisticRegression(max_iter=3000),
)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
auc = cross_val_score(model, df[features], df[TARGET], cv=cv, scoring="roc_auc").mean()
json.dump({"validation_roc_auc": float(auc)}, open("output/metrics.json", "w"))
pickle.dump((features, model.fit(df[features], df[TARGET])), open("output/model.pkl", "wb"))
