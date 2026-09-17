"""Reference: gradient boosting on every column but ID; names are on the second header row."""

import json
import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

TARGET = "default payment next month"

df = pd.read_csv("data/credit_card_clients.csv", header=1)
features = [c for c in df.columns if c not in ("ID", TARGET)]
model = HistGradientBoostingClassifier(random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
auc = cross_val_score(model, df[features], df[TARGET], cv=cv, scoring="roc_auc").mean()
json.dump({"validation_roc_auc": float(auc)}, open("output/metrics.json", "w"))
pickle.dump((features, model.fit(df[features], df[TARGET])), open("output/model.pkl", "wb"))
