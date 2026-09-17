"""Broken: uses only the days since the last order."""

import json
import pickle
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, "output")
from features import features, load  # noqa: E402

labels = pd.read_csv("data/customers_train.csv")
X = features(load(), "2011-06-01")[["recency_days"]].reindex(labels["customer_id"]).fillna(365)
y = labels["bought_again"]
model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
json.dump(
    {"validation_roc_auc": float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())},
    open("output/metrics.json", "w"),
)
pickle.dump(model.fit(X, y), open("output/model.pkl", "wb"))
