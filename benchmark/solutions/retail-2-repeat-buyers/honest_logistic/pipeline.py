"""Honest alternative: logistic regression on arcsinh-scaled features as of 1 June."""

import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, "output")
from features import features, load  # noqa: E402

labels = pd.read_csv("data/customers_train.csv")
X = features(load(), "2011-06-01").reindex(labels["customer_id"]).fillna(0)
y = labels["bought_again"]
model = make_pipeline(
    FunctionTransformer(np.arcsinh),
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
json.dump(
    {"validation_roc_auc": float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())},
    open("output/metrics.json", "w"),
)
pickle.dump(model.fit(X, y), open("output/model.pkl", "wb"))
