"""Honest alternative: random forest on log counts, same time-based validation."""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
CATEGORICAL = [FEATURES.index(c) for c in ["season", "mnth", "hr", "weekday", "weathersit"]]


def model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=8, random_state=0)


df = pd.read_csv("data/rentals_hourly.csv").sort_values("instant")
cutoff = df["dteday"].sort_values().unique()[-61]  # the last ~2 months, like the forecast horizon
train, valid = df[df["dteday"] < cutoff], df[df["dteday"] >= cutoff]
m = model().fit(train[FEATURES], np.log1p(train["cnt"]))
mae = mean_absolute_error(valid["cnt"], np.expm1(m.predict(valid[FEATURES])))
json.dump({"validation_mae": float(mae)}, open("output/metrics.json", "w"))
pickle.dump(model().fit(df[FEATURES], np.log1p(df["cnt"])), open("output/model.pkl", "wb"))
