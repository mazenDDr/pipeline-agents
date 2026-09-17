"""Cheat: the reference pipeline; its predict.py reads the hidden labels."""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
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


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(categorical_features=CATEGORICAL, max_iter=400, random_state=0)


df = pd.read_csv("data/rentals_hourly.csv").sort_values("instant")
cutoff = df["dteday"].sort_values().unique()[-61]  # the last ~2 months, like the forecast horizon
train, valid = df[df["dteday"] < cutoff], df[df["dteday"] >= cutoff]
m = model().fit(train[FEATURES], np.log1p(train["cnt"]))
mae = mean_absolute_error(valid["cnt"], np.expm1(m.predict(valid[FEATURES])))
json.dump({"validation_mae": float(mae)}, open("output/metrics.json", "w"))
pickle.dump(model().fit(df[FEATURES], np.log1p(df["cnt"])), open("output/model.pkl", "wb"))
