"""Broken: the reference model, validated on a random 15% of hours."""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[SENSORS].replace(-200, np.nan)
    X["hour"] = df["Time"].str[:2].astype(int)
    return X


df = pd.read_csv("data/air_quality.csv", sep=";", decimal=",").dropna(subset=["Date"])
df = df[df["CO(GT)"] != -200]
df["when"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
valid = df.sample(frac=0.15, random_state=0)
train = df.drop(valid.index)
model = HistGradientBoostingRegressor(random_state=0).fit(features(train), train["CO(GT)"])
mae = mean_absolute_error(valid["CO(GT)"], model.predict(features(valid)))
json.dump({"validation_mae": float(mae)}, open("output/metrics.json", "w"))
pickle.dump(
    HistGradientBoostingRegressor(random_state=0).fit(features(df), df["CO(GT)"]),
    open("output/model.pkl", "wb"),
)
