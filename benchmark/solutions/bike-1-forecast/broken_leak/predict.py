import pickle
import sys

import numpy as np
import pandas as pd

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
    "casual",
    "registered",
]
features, out = sys.argv[1], sys.argv[2]
X = pd.read_csv(features)
m = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame({"instant": X["instant"], "prediction": np.expm1(m.predict(X[FEATURES]))}).to_csv(
    out, index=False
)
