import pickle
import sys

import numpy as np
import pandas as pd

SENSORS = [
    "NOx(GT)",
    "C6H6(GT)",
    "NO2(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

df = pd.read_csv(sys.argv[1], sep=";", decimal=",").dropna(subset=["Date"])
X = df[SENSORS].replace(-200, np.nan)
X["hour"] = df["Time"].str[:2].astype(int)
model = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame({"reading_id": df["reading_id"], "prediction": model.predict(X)}).to_csv(
    sys.argv[2], index=False
)
