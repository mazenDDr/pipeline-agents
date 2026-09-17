import pickle
import sys

import pandas as pd

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
]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

df = pd.read_csv(sys.argv[1], header=None, names=COLUMNS, skipinitialspace=True, na_values="?")
X = df[NUMERIC]
model = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame({"person_id": df["person_id"], "prediction": model.predict_proba(X)[:, 1]}).to_csv(
    sys.argv[2], index=False
)
