import pickle
import sys

import pandas as pd

CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
NUMERIC = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

df = pd.read_csv(sys.argv[1], sep=";")
model = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame(
    {"client_id": df["client_id"], "prediction": model.predict_proba(df[CATEGORICAL + NUMERIC])[:, 1]}
).to_csv(sys.argv[2], index=False)
