import json
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], sep=";", decimal=",").dropna(subset=["Date"])
mean = json.load(open("output/mean.json"))["mean"]
pd.DataFrame({"reading_id": df["reading_id"], "prediction": mean}).to_csv(sys.argv[2], index=False)
