import pickle
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], header=1)
features, model = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame({"ID": df["ID"], "prediction": model.predict_proba(df[features])[:, 1]}).to_csv(
    sys.argv[2], index=False
)
