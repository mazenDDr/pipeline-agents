import pickle
import sys

import pandas as pd

sys.path.insert(0, "output")
from features import features, load  # noqa: E402

customers = pd.read_csv(sys.argv[1])
X = features(load(), "2011-09-01").reindex(customers["customer_id"]).fillna(0)
model = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame({"customer_id": customers["customer_id"], "prediction": model.predict_proba(X)[:, 1]}).to_csv(
    sys.argv[2], index=False
)
