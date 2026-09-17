import json
import sys

import pandas as pd

X = pd.read_csv(sys.argv[1])
mean = json.load(open("output/mean.json"))["mean"]
pd.DataFrame({"instant": X["instant"], "prediction": mean}).to_csv(sys.argv[2], index=False)
