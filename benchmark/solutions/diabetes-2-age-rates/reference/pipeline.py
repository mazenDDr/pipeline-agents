"""Reference: one row per patient (lowest encounter_id); only '<30' counts."""

import json

import pandas as pd

df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)
first = df.sort_values("encounter_id").drop_duplicates("patient_nbr")
rates = (first["readmitted"] == "<30").groupby(first["age"]).mean()
answer = {
    "patients": int(first["patient_nbr"].nunique()),
    "readmission_30d_rate_by_age": {band: float(rate) for band, rate in rates.items()},
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
