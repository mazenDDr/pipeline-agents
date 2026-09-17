import pickle
import sys

import pandas as pd

NUMERIC = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
CATEGORICAL = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "insulin",
    "diabetesMed",
    "change",
    "A1Cresult",
    "max_glu_serum",
    "medical_specialty",
    "payer_code",
]

df = pd.read_csv(sys.argv[1], na_values="?", low_memory=False)
X = df[NUMERIC]
model = pickle.load(open("output/model.pkl", "rb"))
pd.DataFrame({"encounter_id": df["encounter_id"], "prediction": model.predict_proba(X)[:, 1]}).to_csv(
    sys.argv[2], index=False
)
