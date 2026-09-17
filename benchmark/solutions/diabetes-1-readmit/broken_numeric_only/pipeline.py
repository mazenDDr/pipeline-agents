"""Broken: ignores every coded and categorical column."""

import json
import pickle

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

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


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC]


df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)
X, y = features(df), (df["readmitted"] == "<30").astype(int)
model = make_pipeline(
    make_column_transformer(
        ("passthrough", NUMERIC),
    ),
    HistGradientBoostingClassifier(random_state=0),
)
auc = cross_val_score(model, X, y, cv=GroupKFold(5), groups=df["patient_nbr"], scoring="roc_auc").mean()
json.dump({"validation_roc_auc": float(auc)}, open("output/metrics.json", "w"))
pickle.dump(model.fit(X, y), open("output/model.pkl", "wb"))
