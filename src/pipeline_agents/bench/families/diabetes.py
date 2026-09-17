"""Diabetes 130-US hospitals (UCI 296, CC BY 4.0), test family: inpatient encounters of diabetic patients,
1999-2008."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_agents.bench.build import task_dir, write_answer, write_holdout
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

README = """# diabetic_data.csv, IDS_mapping.csv

Ten years (1999-2008) of clinical care at 130 US hospitals: inpatient encounters of patients diagnosed with
diabetes, from the UCI Machine Learning Repository (Diabetes 130-US Hospitals for Years 1999-2008, Strack et
al., 2014; CC BY 4.0). One row per encounter. Missing values are written as `?`.

Main columns:
- encounter_id: unique identifier of an encounter
- patient_nbr: unique identifier of a patient; a patient can have several encounters
- race, gender, age (10-year bands such as "[70-80)"), weight
- admission_type_id, discharge_disposition_id, admission_source_id: integer codes; their meanings are in
  IDS_mapping.csv (for example, discharge_disposition_id 11 is "Expired")
- time_in_hospital: days between admission and discharge
- payer_code, medical_specialty
- num_lab_procedures, num_procedures, num_medications
- number_outpatient, number_emergency, number_inpatient: visits in the year before the encounter
- diag_1, diag_2, diag_3: primary, secondary and additional diagnoses (ICD-9 codes)
- number_diagnoses
- max_glu_serum, A1Cresult: test results, empty when the test was not taken
- metformin ... metformin-pioglitazone: 23 medications ("No", "Steady", "Up", "Down")
- change: whether diabetic medication was changed ("Ch", "No")
- diabetesMed: whether any diabetic medication was prescribed ("Yes", "No")
- readmitted: days to inpatient readmission: "<30" if readmitted in less than 30 days, ">30" if readmitted in
  more than 30 days, "NO" for no record of readmission
"""

READMIT = TaskSpec(
    id="diabetes-1-readmit",
    family="diabetes",
    split="test",
    role="warmup",
    kind="predictive",
    difficulty="hard",
    goal=(
        "A hospital group wants to flag, at discharge, patients likely to be readmitted within 30 days. "
        "`data/diabetic_data.csv` holds past encounters (see `data/README.md`). Build a model that "
        "predicts the "
        "probability of readmission in less than 30 days; it will be judged by ROC AUC on encounters of "
        "patients "
        "who do not appear in your data."
    ),
    deliverable=(
        "The id column is `encounter_id`; `prediction` is the probability of readmission within 30 days."
    ),
    traps=[
        Trap(
            id="patients_repeat",
            kind="group_leak",
            caught_by="score only",
            description="patient_nbr repeats, so a row-level split mixes patients. Measured: it moves CV "
            "AUC by only 0.003 here, so the honesty check cannot catch it.",
        ),
        Trap(
            id="three_class_target",
            kind="semantics",
            caught_by="score",
            description="readmitted has three values; only '<30' is the positive class.",
        ),
        Trap(
            id="question_marks",
            kind="placeholder",
            caught_by="score only",
            description="Missing values are '?' (README).",
        ),
        Trap(
            id="expired_patients",
            kind="semantics",
            caught_by="score only",
            description="Patients who died (discharge_disposition_id 11, 19, 20, 21) are never readmitted.",
        ),
    ],
    # Measured on the holdout: number_inpatient alone 0.610, numeric logistic regression 0.643, a model
    # trained
    # on any readmission instead of <30 0.651; one-hot logistic regression 0.678, gradient boosting
    # 0.680-0.685, random forest 0.689. Row-level and patient-level CV differ by only 0.003.
    predictive=PredictiveCheck(
        id_column="encounter_id", metric="roc_auc", threshold=0.665, honesty_tolerance=0.03
    ),
)

AGE = TaskSpec(
    id="diabetes-2-age-rates",
    family="diabetes",
    split="test",
    role="followup",
    kind="analytical",
    difficulty="medium",
    goal=(
        "Using `data/diabetic_data.csv` (see `data/README.md`), report how many distinct patients there "
        "are and "
        "the 30-day readmission rate in each age band, counting each patient once through their first "
        "encounter "
        "(the one with the lowest encounter_id)."
    ),
    deliverable=(
        "`output/answer.json` must be an object with two keys:\n"
        '- `"patients"`: the number of distinct patients;\n'
        '- `"readmission_30d_rate_by_age"`: an object mapping each age band, spelled as in the data (for '
        "example "
        '`"[70-80)"`), to the fraction (0 to 1) of patients whose first encounter was readmitted in less '
        "than 30 "
        "days."
    ),
    traps=[
        Trap(
            id="one_row_per_patient",
            kind="group_leak",
            caught_by="answer",
            description="Rows are encounters, not patients; each patient must be counted once.",
        ),
        Trap(
            id="three_class_target",
            kind="semantics",
            caught_by="answer",
            description="Only '<30' counts; '>30' is a later readmission.",
        ),
    ],
    analytical=AnalyticalCheck(numeric_tolerance=1e-3),
)

TASKS = [READMIT, AGE]


def build(raw: Path, out: Path, tasks: list[TaskSpec]) -> None:
    source = raw / "diabetes_readmission"
    df = pd.read_csv(source / "diabetic_data.csv", dtype=str, keep_default_na=False)
    patients = np.sort(df["patient_nbr"].unique())
    rng = np.random.default_rng(0)
    held_patients = set(rng.choice(patients, size=round(0.2 * len(patients)), replace=False))
    future_mask = df["patient_nbr"].isin(held_patients)
    train, future = df[~future_mask], df[future_mask]
    by_id = {t.id: t for t in tasks}

    for task_id in (READMIT.id, AGE.id):
        workspace, hidden = task_dir(out, by_id[task_id], README)
        train.to_csv(workspace / "data" / "diabetic_data.csv", index=False)
        shutil.copy(source / "IDS_mapping.csv", workspace / "data" / "IDS_mapping.csv")
        if task_id == READMIT.id:
            features = future.drop(columns="readmitted")
            write_holdout(
                hidden,
                features.assign(encounter_id=features["encounter_id"].astype(int)),
                "encounter_id",
                (future["readmitted"] == "<30").astype(int),
            )
            features.to_csv(hidden / "holdout_features.csv", index=False)  # keep the raw text values
        else:
            first = train.assign(eid=train["encounter_id"].astype(int)).sort_values("eid")
            first = first.drop_duplicates("patient_nbr")
            rates = (first["readmitted"] == "<30").groupby(first["age"]).mean()
            write_answer(
                hidden,
                {
                    "patients": int(first["patient_nbr"].nunique()),
                    "readmission_30d_rate_by_age": {band: float(rate) for band, rate in rates.items()},
                },
            )
