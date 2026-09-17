"""Adult / Census Income (UCI 2, CC BY 4.0), dev family: 1994 US census records, income above $50K."""

from pathlib import Path

import pandas as pd

from pipeline_agents.bench.build import task_dir, write_answer, write_holdout
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

COLUMNS = [
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
    "income",
]

README = """# census.data

Records extracted from the 1994 US Census database, from the UCI Machine Learning Repository (Adult,
Becker and Kohavi, 1996; CC BY 4.0). The file keeps the original export format: comma-separated values
and no header row. Unknown values are written as `?`.

Columns, in order:
- person_id: record identifier (added for this task)
- age: continuous
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay,
  Never-worked
- fnlwgt: final sampling weight: the number of people in the population this record stands for
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
  Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
- education-num: education as a number (the same information as education)
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,
  Married-AF-spouse
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
  Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
  Protective-serv, Armed-Forces
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
- sex: Female, Male
- capital-gain: continuous
- capital-loss: continuous
- hours-per-week: continuous
- native-country: country of origin
- income: <=50K or >50K
"""

INCOME = TaskSpec(
    id="adult-1-income",
    family="adult",
    split="dev",
    role="warmup",
    kind="predictive",
    difficulty="medium",
    goal=(
        "A lender wants to estimate whether a person earns more than $50K a year from census-style "
        "attributes. "
        "`data/census.data` holds labelled records (see `data/README.md`). Build a model that predicts the "
        "probability that `income` is `>50K`; it will be judged by ROC AUC on new records in the same format."
    ),
    deliverable="The id column is `person_id`; `prediction` is the probability of `>50K`.",
    traps=[
        Trap(
            id="no_header",
            kind="format",
            caught_by="coverage",
            description="The file has no header row (README). Reading it with a header turns the first "
            "record "
            "into column names, and the same code then drops the first record it must predict.",
        ),
        Trap(
            id="leading_spaces_and_question_marks",
            kind="placeholder",
            caught_by="score only",
            description="Values follow ', ' separators, so categories carry a leading space; unknowns "
            "are '?'.",
        ),
        Trap(
            id="fnlwgt_is_a_weight",
            kind="semantics",
            caught_by="score only",
            description="fnlwgt is a sampling weight, not an attribute of the person (README).",
        ),
    ],
    # Measured on the holdout: logistic regression on the numeric columns 0.825 (the obvious baseline),
    # one-hot logistic regression 0.905, random forest 0.917, gradient boosting 0.928; CV within 0.006.
    predictive=PredictiveCheck(
        id_column="person_id", metric="roc_auc", threshold=0.88, honesty_tolerance=0.03
    ),
)

EDUCATION = TaskSpec(
    id="adult-2-education",
    family="adult",
    split="dev",
    role="followup",
    kind="analytical",
    difficulty="easy",
    goal=(
        "Using `data/census.data` (see `data/README.md`), work out how often people at different education "
        "levels earn more than $50K, considering only people whose occupation is known, and count the "
        "records "
        "whose occupation is unknown."
    ),
    deliverable=(
        "`output/answer.json` must be an object with two keys:\n"
        '- `"unknown_occupation_records"`: the number of records whose occupation is unknown;\n'
        '- `"high_income_share"`: an object with keys `"Doctorate"`, `"Masters"`, `"Bachelors"`, '
        '`"HS-grad"`, '
        "each the fraction (0 to 1) of records at that education level earning `>50K`, among records with a "
        "known occupation."
    ),
    traps=[
        Trap(
            id="leading_space_unknowns",
            kind="placeholder",
            caught_by="answer",
            description="Unknowns are ' ?' after the separator, so comparing with '?' finds none unless the "
            "spaces are stripped.",
        ),
        Trap(
            id="unknown_occupation_filter",
            kind="semantics",
            caught_by="answer",
            description="Shares are over records with a known occupation only.",
        ),
    ],
    # Measured: dropping the 24 duplicate records moves a share by <= 0.0003 (an acceptable choice); counting
    # unknown occupations moves shares by up to 0.0052 (a mistake).
    analytical=AnalyticalCheck(numeric_tolerance=0.002),
)

TASKS = [INCOME, EDUCATION]
LEVELS = ["Doctorate", "Masters", "Bachelors", "HS-grad"]


def _raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True, comment="|", dtype=str)
    return df.dropna(subset=["income"]).reset_index(drop=True)


def _export(df: pd.DataFrame) -> str:
    """The original UCI layout: ', ' separators, no header, no quoting."""
    return "".join(", ".join(str(v) for v in row) + "\n" for row in df.itertuples(index=False))


def build(raw: Path, out: Path, tasks: list[TaskSpec]) -> None:
    train = _raw(raw / "adult" / "adult.data")
    test = _raw(raw / "adult" / "adult.test")
    train.insert(0, "person_id", range(1, len(train) + 1))
    test.insert(0, "person_id", range(100_001, 100_001 + len(test)))
    test["income"] = test["income"].str.rstrip(".")  # the test file writes '>50K.'
    by_id = {t.id: t for t in tasks}

    task = by_id[INCOME.id]
    workspace, hidden = task_dir(out, task, README)
    (workspace / "data" / "census.data").write_text(_export(train))
    features = test.drop(columns="income")
    write_holdout(hidden, features, "person_id", (test["income"] == ">50K").astype(int), _export(features))

    task = by_id[EDUCATION.id]
    workspace, hidden = task_dir(out, task, README)
    (workspace / "data" / "census.data").write_text(_export(train))
    known = train[train["occupation"] != "?"]
    share = (known["income"] == ">50K").groupby(known["education"]).mean()
    write_answer(
        hidden,
        {
            "unknown_occupation_records": int((train["occupation"] == "?").sum()),
            "high_income_share": {level: float(share[level]) for level in LEVELS},
        },
    )
