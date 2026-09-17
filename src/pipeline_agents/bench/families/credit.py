"""Default of Credit Card Clients (UCI 350, CC BY 4.0), test family: Taiwan, April-September 2005."""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_agents.bench.build import task_dir, write_answer, write_holdout
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

TARGET = "default payment next month"

README = """# credit_card_clients.csv

Credit card clients of a bank in Taiwan, April to September 2005, from the UCI Machine Learning Repository
(Default of Credit Card Clients, Yeh and Lien, 2009; CC BY 4.0). Exported from the original spreadsheet, which
has two header rows: generic variable codes (X1 ... X23, Y) above the column names.

- ID: client identifier
- LIMIT_BAL (X1): amount of given credit in NT dollars, including individual and family credit
- SEX (X2): 1 = male, 2 = female
- EDUCATION (X3): 1 = graduate school, 2 = university, 3 = high school, 4 = others
- MARRIAGE (X4): 1 = married, 2 = single, 3 = others
- AGE (X5): age in years
- PAY_0, PAY_2 ... PAY_6 (X6-X11): history of past payment, from September (PAY_0) back to April (PAY_6) 2005:
  -1 = pay duly, 1 = payment delay for one month, 2 = payment delay for two months, ..., 8 = payment delay
  for eight months, 9 = payment delay for nine months and above
- BILL_AMT1 ... BILL_AMT6 (X12-X17): amount of bill statement in NT dollars, September back to April 2005
- PAY_AMT1 ... PAY_AMT6 (X18-X23): amount of previous payment in NT dollars, September back to April 2005
- default payment next month (Y): 1 = yes, 0 = no
"""

DEFAULT = TaskSpec(
    id="credit-1-default",
    family="credit",
    split="test",
    role="warmup",
    kind="predictive",
    difficulty="medium",
    goal=(
        "The bank wants to estimate each client's risk of defaulting on next month's payment. "
        "`data/credit_card_clients.csv` holds labelled clients (see `data/README.md`). Build a model that "
        "predicts the probability of `default payment next month`; it will be judged by ROC AUC on other "
        "clients."
    ),
    deliverable="The id column is `ID`; `prediction` is the probability of default.",
    traps=[
        Trap(
            id="two_header_rows",
            kind="format",
            caught_by="clean_rerun",
            description="The first line holds codes X1 ... Y and the second the names (README); reading only "
            "the first as a header turns every column into text.",
        ),
        Trap(
            id="undocumented_codes",
            kind="semantics",
            caught_by="score only",
            description="EDUCATION uses 0, 5 and 6, MARRIAGE uses 0, and PAY_* uses -2 and 0; none is in the "
            "README.",
        ),
        Trap(
            id="codes_not_quantities",
            kind="int_coded",
            caught_by="score only",
            description="SEX, EDUCATION and MARRIAGE are codes, not quantities.",
        ),
    ],
    # Measured on the holdout: PAY_0 alone 0.692, logistic regression on the raw columns 0.727; one-hot
    # logistic regression 0.776, random forest 0.780, gradient boosting 0.785. Shuffled 5-fold CV within 0.01.
    predictive=PredictiveCheck(id_column="ID", metric="roc_auc", threshold=0.755, honesty_tolerance=0.03),
)

EDUCATION = TaskSpec(
    id="credit-2-education",
    family="credit",
    split="test",
    role="followup",
    kind="analytical",
    difficulty="easy",
    goal=(
        "Using `data/credit_card_clients.csv` (see `data/README.md`), report the default rate for each "
        "education "
        "level the README documents, and how many clients have an education code the README does not "
        "document."
    ),
    deliverable=(
        "`output/answer.json` must be an object with two keys:\n"
        '- `"undocumented_education_clients"`: the number of clients whose EDUCATION code is not '
        "documented;\n"
        '- `"default_rate_by_education"`: an object with keys `"graduate school"`, `"university"`, '
        '`"high school"`, `"others"`, each the fraction (0 to 1) of clients with that documented code who '
        "defaulted."
    ),
    traps=[
        Trap(
            id="two_header_rows",
            kind="format",
            caught_by="clean_rerun",
            description="Two header rows; the names are on the second.",
        ),
        Trap(
            id="undocumented_codes_are_not_others",
            kind="semantics",
            caught_by="answer",
            description="Codes 0, 5 and 6 are not documented; folding them into 'others' (4) moves its "
            "default rate from 0.041 to 0.063 in this data.",
        ),
    ],
    analytical=AnalyticalCheck(numeric_tolerance=1e-3),
)

TASKS = [DEFAULT, EDUCATION]


def _export(df: pd.DataFrame, codes: list[str], path: Path) -> None:
    """The spreadsheet's two header rows (codes, then names) above the data."""
    with path.open("w") as f:
        f.write(",".join(codes) + "\n")
        df.to_csv(f, index=False)


def build(raw: Path, out: Path, tasks: list[TaskSpec]) -> None:
    sheet = raw / "credit_default" / "default of credit card clients.xls"
    codes = ["" if pd.isna(c) else str(c) for c in pd.read_excel(sheet, header=None, nrows=1).iloc[0]]
    df = pd.read_excel(sheet, header=1)
    rng = np.random.default_rng(0)
    holdout = np.zeros(len(df), dtype=bool)
    for label in (0, 1):  # stratified 20%
        idx = np.flatnonzero(df[TARGET].to_numpy() == label)
        holdout[rng.choice(idx, size=round(0.2 * len(idx)), replace=False)] = True
    train, future = df[~holdout], df[holdout]
    by_id = {t.id: t for t in tasks}

    task = by_id[DEFAULT.id]
    workspace, hidden = task_dir(out, task, README)
    _export(train, codes, workspace / "data" / "credit_card_clients.csv")
    features = future.drop(columns=TARGET)
    write_holdout(hidden, features, "ID", future[TARGET])
    _export(features, codes[:-1], hidden / "holdout_features.csv")

    task = by_id[EDUCATION.id]
    workspace, hidden = task_dir(out, task, README)
    _export(train, codes, workspace / "data" / "credit_card_clients.csv")
    labels = {1: "graduate school", 2: "university", 3: "high school", 4: "others"}
    documented = train[train["EDUCATION"].isin(labels)]
    rates = documented.groupby("EDUCATION")[TARGET].mean()
    write_answer(
        hidden,
        {
            "undocumented_education_clients": int((~train["EDUCATION"].isin(labels)).sum()),
            "default_rate_by_education": {name: float(rates[code]) for code, name in labels.items()},
        },
    )
