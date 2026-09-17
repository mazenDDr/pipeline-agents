"""Bank Marketing (UCI 222, CC BY 4.0), test family: phone campaigns of a Portuguese bank, 2008-2010."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_agents.bench.build import task_dir, write_answer, write_holdout
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

README = """# bank_marketing.csv

Direct marketing phone campaigns of a Portuguese banking institution (May 2008 to November 2010), from the UCI
Machine Learning Repository (Bank Marketing, Moro, Cortez and Rita, 2014; CC BY 4.0). Semicolon-separated.

Client data:
- client_id: record identifier (added for this task)
- age (numeric)
- job: type of job ("admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
  "self-employed", "services", "student", "technician", "unemployed", "unknown")
- marital: marital status ("divorced", "married", "single", "unknown"; "divorced" means divorced or widowed)
- education ("basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course",
  "university.degree", "unknown")
- default: has credit in default? ("no", "yes", "unknown")
- housing: has housing loan? ("no", "yes", "unknown")
- loan: has personal loan? ("no", "yes", "unknown")

Last contact of the current campaign:
- contact: contact communication type ("cellular", "telephone")
- month: last contact month of year
- day_of_week: last contact day of the week
- duration: last contact duration, in seconds. Important note: this attribute highly affects the output
  target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also,
  after the end of the call y is obviously known.

Other attributes:
- campaign: number of contacts performed during this campaign and for this client (includes last contact)
- pdays: number of days that passed after the client was last contacted from a previous campaign
  (999 means the client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client
- poutcome: outcome of the previous marketing campaign ("failure", "nonexistent", "success")

Social and economic context:
- emp.var.rate: employment variation rate, quarterly indicator
- cons.price.idx: consumer price index, monthly indicator
- cons.conf.idx: consumer confidence index, monthly indicator
- euribor3m: euribor 3 month rate, daily indicator
- nr.employed: number of employees, quarterly indicator

Target:
- y: has the client subscribed a term deposit? ("yes", "no")
"""

SUBSCRIBE = TaskSpec(
    id="bank-1-subscribe",
    family="bank",
    split="test",
    role="warmup",
    kind="predictive",
    difficulty="medium",
    goal=(
        "The bank wants to decide whom to call in its next campaign. `data/bank_marketing.csv` holds past "
        "calls (see `data/README.md`). Build a model that predicts, before calling a client, the probability "
        "that they subscribe (`y` is `yes`); it will be judged by ROC AUC on other clients."
    ),
    deliverable="The id column is `client_id`; `prediction` is the probability that `y` is `yes`.",
    traps=[
        Trap(
            id="duration_known_after_call",
            kind="target_leak",
            caught_by="predict",
            description="duration is only known after the call (README), so the clients to score before "
            "calling have no duration column.",
        ),
        Trap(
            id="pdays_999",
            kind="sentinel",
            caught_by="score only",
            description="pdays = 999 means never contacted (96% of rows), not a very long time ago.",
        ),
        Trap(
            id="unknown_categories",
            kind="placeholder",
            caught_by="score only",
            description="'unknown' appears in job, marital, education, default, housing and loan.",
        ),
    ],
    # The rows are in time order and the subscription rate climbs from 3% to 46% across the file as interest
    # rates fall. A time-based holdout would mostly measure that shift, so the holdout is a random 20% of
    # clients; this task tests the duration leak, not forecasting.
    # Measured on the holdout: euribor3m alone 0.758 (the obvious baseline), numeric logistic regression
    # 0.776,
    # one-hot logistic regression 0.807, gradient boosting 0.811; shuffled 5-fold CV within 0.02 of holdout.
    predictive=PredictiveCheck(
        id_column="client_id", metric="roc_auc", threshold=0.79, honesty_tolerance=0.04
    ),
)

PREVIOUS = TaskSpec(
    id="bank-2-previous-campaign",
    family="bank",
    split="test",
    role="followup",
    kind="analytical",
    difficulty="easy",
    goal=(
        "Using `data/bank_marketing.csv` (see `data/README.md`), describe what earlier campaigns tell "
        "us: how "
        "long ago clients were last contacted, where that is recorded, and how the outcome of the previous "
        "campaign relates to subscribing now."
    ),
    deliverable=(
        "`output/answer.json` must be an object with three keys:\n"
        '- `"clients_with_days_since_previous_contact"`: the number of clients for whom the number of '
        "days since "
        "a previous-campaign contact is recorded;\n"
        '- `"mean_days_since_previous_contact"`: the mean of that number over those clients;\n'
        '- `"subscription_rate_by_previous_outcome"`: an object with keys `"failure"`, `"nonexistent"`, '
        '`"success"` (the values of poutcome), each the fraction (0 to 1) of those clients with `y` = `yes`.'
    ),
    traps=[
        Trap(
            id="pdays_999",
            kind="sentinel",
            caught_by="answer",
            description="pdays = 999 means no previous contact (README); averaging it in gives a mean near "
            "960 instead of about 6.",
        ),
    ],
    # Data finding: 3,311 clients have previous > 0 and poutcome "failure" yet pdays = 999, contradicting the
    # README. "Never contacted" is therefore ambiguous (31,731 clients by pdays, 28,420 by previous), so the
    # task only asks what is unambiguous: recorded day counts, and rates by poutcome.
    analytical=AnalyticalCheck(numeric_tolerance=1e-3),
)

TASKS = [SUBSCRIBE, PREVIOUS]


def _export(df: pd.DataFrame, path: Path) -> None:
    """The original layout: semicolon-separated, text quoted, numbers bare."""
    df.to_csv(path, sep=";", index=False, quoting=csv.QUOTE_NONNUMERIC)


def build(raw: Path, out: Path, tasks: list[TaskSpec]) -> None:
    df = pd.read_csv(
        raw / "bank_marketing" / "bank-additional" / "bank-additional" / "bank-additional-full.csv", sep=";"
    )
    df.insert(0, "client_id", np.arange(1, len(df) + 1))
    rng = np.random.default_rng(0)
    holdout = np.zeros(len(df), dtype=bool)
    for label in ("yes", "no"):  # stratified 20%
        idx = np.flatnonzero(df["y"].to_numpy() == label)
        holdout[rng.choice(idx, size=round(0.2 * len(idx)), replace=False)] = True
    train, future = df[~holdout], df[holdout]
    by_id = {t.id: t for t in tasks}

    task = by_id[SUBSCRIBE.id]
    workspace, hidden = task_dir(out, task, README)
    _export(train, workspace / "data" / "bank_marketing.csv")
    features = future.drop(columns=["y", "duration"])
    _export(features, hidden / "holdout_features.csv")
    write_holdout(
        hidden,
        features,
        "client_id",
        (future["y"] == "yes").astype(int),
        (hidden / "holdout_features.csv").read_text(),
    )

    task = by_id[PREVIOUS.id]
    workspace, hidden = task_dir(out, task, README)
    _export(train, workspace / "data" / "bank_marketing.csv")
    recorded = train["pdays"] != 999
    rates = (train["y"] == "yes").groupby(train["poutcome"]).mean()
    write_answer(
        hidden,
        {
            "clients_with_days_since_previous_contact": int(recorded.sum()),
            "mean_days_since_previous_contact": float(train.loc[recorded, "pdays"].mean()),
            "subscription_rate_by_previous_outcome": {
                k: float(rates[k]) for k in ("failure", "nonexistent", "success")
            },
        },
    )
