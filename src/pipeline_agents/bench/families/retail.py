"""Online Retail (UCI 352, CC BY 4.0), test family: a UK online gift retailer's transactions, 2010-2011."""

from pathlib import Path

import pandas as pd

from pipeline_agents.bench.build import task_dir, write_answer, write_holdout
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

DATA_END = pd.Timestamp("2011-09-01")  # both tasks see transactions before this date only
LABEL_START = pd.Timestamp("2011-06-01")  # training labels: orders from here to DATA_END
HOLDOUT_END = pd.Timestamp("2011-12-01")  # holdout labels: orders from DATA_END to here

README = """# transactions.csv

Every transaction of a UK-based online retailer of all-occasion gifts, many of whose customers are
wholesalers, from 1 December 2010 up to 31 August 2011, from the UCI Machine Learning Repository (Online
  Retail,
Chen, Sain and Guo, 2012; CC BY 4.0). One row per invoice line.

- InvoiceNo: invoice number, a 6-digit number uniquely assigned to each transaction. If this code starts with
  the letter 'C', it indicates a cancellation.
- StockCode: product code, a 5-digit number uniquely assigned to each distinct product
- Description: product name
- Quantity: the quantity of each product per transaction
- InvoiceDate: the day and time when the transaction was generated, as exported (M/D/YYYY H:MM)
- UnitPrice: product price per unit in sterling
- CustomerID: customer number, a 5-digit number uniquely assigned to each customer; empty when unknown
- Country: the country where the customer resides
"""

LABELS_README = """
# customers_train.csv

- customer_id: a customer with at least one non-cancelled invoice before 1 June 2011
- bought_again: 1 if that customer placed a non-cancelled invoice between 1 June 2011 and 31 August 2011
"""

REVENUE = TaskSpec(
    id="retail-1-monthly-revenue",
    family="retail",
    split="test",
    role="warmup",
    kind="analytical",
    difficulty="medium",
    goal=(
        "Using `data/transactions.csv` (see `data/README.md`), report the retailer's net revenue for "
        "each month "
        "and how many invoices were cancelled."
    ),
    deliverable=(
        "`output/answer.json` must be an object with two keys:\n"
        '- `"cancelled_invoices"`: the number of distinct cancelled invoices;\n'
        '- `"net_revenue_by_month"`: an object mapping each month from `"2010-12"` to `"2011-08"` to the '
        "sum of "
        "Quantity × UnitPrice over all of that month's lines, so that cancellations and adjustments "
        "reduce it."
    ),
    traps=[
        Trap(
            id="month_first_dates",
            kind="format",
            caught_by="answer",
            description="InvoiceDate is M/D/YYYY (README); day-first parsing moves the first 12 days of "
            "every "
            "month into other months.",
        ),
        Trap(
            id="cancellations_are_negative",
            kind="semantics",
            caught_by="answer",
            description="Cancellation lines (InvoiceNo starting with C) have negative quantities; dropping "
            "negative lines as 'bad data' overstates revenue.",
        ),
        Trap(
            id="invoices_not_lines",
            kind="semantics",
            caught_by="answer",
            description="A cancelled invoice has several lines; the count is of distinct invoices.",
        ),
    ],
    analytical=AnalyticalCheck(numeric_tolerance=1e-3),
)

REPEAT = TaskSpec(
    id="retail-2-repeat-buyers",
    family="retail",
    split="test",
    role="followup",
    kind="predictive",
    difficulty="hard",
    goal=(
        "The retailer wants to know which customers will order again in the next three months. "
        "`data/transactions.csv` holds every transaction up to 31 August 2011 and `data/customers_train.csv` "
        "shows, for customers seen before 1 June 2011, who ordered again between June and August (see "
        "`data/README.md`). Build a model that predicts, for customers seen before 1 September 2011, the "
        "probability that they place a non-cancelled order between 1 September and 30 November 2011; it "
        "will be "
        "judged by ROC AUC."
    ),
    deliverable=(
        "The id column is `customer_id`; `prediction` is the probability of a new order in September to "
        "November "
        "2011. The features file is a list of `customer_id`s; `predict.py` may read `data/transactions.csv`."
    ),
    traps=[
        Trap(
            id="features_from_the_label_window",
            kind="temporal",
            caught_by="honest_estimate",
            description="Training labels come from June-August, so training features must use transactions "
            "before 1 June only; features built from all transactions contain the answer.",
        ),
        Trap(
            id="cancellations_and_unknown_customers",
            kind="semantics",
            caught_by="score only",
            description="Cancelled invoices are not orders, and lines without a CustomerID belong to nobody.",
        ),
    ],
    # Measured on the holdout: recency alone 0.666; gradient boosting 0.718 (CV 0.711), logistic regression
    # 0.739
    # (CV 0.752). Features built from the label window: CV 1.000, holdout 0.640, caught by the honesty check.
    predictive=PredictiveCheck(
        id_column="customer_id", metric="roc_auc", threshold=0.695, honesty_tolerance=0.05
    ),
)

TASKS = [REVENUE, REPEAT]


def _orders(df: pd.DataFrame) -> pd.DataFrame:
    """Non-cancelled invoice lines with a known customer."""
    return df[~df["InvoiceNo"].str.startswith("C") & df["CustomerID"].notna()]


def _export(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    when = out["InvoiceDate"]
    out["InvoiceDate"] = (
        when.dt.month.astype(str)
        + "/"
        + when.dt.day.astype(str)
        + "/"
        + when.dt.year.astype(str)
        + " "
        + when.dt.hour.astype(str)
        + ":"
        + when.dt.strftime("%M")
    )
    out["CustomerID"] = out["CustomerID"].astype("Int64")
    out.to_csv(path, index=False)


def build(raw: Path, out: Path, tasks: list[TaskSpec]) -> None:
    cache = raw / "online_retail" / "online_retail.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = pd.read_excel(
            raw / "online_retail" / "Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str}
        )
        df.to_pickle(cache)
    seen = df[df["InvoiceDate"] < DATA_END]
    by_id = {t.id: t for t in tasks}

    task = by_id[REVENUE.id]
    workspace, hidden = task_dir(out, task, README)
    _export(seen, workspace / "data" / "transactions.csv")
    value = seen["Quantity"] * seen["UnitPrice"]
    monthly = value.groupby(seen["InvoiceDate"].dt.strftime("%Y-%m")).sum()
    write_answer(
        hidden,
        {
            "cancelled_invoices": int(seen.loc[seen["InvoiceNo"].str.startswith("C"), "InvoiceNo"].nunique()),
            "net_revenue_by_month": {month: float(v) for month, v in monthly.items()},
        },
    )

    task = by_id[REPEAT.id]
    workspace, hidden = task_dir(out, task, README + LABELS_README)
    _export(seen, workspace / "data" / "transactions.csv")
    orders = _orders(df)

    def customers_between(start: pd.Timestamp | None, end: pd.Timestamp) -> set[int]:
        window = orders[(orders["InvoiceDate"] < end) & ((orders["InvoiceDate"] >= start) if start else True)]
        return set(window["CustomerID"].astype(int))

    train_customers = sorted(customers_between(None, LABEL_START))
    bought = customers_between(LABEL_START, DATA_END)
    pd.DataFrame(
        {"customer_id": train_customers, "bought_again": [int(c in bought) for c in train_customers]}
    ).to_csv(workspace / "data" / "customers_train.csv", index=False)
    future_customers = sorted(customers_between(None, DATA_END))
    later = customers_between(DATA_END, HOLDOUT_END)
    features = pd.DataFrame({"customer_id": future_customers})
    write_holdout(hidden, features, "customer_id", pd.Series([int(c in later) for c in future_customers]))
