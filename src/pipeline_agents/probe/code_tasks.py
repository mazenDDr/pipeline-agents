"""Small pandas/sklearn coding tasks for choosing a model (T1).

Each task writes its input files into a work dir, keeps any answers in a separate hidden dir the
generated script cannot see, and checks the files the script leaves behind. Several tasks contain
a trap a careful data scientist avoids: sentinel values, int-coded categories, a leaking column,
an unseen category at prediction time, zero-padded keys.

These are deliberately unlike the T2 benchmark tasks: single-step, synthetic, one file out.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class CodeTask:
    id: str
    prompt: str
    build: Callable[[Path, Path, int], None]  # (workdir, hidden_dir, seed)
    check: Callable[[Path, Path], CheckResult]
    reference: str  # a correct script; tests prove check() accepts it
    broken: str  # a plausible wrong script; tests prove check() rejects it


def _read(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _missing(name: str) -> CheckResult:
    return CheckResult(False, f"{name} not written")


# --- 1. median imputation on a skewed column -------------------------------------------------


def _build_impute(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 400
    income = rng.lognormal(10, 1, n).round(2)
    income[rng.random(n) < 0.15] = np.nan
    city = rng.choice(["Cairo", "Giza", "Alexandria"], n)
    pd.DataFrame({"id": range(n), "income": income, "city": city}).to_csv(work / "data.csv", index=False)
    pd.read_csv(work / "data.csv").to_csv(hidden / "original.csv", index=False)


def _check_impute(work: Path, hidden: Path) -> CheckResult:
    out, orig = _read(work / "clean.csv"), pd.read_csv(hidden / "original.csv")
    if out is None:
        return _missing("clean.csv")
    if len(out) != len(orig) or list(out.columns) != list(orig.columns):
        return CheckResult(False, f"shape/columns changed: {out.shape} {list(out.columns)}")
    out = out.sort_values("id").reset_index(drop=True)
    median = orig["income"].median()
    was_missing = orig["income"].isna()
    if out["income"].isna().any():
        return CheckResult(False, "nulls remain")
    if not np.allclose(out.loc[was_missing, "income"], median):
        return CheckResult(False, f"filled values != median {median}")
    if not np.allclose(out.loc[~was_missing, "income"], orig.loc[~was_missing, "income"]):
        return CheckResult(False, "observed values changed")
    return CheckResult(True, "ok")


IMPUTE = CodeTask(
    id="impute_skewed",
    prompt=(
        "data.csv has columns id, income, city. income is right-skewed and has missing values. "
        "Fill the missing income values with the median of the observed incomes. Leave everything else "
        "unchanged and write the result, with the same columns and rows, to clean.csv (no index column)."
    ),
    build=_build_impute,
    check=_check_impute,
    reference=(
        "import pandas as pd\ndf = pd.read_csv('data.csv')\n"
        "df['income'] = df['income'].fillna(df['income'].median())\ndf.to_csv('clean.csv', index=False)\n"
    ),
    broken=(
        "import pandas as pd\ndf = pd.read_csv('data.csv')\n"
        "df['income'] = df['income'].fillna(df['income'].mean())\ndf.to_csv('clean.csv', index=False)\n"
    ),
)


# --- 2. sentinel values documented in a data dictionary --------------------------------------


def _build_sentinel(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    age = rng.integers(18, 80, n).astype(float)
    age[rng.random(n) < 0.1] = -999
    age[rng.random(n) < 0.05] = np.nan
    pd.DataFrame({"customer_id": range(n), "age": age, "score": rng.random(n).round(3)}).to_csv(
        work / "customers.csv", index=False
    )
    (work / "README.txt").write_text(
        "customers.csv\n- customer_id: unique id\n- age: age in years; -999 means not recorded\n"
        "- score: satisfaction score 0-1\n"
    )
    known = age[(age != -999) & ~np.isnan(age)]
    (hidden / "answer.json").write_text(json.dumps({"mean_age": float(known.mean())}))


def _check_sentinel(work: Path, hidden: Path) -> CheckResult:
    path = work / "result.json"
    if not path.exists():
        return _missing("result.json")
    try:
        got = float(json.loads(path.read_text())["mean_age"])
    except (ValueError, KeyError, TypeError) as e:
        return CheckResult(False, f"bad result.json: {e}")
    want = json.loads((hidden / "answer.json").read_text())["mean_age"]
    return CheckResult(abs(got - want) < 1e-6, f"got {got:.4f}, want {want:.4f}")


SENTINEL = CodeTask(
    id="sentinel_missing",
    prompt=(
        "customers.csv is described in README.txt. Compute the mean age of the customers whose age is "
        'known and write it to result.json as {"mean_age": <number>}.'
    ),
    build=_build_sentinel,
    check=_check_sentinel,
    reference=(
        "import json, pandas as pd\ndf = pd.read_csv('customers.csv')\n"
        "age = df['age'].replace(-999, float('nan'))\n"
        "json.dump({'mean_age': float(age.mean())}, open('result.json', 'w'))\n"
    ),
    broken=(
        "import json, pandas as pd\ndf = pd.read_csv('customers.csv')\n"
        "json.dump({'mean_age': float(df['age'].mean())}, open('result.json', 'w'))\n"
    ),
)


# --- 3. int-coded categorical ----------------------------------------------------------------


def _build_onehot(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 200
    df = pd.DataFrame({"store": range(n), "region_code": rng.integers(1, 6, n), "sales": rng.random(n)})
    df.to_csv(work / "stores.csv", index=False)
    df.to_csv(hidden / "original.csv", index=False)


def _check_onehot(work: Path, hidden: Path) -> CheckResult:
    out, orig = _read(work / "encoded.csv"), pd.read_csv(hidden / "original.csv")
    if out is None:
        return _missing("encoded.csv")
    want_cols = {"store", "sales"} | {f"region_code_{k}" for k in range(1, 6)}
    if set(out.columns) != want_cols:
        return CheckResult(False, f"columns {sorted(out.columns)}")
    out = out.sort_values("store").reset_index(drop=True)
    for k in range(1, 6):
        if not (out[f"region_code_{k}"].astype(int) == (orig["region_code"] == k).astype(int)).all():
            return CheckResult(False, f"region_code_{k} wrong")
    return CheckResult(True, "ok")


ONEHOT = CodeTask(
    id="int_coded_categorical",
    prompt=(
        "stores.csv has columns store, region_code, sales. region_code is a region identifier from 1 to 5 "
        "(a category, not a quantity). One-hot encode it into integer 0/1 columns named region_code_1 ... "
        "region_code_5, drop the original region_code column, keep store and sales, and write encoded.csv."
    ),
    build=_build_onehot,
    check=_check_onehot,
    reference=(
        "import pandas as pd\ndf = pd.read_csv('stores.csv')\n"
        "df = pd.get_dummies(df, columns=['region_code'], dtype=int)\ndf.to_csv('encoded.csv', index=False)\n"
    ),
    broken=(
        "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\n"
        "df = pd.read_csv('stores.csv')\n"
        "df['region_code'] = StandardScaler().fit_transform(df[['region_code']])\n"
        "df.to_csv('encoded.csv', index=False)\n"
    ),
)


# --- 4. duplicates, exact and after normalization --------------------------------------------


def _build_dedupe(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = [{"email": f"user{i}@mail.com", "plan": rng.choice(["a", "b"])} for i in range(60)]
    extra = []
    for i in rng.choice(60, 10, replace=False):
        extra.append(dict(rows[i]))  # exact copy
    for i in rng.choice(60, 10, replace=False):
        extra.append({"email": "  " + rows[i]["email"].upper() + " ", "plan": rows[i]["plan"]})
    all_rows = rows + extra
    order = rng.permutation(len(all_rows))
    df = pd.DataFrame([all_rows[i] for i in order])
    df.insert(0, "row_id", range(len(df)))
    df.to_csv(work / "signups.csv", index=False)
    key = df["email"].str.strip().str.lower()
    keep = df.loc[~key.duplicated(keep="first"), "row_id"]
    keep.to_csv(hidden / "keep.csv", index=False)


def _check_dedupe(work: Path, hidden: Path) -> CheckResult:
    out = _read(work / "dedup.csv")
    if out is None:
        return _missing("dedup.csv")
    if "row_id" not in out.columns:
        return CheckResult(False, "row_id column missing")
    want = set(pd.read_csv(hidden / "keep.csv")["row_id"])
    got = set(out["row_id"])
    return CheckResult(got == want, f"kept {len(got)} rows, want {len(want)}")


DEDUPE = CodeTask(
    id="dedupe_normalized",
    prompt=(
        "signups.csv has columns row_id, email, plan. Some sign-ups are duplicates: the same email address, "
        "possibly with different letter case or surrounding spaces. Keep only the first occurrence (lowest "
        "row_id) of each email address and write the kept rows, all columns, to dedup.csv."
    ),
    build=_build_dedupe,
    check=_check_dedupe,
    reference=(
        "import pandas as pd\ndf = pd.read_csv('signups.csv').sort_values('row_id')\n"
        "key = df['email'].str.strip().str.lower()\n"
        "df[~key.duplicated()].to_csv('dedup.csv', index=False)\n"
    ),
    broken=(
        "import pandas as pd\ndf = pd.read_csv('signups.csv')\n"
        "df.drop_duplicates(subset=['email']).to_csv('dedup.csv', index=False)\n"
    ),
)


# --- 5. mixed date formats -------------------------------------------------------------------


def _build_dates(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    days = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 180, 250), unit="D")
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y"]
    text = [d.strftime(fmts[i % 3]) for i, d in enumerate(days)]
    pd.DataFrame({"order_id": range(len(days)), "order_date": text}).to_csv(work / "orders.csv", index=False)
    (work / "README.txt").write_text(
        "orders.csv: order_date was typed by hand in three formats: 2024-03-05, 05/03/2024 (day first), "
        "and March 05, 2024.\n"
    )
    counts = pd.Series(days.strftime("%Y-%m")).value_counts().sort_index()
    (hidden / "answer.json").write_text(json.dumps({k: int(v) for k, v in counts.items()}))


def _check_dates(work: Path, hidden: Path) -> CheckResult:
    path = work / "result.json"
    if not path.exists():
        return _missing("result.json")
    try:
        got = {str(k): int(v) for k, v in json.loads(path.read_text()).items()}
    except (ValueError, AttributeError, TypeError) as e:
        return CheckResult(False, f"bad result.json: {e}")
    want = json.loads((hidden / "answer.json").read_text())
    return CheckResult(got == want, f"got {got}" if got != want else "ok")


DATES = CodeTask(
    id="mixed_dates",
    prompt=(
        "orders.csv and README.txt describe hand-typed order dates. Count the orders per calendar month and "
        'write result.json as an object mapping "YYYY-MM" to the count, e.g. {"2024-01": 12, ...}.'
    ),
    build=_build_dates,
    check=_check_dates,
    reference=(
        "import json, pandas as pd\ndf = pd.read_csv('orders.csv')\n"
        "def parse(s):\n"
        "    for f in ('%Y-%m-%d', '%d/%m/%Y', '%B %d, %Y'):\n"
        "        try:\n            return pd.to_datetime(s, format=f)\n"
        "        except ValueError:\n            pass\n"
        "    raise ValueError(s)\n"
        "m = df['order_date'].map(parse).dt.strftime('%Y-%m').value_counts()\n"
        "json.dump({k: int(v) for k, v in m.items()}, open('result.json', 'w'))\n"
    ),
    broken=(
        "import json, pandas as pd\ndf = pd.read_csv('orders.csv')\n"
        "d = pd.to_datetime(df['order_date'], format='mixed')\n"
        "m = d.dt.strftime('%Y-%m').value_counts()\n"
        "json.dump({k: int(v) for k, v in m.items()}, open('result.json', 'w'))\n"
    ),
)


# --- 6. a leaking column ---------------------------------------------------------------------


def _churn_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(60, 20, n).clip(10)
    tickets = rng.poisson(1.5, n)
    # Strong enough that every honest model family clears the threshold. The first version let an honest
    # random forest score 0.67-0.73 against a 0.70 threshold, so the task measured model choice, not leakage.
    logit = 0.5 - 0.10 * tenure + 0.05 * (monthly - 60) + 1.0 * tickets
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_fee": monthly.round(2),
            "support_tickets": tickets,
            "churned": churned,
        }
    )


def _build_leak(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    train, test = _churn_frame(rng, 1500), _churn_frame(rng, 500)
    reasons = np.array(["price", "service", "moved"])
    train["cancel_reason"] = np.where(train["churned"] == 1, rng.choice(reasons, len(train)), "")
    train.insert(0, "customer_id", range(len(train)))
    test.insert(0, "customer_id", range(10_000, 10_000 + len(test)))
    test[["customer_id", "churned"]].to_csv(hidden / "labels.csv", index=False)
    test = test.drop(columns="churned")
    # At prediction time the column is noise: about half the customers still carry a reason from an earlier,
    # closed account. Without this, a model that uses the leak sees a constant column and still ranks well
    # (T1 re-check: two leaky scripts passed at AUC 0.86 and 0.91).
    test["cancel_reason"] = np.where(rng.random(len(test)) < 0.5, rng.choice(reasons, len(test)), "")
    train.to_csv(work / "train.csv", index=False)
    test.to_csv(work / "test.csv", index=False)
    (work / "README.txt").write_text(
        "train.csv / test.csv: telecom customers.\n- tenure_months, monthly_fee, support_tickets\n"
        "- cancel_reason: filled in by the support team when a customer cancels\n"
        "- churned: 1 if the customer left (train only)\n"
    )


def _check_leak(work: Path, hidden: Path) -> CheckResult:
    out = _read(work / "predictions.csv")
    if out is None:
        return _missing("predictions.csv")
    if not {"customer_id", "churn_probability"} <= set(out.columns):
        return CheckResult(False, f"columns {list(out.columns)}")
    merged = pd.read_csv(hidden / "labels.csv").merge(out, on="customer_id", how="left")
    if merged["churn_probability"].isna().any():
        return CheckResult(False, "missing predictions for some customers")
    auc = roc_auc_score(merged["churned"], merged["churn_probability"])
    # Measured over 10 seeds: honest LR / RF 0.86-0.93; models that use cancel_reason 0.48-0.76.
    return CheckResult(auc >= 0.80, f"AUC {auc:.3f} (need >= 0.80)")


LEAK = CodeTask(
    id="leaky_column",
    prompt=(
        "Build a model that predicts churn. train.csv and test.csv are described in README.txt. Write "
        "predictions.csv with columns customer_id, churn_probability for every customer in test.csv."
    ),
    build=_build_leak,
    check=_check_leak,
    reference=(
        "import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\n"
        "X = ['tenure_months', 'monthly_fee', 'support_tickets']\n"
        "m = LogisticRegression(max_iter=1000).fit(tr[X], tr['churned'])\n"
        "pd.DataFrame({'customer_id': te['customer_id'], 'churn_probability': m.predict_proba(te[X])[:, 1]})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
    broken=(
        "import pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\n"
        "f = lambda d: pd.DataFrame({'t': d['tenure_months'], 'r': d['cancel_reason'].notna().astype(int)})\n"
        "m = DecisionTreeClassifier(max_depth=2).fit(f(tr), tr['churned'])\n"
        "pd.DataFrame({'customer_id': te['customer_id'], 'churn_probability': m.predict_proba(f(te))[:, 1]})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
)


# --- 7. regression on a skewed target --------------------------------------------------------


def _house_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    area = rng.lognormal(4.7, 0.8, n).round().clip(10)
    rooms = np.clip((area / 30 + rng.normal(0, 1, n)).round(), 1, 10)
    age = rng.integers(0, 60, n)
    log_price = 9 + 1.5 * np.log(area) + 0.05 * rooms - 0.006 * age + rng.normal(0, 0.25, n)
    return pd.DataFrame(
        {"area_m2": area, "rooms": rooms, "age_years": age, "price": np.exp(log_price).round()}
    )


def _build_regression(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    train, test = _house_frame(rng, 1200), _house_frame(rng, 400)
    test.insert(0, "house_id", range(len(test)))
    train.insert(0, "house_id", range(5000, 5000 + len(train)))
    test[["house_id", "price"]].to_csv(hidden / "labels.csv", index=False)
    train.to_csv(work / "train.csv", index=False)
    test.drop(columns="price").to_csv(work / "test.csv", index=False)


def _check_regression(work: Path, hidden: Path) -> CheckResult:
    out = _read(work / "predictions.csv")
    if out is None:
        return _missing("predictions.csv")
    if not {"house_id", "price"} <= set(out.columns):
        return CheckResult(False, f"columns {list(out.columns)}")
    merged = pd.read_csv(hidden / "labels.csv").merge(out, on="house_id", how="left", suffixes=("", "_pred"))
    pred = merged["price_pred"]
    if pred.isna().any() or (pred <= 0).any():
        return CheckResult(False, "missing or non-positive predictions")
    rmsle = float(np.sqrt(np.mean((np.log(pred) - np.log(merged["price"])) ** 2)))
    # Measured over 6 seeds: log-target linear 0.24-0.26, gradient boosting on log price 0.26-0.30,
    # linear regression on raw price 6.6-7.3 (the trap). Noise sd is 0.25.
    return CheckResult(rmsle <= 0.32, f"RMSLE {rmsle:.3f} (need <= 0.32)")


REGRESSION = CodeTask(
    id="skewed_regression",
    prompt=(
        "train.csv has house features (area_m2, rooms, age_years) and price; test.csv has the same features "
        "without price. Prices are strongly right-skewed. Predict price for every house in test.csv and "
        "write predictions.csv with columns house_id, price. You will be scored by RMSE on log(price)."
    ),
    build=_build_regression,
    check=_check_regression,
    reference=(
        "import numpy as np, pandas as pd\nfrom sklearn.linear_model import LinearRegression\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\n"
        "f = lambda d: np.column_stack([np.log(d['area_m2']), d['rooms'], d['age_years']])\n"
        "m = LinearRegression().fit(f(tr), np.log(tr['price']))\n"
        "pd.DataFrame({'house_id': te['house_id'], 'price': np.exp(m.predict(f(te)))})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
    broken=(
        "import pandas as pd\nfrom sklearn.linear_model import LinearRegression\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\nX = ['area_m2', 'rooms', 'age_years']\n"
        "m = LinearRegression().fit(tr[X], tr['price'])\n"
        "pd.DataFrame({'house_id': te['house_id'], 'price': m.predict(te[X]).clip(1)})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
)


# --- 8. filtered aggregation -----------------------------------------------------------------


def _build_topk(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 800
    cats = ["books", "games", "garden", "kitchen", "music", "sports", "toys"]
    df = pd.DataFrame(
        {
            "order_id": range(n),
            "category": rng.choice(cats, n),
            "revenue": rng.gamma(2, 30, n).round(2),
            "returned": rng.choice(["yes", "no"], n, p=[0.25, 0.75]),
        }
    )
    # Make the gross ranking differ from the net one: "toys" sells most but is often returned.
    toys = df["category"] == "toys"
    df.loc[toys, "revenue"] *= 1.6
    df.loc[toys & (rng.random(n) < 0.6), "returned"] = "yes"
    df.to_csv(work / "sales.csv", index=False)
    net = df[df["returned"] == "no"].groupby("category")["revenue"].sum().sort_values(ascending=False)
    (hidden / "answer.json").write_text(json.dumps({"top3": list(net.index[:3])}))


def _check_topk(work: Path, hidden: Path) -> CheckResult:
    path = work / "result.json"
    if not path.exists():
        return _missing("result.json")
    try:
        got = list(json.loads(path.read_text())["top3"])
    except (ValueError, KeyError, TypeError) as e:
        return CheckResult(False, f"bad result.json: {e}")
    want = json.loads((hidden / "answer.json").read_text())["top3"]
    return CheckResult(got == want, f"got {got}, want {want}")


TOPK = CodeTask(
    id="filtered_topk",
    prompt=(
        "sales.csv has order_id, category, revenue and returned (yes/no). Net revenue excludes returned "
        "orders. Find the 3 categories with the highest total net revenue, highest first, and write "
        'result.json as {"top3": ["...", "...", "..."]}.'
    ),
    build=_build_topk,
    check=_check_topk,
    reference=(
        "import json, pandas as pd\ndf = pd.read_csv('sales.csv')\n"
        "s = df[df['returned'] == 'no'].groupby('category')['revenue'].sum().sort_values(ascending=False)\n"
        "json.dump({'top3': list(s.index[:3])}, open('result.json', 'w'))\n"
    ),
    broken=(
        "import json, pandas as pd\ndf = pd.read_csv('sales.csv')\n"
        "s = df.groupby('category')['revenue'].sum().sort_values(ascending=False)\n"
        "json.dump({'top3': list(s.index[:3])}, open('result.json', 'w'))\n"
    ),
)


# --- 9. an unseen category at prediction time ------------------------------------------------


def _plan_frame(rng: np.random.Generator, n: int, plans: list[str]) -> pd.DataFrame:
    plan = rng.choice(plans, n)
    usage = rng.gamma(2, 10, n)
    # Strong enough that any sensible model passes; the first version failed an honest random forest
    # (AUC 0.59).
    base = {"basic": -1.5, "pro": 1.0, "enterprise": 2.0}
    logit = np.array([base[p] for p in plan]) + 0.15 * (usage - 20)
    upgraded = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"plan_type": plan, "weekly_hours": usage.round(1), "upgraded": upgraded})


def _build_unseen(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    train = _plan_frame(rng, 1000, ["basic", "pro"])
    test = _plan_frame(rng, 400, ["basic", "pro", "enterprise"])
    train.insert(0, "account_id", range(len(train)))
    test.insert(0, "account_id", range(20_000, 20_000 + len(test)))
    test[["account_id", "upgraded"]].to_csv(hidden / "labels.csv", index=False)
    train.to_csv(work / "train.csv", index=False)
    test.drop(columns="upgraded").to_csv(work / "test.csv", index=False)


def _check_unseen(work: Path, hidden: Path) -> CheckResult:
    out = _read(work / "predictions.csv")
    if out is None:
        return _missing("predictions.csv")
    if not {"account_id", "upgrade_probability"} <= set(out.columns):
        return CheckResult(False, f"columns {list(out.columns)}")
    merged = pd.read_csv(hidden / "labels.csv").merge(out, on="account_id", how="left")
    if merged["upgrade_probability"].isna().any():
        return CheckResult(False, "missing predictions for some accounts")
    auc = roc_auc_score(merged["upgraded"], merged["upgrade_probability"])
    # Measured over 10 seeds: LR / RF / HGB / GB with unknown categories handled score 0.75-0.89. The trap
    # is a crash or missing rows on the unseen "enterprise" plan, not the choice of model.
    return CheckResult(auc >= 0.70, f"AUC {auc:.3f} (need >= 0.70)")


UNSEEN = CodeTask(
    id="unseen_category",
    prompt=(
        "train.csv has account_id, plan_type, weekly_hours and upgraded (0/1); test.csv has the same columns "
        "without upgraded. Train a classifier and write predictions.csv with columns account_id, "
        "upgrade_probability for every account in test.csv."
    ),
    build=_build_unseen,
    check=_check_unseen,
    reference=(
        "import pandas as pd\nfrom sklearn.compose import make_column_transformer\n"
        "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\nX = ['plan_type', 'weekly_hours']\n"
        "ct = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['plan_type']), "
        "remainder='passthrough')\n"
        "m = make_pipeline(ct, LogisticRegression(max_iter=1000)).fit(tr[X], tr['upgraded'])\n"
        "pd.DataFrame({'account_id': te['account_id'], 'upgrade_probability': m.predict_proba(te[X])[:, 1]})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
    broken=(
        "import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
        "tr, te = pd.read_csv('train.csv'), pd.read_csv('test.csv')\n"
        "Xtr = pd.get_dummies(tr[['plan_type', 'weekly_hours']])\n"
        "Xte = pd.get_dummies(te[['plan_type', 'weekly_hours']])\n"
        "m = LogisticRegression(max_iter=1000).fit(Xtr, tr['upgraded'])\n"
        "pd.DataFrame({'account_id': te['account_id'], 'upgrade_probability': m.predict_proba(Xte)[:, 1]})"
        ".to_csv('predictions.csv', index=False)\n"
    ),
)


# --- 10. zero-padded join keys ---------------------------------------------------------------


def _build_merge(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(1, 5000), 80, replace=False)
    customers = pd.DataFrame({"customer_id": [f"{i:05d}" for i in ids], "name": [f"c{i}" for i in ids]})
    buyers = rng.choice(ids, 60, replace=False)
    orders = pd.DataFrame(
        {
            "order_id": range(300),
            "customer_id": rng.choice(buyers, 300),
            "amount": rng.gamma(2, 20, 300).round(2),
        }
    )
    customers.to_csv(work / "customers.csv", index=False)
    orders.to_csv(work / "orders.csv", index=False)
    (work / "README.txt").write_text(
        "customers.csv: customer_id is a 5-character zero-padded code, e.g. 00042.\n"
        "orders.csv: customer_id was exported by another system as a plain number, e.g. 42.\n"
    )
    totals = orders.groupby("customer_id")["amount"].sum()
    want = {f"{i:05d}": round(float(totals.get(i, 0.0)), 2) for i in ids}
    (hidden / "answer.json").write_text(json.dumps(want))


def _check_merge(work: Path, hidden: Path) -> CheckResult:
    out = _read(work / "totals.csv", dtype={"customer_id": str})
    if out is None:
        return _missing("totals.csv")
    if not {"customer_id", "total_amount"} <= set(out.columns):
        return CheckResult(False, f"columns {list(out.columns)}")
    want = json.loads((hidden / "answer.json").read_text())
    got = dict(zip(out["customer_id"], out["total_amount"], strict=False))
    if set(got) != set(want):
        return CheckResult(False, f"{len(set(want) - set(got))} customer ids missing or malformed")
    bad = [k for k in want if abs(float(got[k]) - want[k]) > 0.01]
    return CheckResult(not bad, f"{len(bad)} wrong totals" if bad else "ok")


MERGE = CodeTask(
    id="padded_keys",
    prompt=(
        "customers.csv and orders.csv are described in README.txt. Write totals.csv with one row per "
        "customer in customers.csv: customer_id (exactly as written in customers.csv) and total_amount (the "
        "sum of that customer's order amounts, 0 if they have no orders)."
    ),
    build=_build_merge,
    check=_check_merge,
    reference=(
        "import pandas as pd\nc = pd.read_csv('customers.csv', dtype={'customer_id': str})\n"
        "o = pd.read_csv('orders.csv')\no['customer_id'] = o['customer_id'].map(lambda i: f'{i:05d}')\n"
        "t = o.groupby('customer_id')['amount'].sum().rename('total_amount')\n"
        "out = c[['customer_id']].merge(t, on='customer_id', how='left').fillna({'total_amount': 0})\n"
        "out.to_csv('totals.csv', index=False)\n"
    ),
    broken=(
        "import pandas as pd\nc = pd.read_csv('customers.csv')\no = pd.read_csv('orders.csv')\n"
        "t = o.groupby('customer_id')['amount'].sum().rename('total_amount')\n"
        "out = c[['customer_id']].merge(t, on='customer_id', how='left').fillna({'total_amount': 0})\n"
        "out.to_csv('totals.csv', index=False)\n"
    ),
)


# --- 11. percentile clipping -----------------------------------------------------------------


def _build_clip(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 1000
    amount = rng.normal(100, 20, n)
    amount[rng.choice(n, 15, replace=False)] *= rng.uniform(5, 50, 15)
    df = pd.DataFrame({"txn_id": range(n), "amount": amount.round(2)})
    df.to_csv(work / "transactions.csv", index=False)
    lo, hi = np.percentile(df["amount"], [1, 99])
    df.assign(amount=df["amount"].clip(lo, hi)).to_csv(hidden / "answer.csv", index=False)


def _check_clip(work: Path, hidden: Path) -> CheckResult:
    out = _read(work / "clipped.csv")
    if out is None:
        return _missing("clipped.csv")
    want = pd.read_csv(hidden / "answer.csv")
    if "amount" not in out.columns or "txn_id" not in out.columns or len(out) != len(want):
        return CheckResult(False, f"shape/columns {out.shape} {list(out.columns)}")
    out = out.sort_values("txn_id").reset_index(drop=True)
    ok = np.allclose(out["amount"], want["amount"], atol=0.01)
    return CheckResult(bool(ok), "ok" if ok else "clipped values differ")


CLIP = CodeTask(
    id="percentile_clip",
    prompt=(
        "transactions.csv has txn_id and amount, with a few extreme outliers. Clip amount to the range "
        "between its 1st and 99th percentiles (numpy's default linear interpolation), keep every row, and "
        "write clipped.csv with columns txn_id, amount."
    ),
    build=_build_clip,
    check=_check_clip,
    reference=(
        "import numpy as np, pandas as pd\ndf = pd.read_csv('transactions.csv')\n"
        "lo, hi = np.percentile(df['amount'], [1, 99])\ndf['amount'] = df['amount'].clip(lo, hi)\n"
        "df.to_csv('clipped.csv', index=False)\n"
    ),
    broken=(
        "import numpy as np, pandas as pd\ndf = pd.read_csv('transactions.csv')\n"
        "lo, hi = np.percentile(df['amount'], [1, 99])\n"
        "df = df[(df['amount'] >= lo) & (df['amount'] <= hi)]\n"
        "df.to_csv('clipped.csv', index=False)\n"
    ),
)


# --- 12. evaluation without leakage ----------------------------------------------------------


def _build_cv(work: Path, hidden: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 600
    X = rng.normal(0, 1, (n, 6)) * np.array([1, 10, 100, 1, 1000, 1])
    logit = X[:, 0] + 0.1 * X[:, 1] - 0.01 * X[:, 2] + 0.001 * X[:, 4]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X.round(4), columns=[f"f{i}" for i in range(6)]).assign(label=y)
    df.to_csv(work / "data.csv", index=False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = df.drop(columns="label")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    auc = cross_val_score(model, features, df["label"], cv=cv, scoring="roc_auc").mean()
    (hidden / "answer.json").write_text(json.dumps({"cv_auc": float(auc)}))


def _check_cv(work: Path, hidden: Path) -> CheckResult:
    path = work / "result.json"
    if not path.exists():
        return _missing("result.json")
    try:
        got = float(json.loads(path.read_text())["cv_auc"])
    except (ValueError, KeyError, TypeError) as e:
        return CheckResult(False, f"bad result.json: {e}")
    want = json.loads((hidden / "answer.json").read_text())["cv_auc"]
    return CheckResult(abs(got - want) < 1e-4, f"got {got:.5f}, want {want:.5f}")


CV = CodeTask(
    id="cv_pipeline",
    prompt=(
        "data.csv has numeric features f0..f5 on very different scales and a binary label. Report the mean "
        "ROC AUC over 5-fold cross-validation of logistic regression (max_iter=1000) with standard scaling "
        "fitted inside each training fold only. Use StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=0). "
        'Write result.json as {"cv_auc": <number>}.'
    ),
    build=_build_cv,
    check=_check_cv,
    reference=(
        "import json, pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
        "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
        "df = pd.read_csv('data.csv')\nX, y = df.drop(columns='label'), df['label']\n"
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)\n"
        "m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\n"
        "json.dump({'cv_auc': float(cross_val_score(m, X, y, cv=cv, scoring='roc_auc').mean())}, "
        "open('result.json', 'w'))\n"
    ),
    broken=(
        "import json, pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
        "df = pd.read_csv('data.csv')\nX, y = df.drop(columns='label'), df['label']\n"
        "X = (X - X.mean()) / X.std()\n"
        "s = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring='roc_auc').mean()\n"
        "json.dump({'cv_auc': float(s)}, open('result.json', 'w'))\n"
    ),
)


TASKS: list[CodeTask] = [
    IMPUTE,
    SENTINEL,
    ONEHOT,
    DEDUPE,
    DATES,
    LEAK,
    REGRESSION,
    TOPK,
    UNSEEN,
    MERGE,
    CLIP,
    CV,
]
