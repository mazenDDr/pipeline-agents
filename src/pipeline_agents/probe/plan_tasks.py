"""Structured-plan probe: can a model return a valid, sensible plan as JSON? (T1)

Each case pairs a short dataset profile with a goal. A response passes in two stages:
1. **valid**: parses as JSON and matches `ProbePlan`;
2. **sound**: a few structural rules any reasonable pipeline plan follows.
The sound rules are deliberately loose: they catch broken plans, not style.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

StepKind = Literal["load", "profile", "clean", "feature", "split", "train", "evaluate", "analyze", "report"]


class ProbeStep(BaseModel):
    id: str
    kind: StepKind
    intent: str = Field(min_length=5)
    depends_on: list[str] = []
    acceptance_checks: list[str] = Field(min_length=1)


class ProbePlan(BaseModel):
    goal_type: Literal["predictive", "analytical"]
    target_column: str | None
    steps: list[ProbeStep] = Field(min_length=2, max_length=15)


@dataclass(frozen=True)
class PlanCase:
    id: str
    profile: str
    goal: str
    goal_type: str
    target: str | None


@dataclass(frozen=True)
class PlanVerdict:
    valid: bool
    sound: bool
    detail: str


def extract_json(text: str) -> str:
    """The first {...} block, tolerating code fences and prose around it."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def judge(case: PlanCase, text: str) -> PlanVerdict:
    try:
        plan = ProbePlan.model_validate(json.loads(extract_json(text)))
    except (json.JSONDecodeError, ValidationError) as e:
        return PlanVerdict(False, False, f"invalid: {str(e).splitlines()[0][:200]}")

    problems = []
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        problems.append("duplicate step ids")
    seen: set[str] = set()
    for step in plan.steps:
        if any(dep not in seen for dep in step.depends_on):
            problems.append(f"{step.id} depends on a later or unknown step")
        seen.add(step.id)
    if plan.goal_type != case.goal_type:
        problems.append(f"goal_type {plan.goal_type} != {case.goal_type}")
    kinds = [s.kind for s in plan.steps]
    if case.goal_type == "predictive":
        if plan.target_column != case.target:
            problems.append(f"target {plan.target_column} != {case.target}")
        if "train" not in kinds or "evaluate" not in kinds:
            problems.append("predictive plan without train and evaluate")
        elif "split" not in kinds or kinds.index("split") > kinds.index("train"):
            problems.append("no split before training")
        elif kinds.index("evaluate") < kinds.index("train"):
            problems.append("evaluate before train")
    elif "analyze" not in kinds:
        problems.append("analytical plan without an analyze step")
    return PlanVerdict(True, not problems, "; ".join(problems) or "ok")


SCHEMA = ProbePlan.model_json_schema()

CASES: list[PlanCase] = [
    PlanCase(
        "churn",
        "telecom.csv: 7,043 rows. customer_id (str, unique), tenure (int), monthly_charges (float), "
        "total_charges (str, 11 blanks), contract (str: month-to-month/one year/two year), churn "
        "(str: Yes/No).",
        "Build a model to predict whether a customer will churn.",
        "predictive",
        "churn",
    ),
    PlanCase(
        "house_prices",
        "houses.csv: 1,460 rows, 81 columns. SalePrice (int, right-skewed), LotArea (int), YearBuilt (int), "
        "Neighborhood (str, 25 values), PoolQC (str, 99.5% null), GarageYrBlt (float, 5.5% null).",
        "Predict SalePrice for new listings.",
        "predictive",
        "SalePrice",
    ),
    PlanCase(
        "bike_drivers",
        "rentals.csv: 17,379 rows. dteday (str date), hr (int 0-23), season (int 1-4), weathersit (int 1-4), "
        "temp (float, normalized), hum (float), casual (int), registered (int), cnt (int = casual + "
        "registered).",
        "Find which conditions drive hourly bike demand and summarize the main effects.",
        "analytical",
        None,
    ),
    PlanCase(
        "credit_default",
        "loans.csv: 30,000 rows. id (int), limit_bal (float), sex (int 1-2), education (int 0-6), age (int), "
        "pay_0..pay_6 (int, -2 to 8), bill_amt1..6 (float), default_next_month (int 0/1, 22% positive).",
        "Predict default next month; the positive class is rare, so pick a sensible metric.",
        "predictive",
        "default_next_month",
    ),
    PlanCase(
        "sales_quality",
        "orders.csv: 51,290 rows. order_id (str, 3% duplicated), order_date (str, two formats), "
        "region (str), "
        "sales (float, some negative), discount (float 0-0.8), profit (float).",
        "Clean this dataset and report which regions lose money on discounted orders.",
        "analytical",
        None,
    ),
    PlanCase(
        "readmission",
        "patients.csv: 101,766 rows. encounter_id, patient_nbr (repeats across encounters), race "
        "(str, '?' for "
        "unknown), age (str bands), num_medications (int), discharge_disposition_id (int code), "
        "readmitted (str: NO/<30/>30).",
        "Predict readmission within 30 days without leaking patients between train and test.",
        "predictive",
        "readmitted",
    ),
    PlanCase(
        "wine",
        "wine.csv: 6,497 rows. 11 physico-chemical float columns, color (str: red/white), quality (int 3-9).",
        "Explain which properties are associated with higher quality, separately for red and white wine.",
        "analytical",
        None,
    ),
    PlanCase(
        "fraud",
        "transactions.csv: 284,807 rows. time (float seconds), v1..v28 (float, PCA components), "
        "amount (float), "
        "class (int 0/1, 0.17% positive).",
        "Build a fraud detector and report precision at a fixed recall of 0.8.",
        "predictive",
        "class",
    ),
]

SYSTEM = (
    "You are the planning component of a data-science agent. Given a dataset profile and a goal, return a "
    "step-by-step plan as a single JSON object and nothing else. Each step has: id (s1, s2, ...), kind (one "
    "of load, profile, clean, feature, split, train, evaluate, analyze, report), intent (what the step does "
    "and why), depends_on (ids of earlier steps it needs), and acceptance_checks (concrete checks that show "
    'the step worked). Also give goal_type ("predictive" or "analytical") and target_column (null for '
    "analytical goals)."
)


def user_prompt(case: PlanCase, with_schema: bool) -> str:
    text = f"Dataset profile:\n{case.profile}\n\nGoal: {case.goal}\n"
    if with_schema:
        text += (
            "\nYour answer must be a plan object that conforms to this JSON schema. Return the plan, "
            f"not the schema:\n{json.dumps(SCHEMA)}\n"
        )
    return text
