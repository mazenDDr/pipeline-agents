"""A field test: datasets the benchmark has never seen, scored by hand-written references (T16).

The benchmark tasks were designed with their traps and thresholds. A field test asks a different question:
what happens on data nobody prepared for this system? Three public datasets, none of them in the benchmark,
each with a goal written the way a person would write it, and an answer I compute myself in pandas.

Nothing here is given to the agents except the goal, the data dictionary and the data.
"""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW = Path("data/field")


@dataclass(frozen=True)
class FieldTask:
    id: str
    url: str
    member: str  # the file inside the archive
    goal: str
    kind: str  # analytical | predictive
    readme: str
    separator: str = ","
    id_column: str | None = None
    target: str | None = None
    metric: str = "mae"
    holdout_fraction: float = 0.2
    tolerance: float = 0.001  # analytical: relative difference allowed on each number
    threshold: float | None = None  # predictive: the metric the delivered model must reach


WINE = FieldTask(
    id="wine-quality",
    url="https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    member="winequality-white.csv",
    kind="predictive",
    separator=";",
    id_column="sample_id",
    target="quality",
    metric="mae",
    # Measured on this exact split before any agent ran: always predicting the mean gives MAE 0.658, a
    # linear regression 0.575, a random forest 0.424. The threshold sits between the linear model and the
    # forest, so a delivered model has to be better than a one-line regression.
    threshold=0.50,
    goal=(
        "A winery wants to score the quality of a white wine from its laboratory measurements, before the "
        "tasting panel sees it. `data/wine.csv` holds measurements and the panel's score for wines that have "
        "already been tasted. Build a model that predicts `quality` for new wines; it will be judged by mean "
        "absolute error on wines held back from you."
    ),
    readme=(
        "# wine.csv\n\n"
        "Physicochemical measurements of white *vinho verde* wines and the median score of three sensory "
        "assessors, from the UCI Machine Learning Repository (Wine Quality, Cortez et al., 2009; CC BY 4.0). "
        "The file is the original export: semicolon-separated.\n\n"
        "- sample_id: row identifier (added for this task)\n"
        "- fixed acidity, volatile acidity, citric acid, residual sugar, chlorides: g/dm^3\n"
        "- free sulfur dioxide, total sulfur dioxide: mg/dm^3\n"
        "- density: g/cm^3\n"
        "- pH, sulphates, alcohol: pH units, g/dm^3, % by volume\n"
        "- quality: the panel's score, an integer from 0 to 10\n"
    ),
)

STUDENTS = FieldTask(
    id="student-grades",
    url="https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    member="student-mat.csv",
    kind="analytical",
    separator=";",
    goal=(
        "Using `data/students.csv` (see `data/README.md`), describe how the final grade relates to past "
        "failures and to time spent studying.\n\n"
        "`output/answer.json` must be an object with three keys:\n"
        '- `"students"`: the number of students in the file;\n'
        '- `"mean_final_grade_by_failures"`: an object mapping the number of past class failures '
        '("0", "1", "2", "3") to the mean of `G3` for those students;\n'
        '- `"mean_final_grade_by_studytime"`: an object mapping the study-time code ("1", "2", "3", "4") to '
        "the mean of `G3` for those students."
    ),
    readme=(
        "# students.csv\n\n"
        "Secondary-school mathematics results from two Portuguese schools, from the UCI Machine Learning "
        "Repository (Student Performance, Cortez and Silva, 2008; CC BY 4.0). The file is the original "
        "export: semicolon-separated, with text values in double quotes. One row per student.\n\n"
        "- school, sex, age, address, famsize, Pstatus: background\n"
        "- studytime: weekly study time as a code: 1 = under 2 hours, 2 = 2 to 5, 3 = 5 to 10, 4 = over 10\n"
        "- failures: number of past class failures (3 means three or more)\n"
        "- G1, G2: first and second period grades, 0 to 20\n"
        "- G3: final grade, 0 to 20\n"
        "- the remaining columns describe family, support and free time\n"
    ),
)

POWER = FieldTask(
    id="household-power",
    url="https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    member="household_power_consumption.txt",
    kind="analytical",
    separator=";",
    goal=(
        "Using `data/power.csv` (see `data/README.md`), summarise household electricity use in December "
        "2008.\n\n"
        "`output/answer.json` must be an object with three keys:\n"
        '- `"missing_minutes"`: the number of minutes in December 2008 whose active power reading is '
        "missing;\n"
        '- `"daily_mean_active_power"`: an object mapping each date of December 2008 as "YYYY-MM-DD" to the '
        "mean of `Global_active_power` over that day's valid readings, in kilowatts;\n"
        '- `"peak_day"`: the date with the highest mean.'
    ),
    readme=(
        "# power.csv\n\n"
        "One row per minute of electricity use in one French household, from the UCI Machine Learning "
        "Repository (Individual Household Electric Power Consumption, Hebrail and Berard, 2012; CC BY 4.0). "
        "The file is the original export: semicolon-separated, about 2 million rows.\n\n"
        "- Date: the day, as DD/MM/YYYY\n"
        "- Time: the minute, as HH:MM:SS\n"
        "- Global_active_power: household global minute-averaged active power, in kilowatts\n"
        "- Global_reactive_power, Voltage, Global_intensity: the rest of the electrical measurements\n"
        "- Sub_metering_1, Sub_metering_2, Sub_metering_3: watt-hours of active energy per minute in the "
        "kitchen, the laundry room and for the water heater and air conditioner\n\n"
        "Missing readings are written as `?`. About 1.25% of the rows have no measurements at all.\n"
    ),
)

TASKS = [WINE, STUDENTS, POWER]
BY_ID = {t.id: t for t in TASKS}


# --- references: what the answer should be, computed here and never shown to the agents ----------


def students_answer(df: pd.DataFrame) -> dict:
    # raw_frame reads everything as text, as the file has it; the reference does its own conversion.
    grade = pd.to_numeric(df["G3"])
    means = {
        column: {str(int(k)): float(v) for k, v in grade.groupby(pd.to_numeric(df[column])).mean().items()}
        for column in ("failures", "studytime")
    }
    return {
        "students": int(len(df)),
        "mean_final_grade_by_failures": means["failures"],
        "mean_final_grade_by_studytime": means["studytime"],
    }


def power_answer(df: pd.DataFrame) -> dict:
    stamp = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    power = pd.to_numeric(df["Global_active_power"], errors="coerce")
    december = (stamp.dt.year == 2008) & (stamp.dt.month == 12)
    day = stamp[december].dt.strftime("%Y-%m-%d")
    daily = power[december].groupby(day).mean()
    return {
        "missing_minutes": int(power[december].isna().sum()),
        "daily_mean_active_power": {d: float(v) for d, v in daily.items()},
        "peak_day": str(daily.idxmax()),
    }


REFERENCES = {"student-grades": students_answer, "household-power": power_answer}


# --- preparing and scoring -----------------------------------------------------------------------


def unzip(archive: Path, dest: Path) -> None:
    """Extract, including zips nested inside the archive (UCI often ships a zip of zips)."""
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    for inner in dest.rglob("*.zip"):
        if inner.resolve() == archive.resolve():
            continue
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
        inner.unlink()


def raw_frame(task: FieldTask, root: Path = RAW) -> pd.DataFrame:
    """The dataset as it is on disk, with nothing cleaned: the agents get the same file."""
    path = next((root / task.id).rglob(task.member))
    return pd.read_csv(path, sep=task.separator, dtype=str, keep_default_na=False)


def split(task: FieldTask, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For a predictive task: the rows the agents see, and the rows held back to score them."""
    if task.kind != "predictive":
        return frame, frame.iloc[:0]
    frame = frame.copy()
    if task.id_column and task.id_column not in frame.columns:
        frame.insert(0, task.id_column, range(1, len(frame) + 1))
    cut = int(len(frame) * (1 - task.holdout_fraction))
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return shuffled.iloc[:cut], shuffled.iloc[cut:]


def score_analytical(task: FieldTask, answer: dict, reference: dict) -> tuple[bool, list[str]]:
    """Every number in the reference must appear in the answer, within the tolerance."""
    problems: list[str] = []

    def compare(path: str, want, got) -> None:
        if isinstance(want, dict):
            if not isinstance(got, dict):
                problems.append(f"{path}: expected an object, got {type(got).__name__}")
                return
            for key, value in want.items():
                if key not in got:
                    problems.append(f"{path}.{key}: missing")
                else:
                    compare(f"{path}.{key}", value, got[key])
            return
        if isinstance(want, str):
            if str(got).strip() != want:
                problems.append(f"{path}: got {got!r}, want {want!r}")
            return
        try:
            number = float(got)
        except (TypeError, ValueError):
            problems.append(f"{path}: got {got!r}, want {want}")
            return
        scale = max(abs(float(want)), 1e-9)
        if abs(number - float(want)) / scale > task.tolerance:
            problems.append(f"{path}: got {number}, want {want}")

    for key, value in reference.items():
        if key not in answer:
            problems.append(f"{key}: missing")
        else:
            compare(key, value, answer[key])
    return not problems, problems[:8]


def score_predictions(task: FieldTask, predictions: pd.DataFrame, holdout: pd.DataFrame) -> tuple[float, str]:
    """Mean absolute error over the held-back rows, matched on the id column."""
    ids = task.id_column or "id"
    if ids not in predictions.columns or "prediction" not in predictions.columns:
        return float("nan"), f"predictions need columns {ids} and prediction, got {list(predictions.columns)}"
    merged = holdout.merge(
        predictions.assign(**{ids: predictions[ids].astype(str)}),
        left_on=holdout[ids].astype(str),
        right_on=ids,
        suffixes=("", "_pred"),
    )
    if len(merged) != len(holdout):
        return float("nan"), f"{len(merged)} of {len(holdout)} held-back rows got a prediction"
    want = pd.to_numeric(merged[task.target], errors="coerce")
    got = pd.to_numeric(merged["prediction"], errors="coerce")
    if got.isna().any():
        return float("nan"), f"{int(got.isna().sum())} predictions are not numbers"
    mae = float((want - got).abs().mean())
    return mae, f"mae {mae:.4f} over {len(merged):,} held-back wines"


def load_answer(workspace: Path) -> dict | None:
    path = workspace / "output" / "answer.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None
