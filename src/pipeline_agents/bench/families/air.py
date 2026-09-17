"""Air Quality (UCI 360, CC BY 4.0), dev family: hourly gas-sensor readings in an Italian city, 2004-05."""

from pathlib import Path

import pandas as pd

from pipeline_agents.bench.build import task_dir, write_answer, write_holdout
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

ESTIMATE_FROM = pd.Timestamp("2005-02-01")  # the analyzer is "retired" from here on: the holdout period
ANALYZER_COLUMNS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

README = """# air_quality.csv

Hourly averaged readings from a gas multisensor device at road level in a polluted area of an Italian city,
from the UCI Machine Learning Repository (Air Quality, De Vito et al., 2008; CC BY 4.0). The file is the
original export: semicolon-separated, with a comma as the decimal separator. Missing values are tagged with
the value -200.

- reading_id: row identifier (added for this task)
- Date: date (DD/MM/YYYY)
- Time: time (HH.MM.SS)
- CO(GT): true hourly averaged CO concentration in mg/m^3, from the reference analyzer
- PT08.S1(CO): hourly averaged sensor response of a tin oxide sensor (nominally CO targeted)
- NMHC(GT): true hourly averaged non-methane hydrocarbons concentration in microg/m^3, from the reference
  analyzer
- C6H6(GT): true hourly averaged benzene concentration in microg/m^3, from the reference analyzer
- PT08.S2(NMHC): hourly averaged sensor response of a titania sensor (nominally NMHC targeted)
- NOx(GT): true hourly averaged NOx concentration in ppb, from the reference analyzer
- PT08.S3(NOx): hourly averaged sensor response of a tungsten oxide sensor (nominally NOx targeted)
- NO2(GT): true hourly averaged NO2 concentration in microg/m^3, from the reference analyzer
- PT08.S4(NO2): hourly averaged sensor response of a tungsten oxide sensor (nominally NO2 targeted)
- PT08.S5(O3): hourly averaged sensor response of an indium oxide sensor (nominally O3 targeted)
- T: temperature in degrees Celsius
- RH: relative humidity (%)
- AH: absolute humidity

Columns marked (GT) come from a certified reference analyzer co-located with the cheap metal oxide
sensors (PT08.*).
"""

DAILY = TaskSpec(
    id="air-1-co-daily",
    family="air",
    split="dev",
    role="warmup",
    kind="analytical",
    difficulty="medium",
    goal=(
        "Using `data/air_quality.csv` (see `data/README.md`), summarise the true carbon monoxide "
        "concentration "
        "`CO(GT)` for November 2004: the mean for each day, using only valid readings, and how many hourly "
        "readings of `CO(GT)` are missing that month."
    ),
    deliverable=(
        "`output/answer.json` must be an object with two keys:\n"
        '- `"missing_co_hours"`: the number of hours in November 2004 whose `CO(GT)` reading is missing;\n'
        '- `"daily_mean_co"`: an object mapping each date of November 2004 as `"YYYY-MM-DD"` to the mean '
        "valid `CO(GT)` of that day, in mg/m^3."
    ),
    traps=[
        Trap(
            id="minus_200_missing",
            kind="sentinel",
            caught_by="answer",
            description="Missing readings are -200 (README); averaging them in drags daily means far down.",
        ),
        Trap(
            id="decimal_comma",
            kind="format",
            caught_by="answer",
            description="Semicolon-separated with decimal commas: '2,6' is 2.6 (README).",
        ),
        Trap(
            id="day_first_dates",
            kind="format",
            caught_by="answer",
            description="Dates are DD/MM/YYYY; month-first parsing moves readings to the wrong day.",
        ),
    ],
    analytical=AnalyticalCheck(numeric_tolerance=1e-3),
)

ESTIMATE = TaskSpec(
    id="air-2-co-estimate",
    family="air",
    split="dev",
    role="followup",
    kind="predictive",
    difficulty="hard",
    goal=(
        "The certified reference analyzer is being removed. From then on, the true CO concentration `CO(GT)` "
        "has to be estimated from the cheap metal oxide sensors and the weather readings. Using "
        "`data/air_quality.csv` (see `data/README.md`), build a model that estimates `CO(GT)` for each hour; "
        "it will be judged by mean absolute error on the months after your data, on hours where the analyzer "
        "reading exists."
    ),
    deliverable="The id column is `reading_id`; `prediction` is the estimated `CO(GT)` in mg/m^3.",
    traps=[
        Trap(
            id="analyzer_columns_unavailable",
            kind="target_leak",
            caught_by="predict",
            description="Every (GT) column comes from the analyzer being removed (README), so none of them "
            "exists in the future features.",
        ),
        Trap(
            id="minus_200_target",
            kind="sentinel",
            caught_by="score",
            description="-200 marks a missing reading, including in CO(GT); training on those rows teaches "
            "the model to predict -200.",
        ),
        Trap(
            id="decimal_comma",
            kind="format",
            caught_by="clean_rerun",
            description="Decimal commas: parsed without decimal=',', the sensor columns are text.",
        ),
        Trap(
            id="random_split_validation",
            kind="temporal",
            caught_by="score only",
            description="Sensors drift over months; a random split overstates accuracy on later months. "
            "Not caught: honest estimates on this data also miss by 36-57%, so the honesty check is off.",
        ),
    ],
    # Measured on the holdout: calibrating CO against its own sensor 0.598, the best single sensor 0.505, the
    # median 1.005; multi-sensor models 0.366-0.384. Honesty is not checked (tolerance 1.0): honest time-based
    # estimates missed the holdout by 36-57% because December-January is more polluted than February-April,
    # while a random split missed by 28%, so no tolerance separates them.
    predictive=PredictiveCheck(id_column="reading_id", metric="mae", threshold=0.45, honesty_tolerance=1.0),
)

TASKS = [DAILY, ESTIMATE]


def _split_raw(path: Path) -> tuple[str, list[str], list[str], list[str]]:
    """Header, lines before ESTIMATE_FROM, lines from it, and the trailing empty lines: original text kept."""
    text = path.read_bytes().decode("latin-1")
    header, *lines = text.split("\r\n")
    before, after, empty = [], [], []
    for line in lines:
        date = line.split(";", 1)[0]
        if not date:
            empty.append(line)
        elif pd.Timestamp(f"{date[6:10]}-{date[3:5]}-{date[0:2]}") < ESTIMATE_FROM:
            before.append(line)
        else:
            after.append(line)
    return header, before, after, [e for e in empty if e]


def _with_ids(header: str, lines: list[str], start: int) -> tuple[str, list[str]]:
    return "reading_id;" + header, [f"{start + i};{line}" for i, line in enumerate(lines)]


def _frame(header: str, lines: list[str]) -> pd.DataFrame:
    rows = [line.split(";") for line in lines]
    df = pd.DataFrame(rows, columns=header.split(";"))
    return df.loc[:, [c for c in df.columns if c]]


def _number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", ".", regex=False))


def build(raw: Path, out: Path, tasks: list[TaskSpec]) -> None:
    header, before, after, empty = _split_raw(raw / "air_quality" / "AirQualityUCI.csv")
    id_header, train_lines = _with_ids(header, before, 1)
    train_text = "\r\n".join([id_header, *train_lines, *[";" + e for e in empty]]) + "\r\n"
    by_id = {t.id: t for t in tasks}

    task = by_id[DAILY.id]
    workspace, hidden = task_dir(out, task, README)
    (workspace / "data" / "air_quality.csv").write_bytes(train_text.encode("latin-1"))
    df = _frame(id_header, train_lines)
    df["day"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    nov = df[df["day"].dt.strftime("%Y-%m") == "2004-11"].assign(co=lambda d: _number(d["CO(GT)"]))
    valid = nov[nov["co"] != -200]
    write_answer(
        hidden,
        {
            "missing_co_hours": int((nov["co"] == -200).sum()),
            "daily_mean_co": {
                d.strftime("%Y-%m-%d"): float(v) for d, v in valid.groupby("day")["co"].mean().items()
            },
        },
    )

    task = by_id[ESTIMATE.id]
    workspace, hidden = task_dir(out, task, README)
    (workspace / "data" / "air_quality.csv").write_bytes(train_text.encode("latin-1"))
    fut_header, fut_lines = _with_ids(header, after, 1 + len(before))
    future = _frame(fut_header, fut_lines)
    future = future[_number(future["CO(GT)"]) != -200]  # only hours the analyzer measured can be scored
    features = future.drop(columns=ANALYZER_COLUMNS)
    text = "\r\n".join(
        [";".join(features.columns) + ";;", *[";".join(r) + ";;" for r in features.itertuples(index=False)]]
    )
    write_holdout(
        hidden,
        features.assign(reading_id=features["reading_id"].astype(int)),
        "reading_id",
        _number(future["CO(GT)"]),
        (text + "\r\n"),
    )
