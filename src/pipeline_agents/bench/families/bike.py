"""Bike Sharing (UCI 275, CC BY 4.0), dev family: hourly rentals in Washington D.C., 2011-2012."""

from pathlib import Path

import pandas as pd

from pipeline_agents.bench.build import task_dir, write_answer, write_holdout
from pipeline_agents.bench.spec import AnalyticalCheck, PredictiveCheck, TaskSpec, Trap

HOLDOUT_FROM = "2012-11-01"  # the last two months are the future the forecast is judged on

README = """# rentals_hourly.csv

Hourly counts of rental bikes from a bike-sharing system in Washington D.C., from the UCI Machine Learning
Repository (Bike Sharing Dataset, Fanaee-T and Gama, 2013; CC BY 4.0). One row per hour.

- instant: record index
- dteday: date
- season: season (1: spring, 2: summer, 3: fall, 4: winter)
- yr: year (0: 2011, 1: 2012)
- mnth: month (1 to 12)
- hr: hour (0 to 23)
- holiday: whether the day is a holiday
- weekday: day of the week
- workingday: 1 if the day is neither a weekend nor a holiday, otherwise 0
- weathersit:
  - 1: Clear, few clouds, partly cloudy
  - 2: Mist + cloudy, mist + broken clouds, mist + few clouds, mist
  - 3: Light snow, light rain + thunderstorm + scattered clouds, light rain + scattered clouds
  - 4: Heavy rain + ice pellets + thunderstorm + mist, snow + fog
- temp: normalized temperature in Celsius; the values are divided by 41 (max)
- atemp: normalized feeling temperature in Celsius; the values are divided by 50 (max)
- hum: normalized humidity; the values are divided by 100 (max)
- windspeed: normalized wind speed; the values are divided by 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes, including both casual and registered
"""

FORECAST = TaskSpec(
    id="bike-1-forecast",
    family="bike",
    split="dev",
    role="warmup",
    kind="predictive",
    difficulty="medium",
    goal=(
        "The operator wants to forecast total hourly rentals (`cnt`) so it can move bikes between stations. "
        "`data/rentals_hourly.csv` holds the history. Build a model that forecasts `cnt` for future hours; "
        "it will be judged on the two months that follow the data you have, by mean absolute error."
    ),
    deliverable="The id column is `instant`; `prediction` is the forecast count for that hour.",
    traps=[
        Trap(
            id="leak_casual_registered",
            kind="target_leak",
            caught_by="predict",
            description="casual + registered = cnt; neither is known before the hour happens, so the "
            "future features do not contain them (README: cnt includes both).",
        ),
        Trap(
            id="random_split_validation",
            kind="temporal",
            caught_by="honest_estimate",
            description="A random split puts neighbouring hours of the same days in train and validation, "
            "which understates the error of a forecast into later months.",
        ),
        Trap(
            id="int_coded_categories",
            kind="int_coded",
            caught_by="score only",
            description="season, weathersit, weekday and hr are codes, not quantities.",
        ),
    ],
    # Measured on the holdout (outputs/runs/t2-validate): tree models 45-52, the obvious baseline (mean cnt
    # per hour x workingday) 63.5, linear models 66-80, the overall mean 140. A useful forecast beats the
    # baseline.
    # Honesty: time-based validation gaps 0.00-0.09; a random 15% split 0.48.
    predictive=PredictiveCheck(id_column="instant", metric="mae", threshold=60.0, honesty_tolerance=0.25),
)

WEATHER = TaskSpec(
    id="bike-2-weather",
    family="bike",
    split="dev",
    role="followup",
    kind="analytical",
    difficulty="easy",
    goal=(
        "Using `data/rentals_hourly.csv`, describe demand on working days by weather: the mean number of "
        "rentals per hour under each weather situation, and the mean temperature in degrees Celsius."
    ),
    deliverable=(
        "`output/answer.json` must be an object with two keys:\n"
        '- `"mean_rentals_per_hour"`: an object with keys `"clear"`, `"mist"`, `"light_rain_snow"`, '
        '`"heavy_rain_snow"` (weathersit 1 to 4), each the mean `cnt` over working-day hours;\n'
        '- `"mean_temp_celsius"`: the mean temperature over working-day hours, in degrees Celsius.'
    ),
    traps=[
        Trap(
            id="weathersit_codes",
            kind="int_coded",
            caught_by="answer",
            description="weathersit is a code 1-4 whose meaning is in the README.",
        ),
        Trap(
            id="normalized_temp",
            kind="semantics",
            caught_by="answer",
            description="temp is divided by 41; degrees Celsius need temp * 41 (README).",
        ),
        Trap(
            id="workingday_filter",
            kind="semantics",
            caught_by="answer",
            description="Only working-day hours count (workingday == 1).",
        ),
    ],
    analytical=AnalyticalCheck(numeric_tolerance=1e-3),
)

TASKS = [FORECAST, WEATHER]


def build(raw: Path, out: Path, tasks: list[TaskSpec]) -> None:
    df = pd.read_csv(raw / "bike_sharing" / "hour.csv")
    history = df[df["dteday"] < HOLDOUT_FROM]
    future = df[df["dteday"] >= HOLDOUT_FROM]
    by_id = {t.id: t for t in tasks}

    task = by_id[FORECAST.id]
    workspace, hidden = task_dir(out, task, README)
    history.to_csv(workspace / "data" / "rentals_hourly.csv", index=False)
    write_holdout(hidden, future.drop(columns=["casual", "registered", "cnt"]), "instant", future["cnt"])

    task = by_id[WEATHER.id]
    workspace, hidden = task_dir(out, task, README)
    history.to_csv(workspace / "data" / "rentals_hourly.csv", index=False)
    working = history[history["workingday"] == 1]
    labels = {1: "clear", 2: "mist", 3: "light_rain_snow", 4: "heavy_rain_snow"}
    means = working.groupby("weathersit")["cnt"].mean()
    write_answer(
        hidden,
        {
            "mean_rentals_per_hour": {labels[k]: float(means.get(k, float("nan"))) for k in labels},
            "mean_temp_celsius": float((working["temp"] * 41).mean()),
        },
    )
