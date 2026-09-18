"""Field-test scoring: the analytical comparison and the prediction scoring, on small frames."""

import pandas as pd

from pipeline_agents.bench.field import (
    STUDENTS,
    WINE,
    score_analytical,
    score_predictions,
    split,
    students_answer,
)


def test_every_reference_number_must_appear_within_tolerance() -> None:
    reference = {"students": 2, "by_group": {"0": 10.0, "1": 8.0}, "peak": "2008-12-27"}
    passed, problems = score_analytical(STUDENTS, dict(reference), reference)
    assert passed and problems == []
    close = {"students": 2, "by_group": {"0": 10.0005, "1": 8.0}, "peak": "2008-12-27"}
    assert score_analytical(STUDENTS, close, reference)[0]
    off = {"students": 2, "by_group": {"0": 10.5, "1": 8.0}, "peak": "2008-12-27"}
    passed, problems = score_analytical(STUDENTS, off, reference)
    assert not passed and "by_group.0" in problems[0]
    missing = {"students": 2, "by_group": {"0": 10.0}, "peak": "2008-12-27"}
    assert "by_group.1: missing" in score_analytical(STUDENTS, missing, reference)[1]
    wrong_day = {**reference, "peak": "2008-12-28"}
    assert not score_analytical(STUDENTS, wrong_day, reference)[0]


def test_the_student_reference_converts_text_columns() -> None:
    frame = pd.DataFrame({"failures": ["0", "0", "1"], "studytime": ["1", "2", "2"], "G3": ["10", "12", "6"]})
    answer = students_answer(frame)
    assert answer["students"] == 3
    assert answer["mean_final_grade_by_failures"] == {"0": 11.0, "1": 6.0}
    assert answer["mean_final_grade_by_studytime"] == {"1": 10.0, "2": 9.0}


def test_predictions_are_matched_on_the_id_and_scored() -> None:
    holdout = pd.DataFrame({"sample_id": ["1", "2"], "quality": ["6", "5"], "alcohol": ["9", "10"]})
    good = pd.DataFrame({"sample_id": [1, 2], "prediction": [6.5, 5.0]})
    mae, detail = score_predictions(WINE, good, holdout)
    assert mae == 0.25 and "2 held-back" in detail
    short = pd.DataFrame({"sample_id": [1], "prediction": [6.0]})
    assert str(score_predictions(WINE, short, holdout)[0]) == "nan"
    wrong_columns = pd.DataFrame({"id": [1, 2], "prediction": [6.0, 5.0]})
    assert "predictions need columns" in score_predictions(WINE, wrong_columns, holdout)[1]


def test_the_split_adds_an_id_and_holds_rows_back() -> None:
    frame = pd.DataFrame({"quality": [str(i % 7) for i in range(100)]})
    seen, holdout = split(WINE, frame)
    assert len(seen) == 80 and len(holdout) == 20
    assert "sample_id" in seen.columns and not set(seen["sample_id"]) & set(holdout["sample_id"])
