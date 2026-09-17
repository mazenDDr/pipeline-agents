"""Profiler detectors, each on a small planted file, and the false alarms seen on real data staying away."""

from pathlib import Path

import numpy as np
import pytest

from pipeline_agents.tools.profile import profile_dir, profile_file


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _rows(n: int, fmt) -> str:
    rng = np.random.default_rng(0)
    return "".join(fmt(i, rng) for i in range(n))


def test_headerless_file_with_spaces_and_question_marks(tmp_path: Path) -> None:
    text = _rows(
        300, lambda i, r: f"{i}, {r.integers(18, 80)}, {r.choice(['Private', '?', 'State-gov'])}, >50K\n"
    )
    rendered = profile_file(_write(tmp_path, "census.data", text)).render()
    assert "NO HEADER ROW" in rendered
    assert "leading/trailing spaces" in rendered and "placeholder values '?'" in rendered


def test_two_header_rows(tmp_path: Path) -> None:
    text = ",X1,X2,Y\nID,LIMIT_BAL,AGE,default\n" + _rows(
        200, lambda i, r: f"{i},{r.integers(1, 9) * 10000},{r.integers(21, 70)},{r.integers(0, 2)}\n"
    )
    profile = profile_file(_write(tmp_path, "credit.csv", text))
    assert profile.header_rows == 2 and [c.name for c in profile.columns] == [
        "ID",
        "LIMIT_BAL",
        "AGE",
        "default",
    ]


def test_semicolons_decimal_commas_sentinels_day_first_crlf_trailing(tmp_path: Path) -> None:
    def row(i, r):
        co = "-200" if i % 7 == 0 else f"{r.uniform(0.1, 9):.1f}".replace(".", ",")
        return f"{(i % 28) + 1:02d}/03/2004;{co};{r.integers(600, 2000)};;\r\n"

    text = "Date;CO(GT);PT08.S1;;\r\n" + _rows(400, row) + ";;;;\r\n" * 5
    rendered = profile_file(_write(tmp_path, "air.csv", text)).render()
    for phrase in (
        "separator ';'",
        "CRLF",
        "5 rows with no values",
        "2 trailing column",
        "DECIMAL COMMA",
        "-200 appears",
        "DAY first",
    ):
        assert phrase in rendered, phrase


def test_month_first_dates_found_even_when_early_rows_are_ambiguous(tmp_path: Path) -> None:
    early = "".join(f"12/{d}/2010 8:26,1\n" for d in [1, 2, 3] * 2000)  # first 6,000 rows: day <= 12
    late = "".join(f"3/{d}/2011 9:00,2\n" for d in range(13, 29))
    rendered = profile_file(_write(tmp_path, "tx.csv", "InvoiceDate,Quantity\n" + early + late)).render()
    assert "MONTH first" in rendered


def test_pdays_style_sentinel_is_the_mode(tmp_path: Path) -> None:
    text = "client,pdays\n" + _rows(1000, lambda i, r: f"{i},{999 if i % 25 else r.integers(0, 27)}\n")
    assert "999 appears in 96.0% of rows" in profile_file(_write(tmp_path, "bank.csv", text)).render()


def test_no_sentinel_alarm_on_binary_or_ordinary_columns(tmp_path: Path) -> None:
    text = "flag,target,amount,hour\n" + _rows(
        500, lambda i, r: f"{r.integers(0, 2)},{int(r.random() < 0.2)},{r.gamma(2, 50):.2f},{i % 24}\n"
    )
    assert "missing-value code" not in profile_file(_write(tmp_path, "ok.csv", text)).render()


def test_identifiers_codes_and_mixed_codes(tmp_path: Path) -> None:
    def row(i, r):
        diag = f"V{r.integers(10, 60)}" if i % 50 == 0 else str(r.integers(1, 999))
        return (
            f"{i},{i // 3},{r.integers(1, 9)},{diag},C{536000 + i}\n"
            if i % 40 == 0
            else f"{i},{i // 3},{r.integers(1, 9)},{diag},{536000 + i}\n"
        )

    rendered = profile_file(
        _write(
            tmp_path,
            "enc.csv",
            "encounter_id,patient_nbr,admission_type_id,diag_1,InvoiceNo\n" + _rows(600, row),
        )
    ).render()
    assert "encounter_id: numeric, 600 distinct" in rendered and "unique per row" in rendered
    assert "an identifier that repeats: 3.00 rows per value" in rendered
    assert "8 integer codes named like an identifier" in rendered
    assert "not numbers (e.g. 'V" in rendered and "not numbers (e.g. 'C536" in rendered


def test_stacked_tables_and_duplicates(tmp_path: Path) -> None:
    text = (
        "type_id,description\n1,Emergency\n2,Urgent\n,\n"
        "discharge_id,description\n1,Home\n11,Expired\n1,Home\n"
    )
    rendered = profile_file(_write(tmp_path, "ids.csv", text)).render()
    assert "look like the header again" in rendered and "1 exact duplicate rows" in rendered


@pytest.mark.parametrize("name", ["README.md", "readme.txt"])
def test_profile_dir_skips_documentation(tmp_path: Path, name: str) -> None:
    _write(tmp_path, name, "not data")
    _write(tmp_path, "a.csv", "x,y\n1,2\n3,4\n5,6\n")
    rendered = profile_dir(tmp_path)
    assert "a.csv" in rendered and name not in rendered


def test_blank_lines_are_counted_apart_from_data_rows(tmp_path: Path) -> None:
    path = tmp_path / "readings.csv"
    path.write_text("id;value;;\n1;2,5;;\n2;3,0;;\n;;;\n;;;\n")
    profile = profile_file(path)
    assert (profile.rows, profile.blank_rows) == (2, 2)
