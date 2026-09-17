"""Dataset profiler (T5): what the Planner sees about the data, beyond the data dictionary.

Files are read as raw text first, with every value kept as a string, because the problems that matter hide in
exactly what a default `pd.read_csv` would silently decide: the separator, whether the first line is a header
(or two lines are), decimal commas, leading spaces, placeholder strings, sentinel numbers and the order of day
and month in dates. Each finding is a plain sentence, so it can go straight into a prompt.

Trusted code: it only reads text. It never imports or unpickles anything an agent wrote.
"""

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_ROWS = 50_000
PLACEHOLDERS = {"?", "unknown", "na", "n/a", "null", "none", "nan", "-", "missing", "not available"}
SENTINEL_CANDIDATES = {-99999, -9999, -999, -200, -99, -9, 99, 999, 9999, 99999}
IDENTIFIER_NAME = re.compile(r"(^|_)(id|nbr|num|number|no|key)$|id$", re.IGNORECASE)
NUMBER = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
DECIMAL_COMMA = re.compile(r"^[-+]?\d+,\d+$")
DATE_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})")
DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")
DATA_SUFFIXES = {".csv", ".data", ".txt", ".tsv"}


@dataclass
class ColumnProfile:
    name: str
    kind: str  # numeric, text, date, empty
    unique: int
    notes: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class FileProfile:
    path: str
    rows: int
    separator: str
    header_rows: int
    columns: list[ColumnProfile]
    notes: list[str]
    blank_rows: int = 0  # lines with no values at all, not counted in `rows`

    def render(self) -> str:
        size = f"{self.rows:,} data rows, {len(self.columns)} columns"
        head = f"## {self.path}: {size}, separator {self.separator!r}"
        lines = [head, *[f"- {n}" for n in self.notes]]
        for c in self.columns:
            flags = f" | {'; '.join(c.notes)}" if c.notes else ""
            lines.append(f"  - {c.name}: {c.kind}, {c.unique:,} distinct, {c.summary}{flags}")
        return "\n".join(lines)


def _is_number(value: str) -> bool:
    return bool(NUMBER.match(value.strip()))


def _sniff_separator(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _header_rows(raw: pd.DataFrame) -> int:
    """0, 1 or 2. A header row has mostly non-numeric cells while the data below it is mostly numeric."""
    if len(raw) < 3:
        return 1
    numeric = raw.iloc[:200].apply(lambda col: col.map(_is_number))
    share = [float(numeric.iloc[i].mean()) for i in range(3)]
    data_share = float(numeric.iloc[2:].mean().mean())
    # A header line is (almost) all text. A data line with a few text codes (e.g. 'V57', 'C536379') is not.
    looks_like_names = [s <= 0.1 and data_share - s > 0.3 for s in share[:2]]
    if looks_like_names[0] and looks_like_names[1] and data_share > 0.3:
        return 2
    if looks_like_names[0]:
        return 1
    # Mostly text data: a header row's cells do not occur again in their columns.
    first = raw.iloc[0].str.strip()
    body = raw.iloc[1:200]
    unseen = np.mean([first.iloc[j] not in set(body.iloc[:, j].str.strip()) for j in range(raw.shape[1])])
    return 1 if unseen > 0.9 and share[0] < 0.5 else 0


def _date_note(values: pd.Series) -> str | None:
    distinct = values.drop_duplicates()
    sample = distinct.sample(min(len(distinct), 5000), random_state=0)  # spread over the file, not its start
    if sample.str.match(DATE_ISO).mean() > 0.9:
        return "ISO dates (YYYY-MM-DD)"
    parts = sample.str.extract(DATE_SLASH).dropna().astype(int)
    if len(parts) < 0.9 * len(sample):
        return None
    first_big, second_big = (parts[0] > 12).any(), (parts[1] > 12).any()
    if first_big and not second_big:
        return "dates with a slash, DAY first (a first field above 12 occurs)"
    if second_big and not first_big:
        return "dates with a slash, MONTH first (a second field above 12 occurs)"
    return "dates with a slash; the day/month order cannot be proven from the values"


def _sentinel_notes(numbers: pd.Series) -> list[str]:
    """An extreme value that is common and sits far from a spread of other values: a missing-value code?"""
    counts = numbers.value_counts()
    if len(counts) <= 2:
        return []  # a binary column has two values, not a sentinel
    notes = []
    for value in dict.fromkeys((numbers.min(), numbers.max())):
        share = counts.get(value, 0) / len(numbers)
        others = numbers[numbers != value]
        if share < 0.005 or others.nunique() < 3:
            continue
        gap = min(abs(value - others.min()), abs(value - others.max()))
        spread = max(others.quantile(0.95) - others.quantile(0.05), 1e-9)
        if (value in SENTINEL_CANDIDATES and gap > 0.5 * spread) or gap > 5 * spread:
            notes.append(
                f"{value:g} appears in {share:.1%} of rows, far from the other values "
                f"({others.min():g} to {others.max():g}): a missing-value code?"
            )
    return notes


def _profile_column(name: str, values: pd.Series) -> ColumnProfile:
    trimmed = values.str.strip()
    filled = trimmed[trimmed != ""]
    notes: list[str] = []
    if len(filled) < len(trimmed):
        notes.append(f"{1 - len(filled) / len(trimmed):.1%} empty")
    if ((values != trimmed) & (trimmed != "")).mean() > 0.5:
        notes.append("values carry leading/trailing spaces")
    placeholder = filled.str.lower().isin(PLACEHOLDERS)
    if placeholder.any():
        counts = filled[placeholder].value_counts()
        notes.append(
            "placeholder values " + ", ".join(f"{v!r} ({n / len(trimmed):.1%})" for v, n in counts.items())
        )
    real = filled[~placeholder]
    if len(real) == 0:
        return ColumnProfile(name, "empty", 0, notes, "no values")

    if date := _date_note(real):
        return ColumnProfile(name, "date", int(real.nunique()), [*notes, date], f"e.g. {real.iloc[0]!r}")

    comma = real.str.match(DECIMAL_COMMA)
    if comma.mean() > 0.3 and (comma | real.map(_is_number)).mean() > 0.95:
        notes.append(f"numbers written with a DECIMAL COMMA (e.g. {real[comma].iloc[0]!r})")
        real = real.str.replace(",", ".", regex=False)

    is_number = real.map(_is_number)
    if is_number.mean() > 0.95:
        if not is_number.all():
            examples = ", ".join(repr(v) for v in real[~is_number].drop_duplicates().head(3))
            notes.append(
                f"{1 - is_number.mean():.1%} of values are not numbers (e.g. {examples}): codes, not "
                f"quantities?"
            )
        numbers = pd.to_numeric(real[is_number])
        unique = int(numbers.nunique())
        is_int = bool((numbers == numbers.round()).all())
        if unique == len(numbers) and is_int and len(numbers) > 20:
            notes.append("unique per row: an identifier?")
        elif is_int and IDENTIFIER_NAME.search(name) and unique <= 50:
            notes.append(f"{unique} integer codes named like an identifier: look up what each code means")
        elif is_int and IDENTIFIER_NAME.search(name) and unique < len(numbers):
            notes.append(f"an identifier that repeats: {len(numbers) / unique:.2f} rows per value on average")
        elif is_int and 2 < unique <= 12:
            values_list = sorted(numbers.unique().astype(int).tolist())
            notes.append(
                f"integer with {unique} distinct values {values_list}: possibly codes rather than quantities"
            )
        notes += _sentinel_notes(numbers)
        summary = f"min {numbers.min():g}, median {numbers.median():g}, max {numbers.max():g}"
        return ColumnProfile(name, "numeric", unique, notes, summary)

    top = real.value_counts().head(4)
    summary = "top " + ", ".join(f"{v!r} ({n / len(trimmed):.0%})" for v, n in top.items())
    if real.nunique() == len(real) and len(real) > 20:
        notes.append("unique per row: an identifier?")
    if 0.3 < is_number.mean() <= 0.95:
        notes.append(f"{is_number.mean():.0%} of values are numbers, the rest are text")
    return ColumnProfile(name, "text", int(real.nunique()), notes, summary)


def profile_file(path: Path, display_name: str | None = None) -> FileProfile:
    raw_bytes = path.read_bytes()
    notes: list[str] = []
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
        notes.append("not valid UTF-8 (decoded as Latin-1)")
    if "\r\n" in text[:10_000]:
        notes.append("Windows line endings (CRLF)")
    sep = _sniff_separator("\n".join(text.splitlines()[:50]))
    raw = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        header=None,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=False,
        engine="python",
        on_bad_lines="skip",
    ).fillna("")
    blank = raw.apply(lambda r: (r.str.strip() == "").all(), axis=1)
    if blank.any():
        notes.append(f"{int(blank.sum())} rows with no values at all")
    raw = raw[~blank]
    empty_cols = [c for c in raw.columns if (raw[c].str.strip() == "").all()]
    if empty_cols:
        notes.append(f"{len(empty_cols)} trailing column(s) with no values (the lines end with {sep!r})")
        raw = raw.drop(columns=empty_cols)

    headers = _header_rows(raw)
    if headers == 0:
        notes.append("NO HEADER ROW: the first line is data; column names must come from the documentation")
        names = [f"column_{i + 1}" for i in range(raw.shape[1])]
    else:
        if headers == 2:
            notes.append(
                f"TWO HEADER ROWS: line 1 {list(raw.iloc[0].head(4))}..., line 2 "
                f"{list(raw.iloc[1].head(4))}...; "
                "the second line holds the names"
            )
        names = [v.strip() or f"column_{i + 1}" for i, v in enumerate(raw.iloc[headers - 1])]
    body = raw.iloc[headers:]
    rows = len(body)
    if headers:
        header = raw.iloc[headers - 1].str.strip().to_numpy()
        matches = (body.apply(lambda col: col.str.strip()).to_numpy() == header).mean(axis=1)
        repeats = int((matches >= 0.5).sum())
        if repeats:
            notes.append(
                f"{repeats} line(s) inside the file look like the header again: several tables stacked in "
                f"one file?"
            )
    duplicates = int(body.duplicated().sum())
    if duplicates:
        notes.append(f"{duplicates:,} exact duplicate rows")
    if rows > SAMPLE_ROWS:
        body = body.iloc[:: -(-rows // SAMPLE_ROWS)]
        notes.append(f"column statistics are from {len(body):,} rows spread evenly over the file")
    columns = [_profile_column(names[i], body.iloc[:, i]) for i in range(body.shape[1])]
    return FileProfile(display_name or path.name, rows, sep, headers, columns, notes, int(blank.sum()))


def profile_dir(data_dir: Path, max_files: int = 10) -> str:
    """A text profile of every data file under data_dir (README files are documentation, not data)."""
    files = [
        p
        for p in sorted(data_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in DATA_SUFFIXES and not p.name.lower().startswith("readme")
    ]
    return "\n\n".join(profile_file(p, str(p.relative_to(data_dir))).render() for p in files[:max_files])
