"""Deterministic, bounded CSV profiling."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from tinymeta.models import ColumnProfile, DatasetProfile

_TIMESTAMP = re.compile(
    r"(^|[^a-z0-9])(date|datetime|time|timestamp|year)([^a-z0-9]|$)", re.IGNORECASE
)
_GEOGRAPHIC = re.compile(
    r"(^|[^a-z0-9])(lat|latitude|lon|lng|longitude|country|city|station|location)([^a-z0-9]|$)",
    re.IGNORECASE,
)
_UNITS = (
    (re.compile(r"(?:ug|µg)[_ /]?m3", re.IGNORECASE), "ug/m3"),
    (re.compile(r"mg[_ /]?m3", re.IGNORECASE), "mg/m3"),
    (re.compile(r"(?:deg[_ ]?c|celsius)", re.IGNORECASE), "degC"),
    (re.compile(r"(?:m[_ /]?s|mps)(?:_|$)", re.IGNORECASE), "m/s"),
    (re.compile(r"(?:hpa|mbar)", re.IGNORECASE), "hPa"),
    (re.compile(r"(?:pct|percent|percentage)(?:_|$)", re.IGNORECASE), "%"),
)


class CSVProfileError(ValueError):
    """Raised when a file cannot produce a meaningful CSV profile."""


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _finite(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit(name: str) -> str | None:
    return next((unit for pattern, unit in _UNITS if pattern.search(name)), None)


def profile_csv(path: Path, *, sample_size: int = 3) -> DatasetProfile:
    if path.suffix.lower() != ".csv":
        raise CSVProfileError(f"unsupported file type: {path.suffix or '<none>'}")
    if not path.is_file():
        raise CSVProfileError(f"file does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise CSVProfileError(f"invalid CSV: {exc}") from exc
    if not len(frame.columns):
        raise CSVProfileError("CSV has no columns")

    columns: list[ColumnProfile] = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.dropna()
        numeric = (
            pd.to_numeric(series, errors="coerce")
            if pd.api.types.is_numeric_dtype(series)
            else None
        )
        stats = numeric.dropna() if numeric is not None else None
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                missing_percentage=(
                    round(float(series.isna().mean() * 100), 4) if len(series) else 0.0
                ),
                cardinality=int(non_null.nunique(dropna=True)),
                minimum=_finite(stats.min()) if stats is not None and len(stats) else None,
                maximum=_finite(stats.max()) if stats is not None and len(stats) else None,
                mean=_finite(stats.mean()) if stats is not None and len(stats) else None,
                standard_deviation=_finite(stats.std())
                if stats is not None and len(stats) > 1
                else None,
                samples=[_json_value(value) for value in non_null.head(sample_size).tolist()],
                is_potential_timestamp=bool(_TIMESTAMP.search(str(name))),
                is_potential_geographic=bool(_GEOGRAPHIC.search(str(name))),
                inferred_unit=_unit(str(name)),
            )
        )
    return DatasetProfile(
        filename=path.name,
        file_size_bytes=path.stat().st_size,
        row_count=len(frame),
        column_count=len(columns),
        columns=columns,
    )
