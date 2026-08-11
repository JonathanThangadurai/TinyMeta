from pathlib import Path

import pytest

from tinymeta.profiling import CSVProfileError, profile_csv


def test_profile_csv_is_deterministic_and_handles_missing(csv_path: Path) -> None:
    first = profile_csv(csv_path)
    second = profile_csv(csv_path)
    assert first == second
    assert (first.row_count, first.column_count) == (2, 3)
    no2 = first.columns[1]
    assert no2.missing_percentage == 50.0
    assert no2.cardinality == 1
    assert no2.mean == 10.0
    assert no2.standard_deviation is None
    assert no2.inferred_unit == "ug/m3"
    assert first.columns[0].is_potential_timestamp


def test_empty_csv_with_headers_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("NO2_ug_m3,station_id\n")
    profile = profile_csv(path)
    assert profile.row_count == 0
    assert profile.columns[0].missing_percentage == 0.0


@pytest.mark.parametrize("name", ["bad.json", "no_extension"])
def test_unsupported_type(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("x\n1\n")
    with pytest.raises(CSVProfileError, match="unsupported"):
        profile_csv(path)


def test_malformed_or_contentless_csv(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(CSVProfileError, match="invalid CSV"):
        profile_csv(path)


def test_unusual_column_names(tmp_path: Path) -> None:
    path = tmp_path / "unicode.csv"
    path.write_text("PM2.5 (µg/m3),Latitude [deg]\n12.4,52.3\n")
    profile = profile_csv(path)
    assert profile.columns[0].inferred_unit == "ug/m3"
    assert profile.columns[1].is_potential_geographic
