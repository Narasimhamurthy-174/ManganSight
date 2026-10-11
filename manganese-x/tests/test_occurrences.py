"""Tests use tiny FABRICATED TOY tables to test code logic only. They are not real data."""
import pandas as pd
import pytest

from src.data.occurrences import (STANDARD_COLUMNS, clean_occurrences,
                                  standardize_mrds, usable_positives, validate_coordinates)

B = {"lat_min": 21.0, "lat_max": 22.7, "lon_min": 79.5, "lon_max": 81.3}


def toy(rows):
    return pd.DataFrame(rows, columns=["latitude", "longitude"])


def test_valid_point_passes():
    r = validate_coordinates(toy([(21.8, 80.2)]), B)
    assert r.coord_valid.all()


def test_swapped_lat_lon_flagged():
    r = validate_coordinates(toy([(80.2, 21.8)]), B)
    assert not r.coord_valid.iloc[0]


def test_impossible_and_missing_flagged():
    r = validate_coordinates(toy([(95.0, 80.0), (None, 80.0), ("abc", 80.0)]), B)
    assert list(r.coord_issue) == ["impossible_coordinates", "missing_or_non_numeric",
                                   "missing_or_non_numeric"]


def test_outside_bounds_flagged():
    r = validate_coordinates(toy([(28.6, 77.2)]), B)
    assert r.coord_issue.iloc[0] == "outside_study_area_bounds"


def test_empty_dataset_does_not_crash():
    out = clean_occurrences(pd.DataFrame(columns=STANDARD_COLUMNS), B)
    assert out.empty


def test_duplicates_flagged_and_excluded_from_positives():
    df = pd.DataFrame({"latitude": [21.80001, 21.80002, 21.9], "longitude": [80.2, 80.2, 80.3]})
    out = clean_occurrences(df, B)
    assert out.is_duplicate.tolist() == [False, True, False]
    assert len(usable_positives(out)) == 2


def test_standardize_mrds_filters_manganese_and_keeps_unknown_as_na():
    mrds = pd.DataFrame({
        "DEP_ID": [1, 2], "LATITUDE": [21.8, 21.9], "LONGITUDE": [80.2, 80.3],
        "COMMOD1": ["Manganese", "Copper"], "COMMOD2": [None, None]})
    out = standardize_mrds(mrds)
    assert len(out) == 1
    assert out.loc[0, "source"] == "USGS MRDS"
    assert pd.isna(out.loc[0, "host_rock"])          # never invented
    assert list(out.columns) == STANDARD_COLUMNS


def test_standardize_mrds_rejects_wrong_table():
    with pytest.raises(ValueError):
        standardize_mrds(pd.DataFrame({"a": [1]}))
