import numpy as np
import pandas as pd
import pytest

from src import data


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def clean(raw):
    return data.clean(raw)


@pytest.fixture(scope="module")
def dataset():
    return data.load_dataset()


def test_column_names_are_stripped(raw):
    assert "Country" in raw.columns
    assert all(c == c.strip() for c in raw.columns)


def test_summary_rows_are_dropped(raw, clean):
    # The spreadsheet ends with totals/notes rows that have no case number
    assert raw["aCase #"].isna().any()
    assert len(clean) == raw["aCase #"].notna().sum() == 294
    assert clean["deaths"].max() < 10  # totals like 353 would show up here


def test_case_ids_are_unique(clean):
    assert clean["case_id"].is_unique


def test_dash_counts_become_zero(raw, clean):
    raw_cases = raw[raw["aCase #"].notna()].reset_index(drop=True)
    dashes = raw_cases["Tesla occupant"].str.strip() == "-"
    assert dashes.any()
    assert (clean.loc[dashes, "tesla_occupant"] == 0).all()
    for col in data.COUNT_COLUMNS.values():
        assert clean[col].dtype.kind == "i"
        assert (clean[col] >= 0).all()


def test_year_comes_from_date(raw, clean):
    # Raw Year contains a typo (202); the cleaned year is taken from Date
    assert (raw["Year"] == 202).any()
    assert clean["year"].between(2013, 2030).all()


def test_target_encoding(clean):
    assert set(clean[data.TARGET].dropna().unique()) == {0.0, 1.0}
    # Blank targets are unknown, not "no"
    assert clean[data.TARGET].isna().sum() == 18


def test_load_dataset_drops_unknown_target(dataset):
    assert len(dataset) == 276
    assert dataset[data.TARGET].isna().sum() == 0
    assert dataset[data.TARGET].sum() == 35


def test_features_are_complete(dataset):
    assert dataset[data.FEATURES].isna().sum().sum() == 0
    assert set(dataset["region"]) <= {"USA", "Europe", "China", "Other"}
    assert set(dataset["state_group"]) <= {"CA", "FL", "Other US", "Non-US"}
    assert set(dataset["model"]) <= {"S", "3", "X", "Y", "Unknown"}


def test_non_us_rows_have_no_state(dataset):
    assert (dataset.loc[dataset["country"] != "USA", "state_group"] == "Non-US").all()


@pytest.mark.parametrize("leaky", ["autopilot", "verified", "nhtsa", "fsd"])
def test_no_leaky_features(leaky):
    assert not any(leaky in f.lower() for f in data.FEATURES)
    for pattern in data.DESCRIPTION_KEYWORDS.values():
        assert leaky not in pattern.lower()


def test_keyword_flags():
    row = pd.DataFrame([{
        "country": "USA", "state": "CA", "deaths": 2, "description": "Tesla hits pedestrian on highway",
    }])
    out = data.add_features(row).iloc[0]
    assert out["kw_pedestrian"] == 1
    assert out["kw_highway"] == 1
    assert out["kw_fire"] == 0
    assert out["multi_fatality"] == 1


def test_add_features_on_app_style_row():
    """The Streamlit app builds a single row by hand; it must produce every feature."""
    row = pd.DataFrame([{
        "year": 2022, "month": 6, "country": "Germany", "state": "-", "model": "3",
        "description": "", "deaths": 1, "tesla_driver": 1, "tesla_occupant": 0,
        "other_vehicle": 0, "cyclists_peds": 0,
    }])
    features = data.add_features(row)[data.FEATURES]
    assert features.shape == (1, len(data.FEATURES))
    assert features.iloc[0]["region"] == "Europe"
    assert features.iloc[0]["state_group"] == "Non-US"
    assert np.issubdtype(features["deaths"].dtype, np.integer)
