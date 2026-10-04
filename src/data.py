"""Data loading, cleaning and feature engineering for the Tesla fatal crash dataset.

Shared by the notebook, the training script and the Streamlit app so every stage
sees exactly the same features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "Tesla Deaths - Deaths.csv"

TARGET = "autopilot_claimed"

# Raw count columns -> clean snake_case names
COUNT_COLUMNS = {
    "Deaths": "deaths",
    "Tesla driver": "tesla_driver",
    "Tesla occupant": "tesla_occupant",
    "Other vehicle": "other_vehicle",
    "Cyclists/ Peds": "cyclists_peds",
}

EUROPE = {
    "Germany", "Netherlands", "UK", "Norway", "Switzerland", "France", "Denmark",
    "Belgium", "Finland", "Sweden", "Austria", "Spain", "Italy", "Portugal",
    "Ireland", "Slovenia", "Czech Republic", "Poland",
}

# Keyword flags pulled from the free-text description. Words that mention
# Autopilot/FSD are deliberately excluded: they would leak the target.
DESCRIPTION_KEYWORDS = {
    "kw_pedestrian": r"pedestrian|walking|crossing|wheelchair|standing",
    "kw_motorcycle": r"motorcycl|motorcyclist|bike|cyclist",
    "kw_fire": r"fire|burn|flame|ignite|fiery",
    "kw_truck": r"truck|semi|tractor|18 wheeler|bus|dump",
    "kw_fixed_object": r"tree|wall|pole|barrier|building|storefront|house|garage",
    "kw_highway": r"highway|freeway|interstate|ramp",
}

NUMERIC_FEATURES = [
    "year", "month", "deaths", "tesla_driver", "tesla_occupant",
    "other_vehicle", "cyclists_peds", "multi_fatality",
    *DESCRIPTION_KEYWORDS,
]
CATEGORICAL_FEATURES = ["region", "state_group", "model"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODELS = ["S", "3", "X", "Y"]


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one tidy row per crash, with the target encoded as 0/1 (NaN = unknown)."""
    # Rows without a case number are the spreadsheet's totals/notes at the bottom
    df = raw[raw["aCase #"].notna()].copy()

    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())

    out = pd.DataFrame(index=df.index)
    out["case_id"] = df["aCase #"].astype(int)

    # "Year" has typos (e.g. 202) and disagrees with Date for some cases, so trust Date
    date = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    out["date"] = date
    out["year"] = date.dt.year.fillna(df["Year"]).astype(int)
    out["month"] = date.dt.month.fillna(6).astype(int)

    out["country"] = df["Country"].replace({"Holland": "Netherlands"}).fillna("Unknown")
    out["state"] = df["State"].where(out["country"] == "USA", "-").fillna("-")
    out["description"] = df["Description"].fillna("")
    out["model"] = df["Model"].where(df["Model"].isin(MODELS), "Unknown")

    # "-" means none/zero in the count columns
    for raw_col, col in COUNT_COLUMNS.items():
        out[col] = pd.to_numeric(df[raw_col].replace("-", "0"), errors="coerce").fillna(0).astype(int)

    # Target: "-" = not claimed, any positive count = claimed, blank = unknown
    ap = pd.to_numeric(df["Autopilot claimed"].replace("-", "0"), errors="coerce")
    out[TARGET] = np.where(ap.isna(), np.nan, (ap > 0).astype(float))

    return out.reset_index(drop=True)


def region_of(country: str) -> str:
    if country == "USA":
        return "USA"
    if country == "China":
        return "China"
    if country in EUROPE:
        return "Europe"
    return "Other"


def state_group_of(country: str, state: str, top_states=("CA", "FL")) -> str:
    if country != "USA":
        return "Non-US"
    return state if state in top_states else "Other US"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready features. Works on cleaned data or a single app input row."""
    df = df.copy()
    df["region"] = df["country"].map(region_of)
    df["state_group"] = [state_group_of(c, s) for c, s in zip(df["country"], df["state"])]
    df["multi_fatality"] = (df["deaths"] > 1).astype(int)

    desc = df["description"].str.lower()
    for col, pattern in DESCRIPTION_KEYWORDS.items():
        df[col] = desc.str.contains(pattern, regex=True).astype(int)
    return df


def load_dataset(path=RAW_PATH, drop_unknown_target=True) -> pd.DataFrame:
    """Raw CSV -> cleaned, feature-engineered frame ready for modelling."""
    df = add_features(clean(load_raw(path)))
    if drop_unknown_target:
        df = df[df[TARGET].notna()].copy()
        df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)
