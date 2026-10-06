"""Clean the Model column of the raw Tesla deaths CSV and save the result.

Drops the totals/notes rows at the bottom of the spreadsheet, then keeps
only valid Tesla models (3, S, X, Y, Roadster); everything else ("-",
blanks, stray numbers) becomes "Unknown". All other columns are written
back unchanged.

Run from the project root:
    python scripts/clean_model_column.py
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "Tesla Deaths - Deaths.csv"
OUT_PATH = PROJECT_ROOT / "data" / "tesla_clean.csv"

VALID_MODELS = {"3": "3", "S": "S", "X": "X", "Y": "Y", "ROADSTER": "Roadster"}


def clean_model(value) -> str:
    if pd.isna(value):
        return "Unknown"
    # Accept "S", " s ", "Model S", "Roadster" etc.
    key = str(value).strip().upper().removeprefix("MODEL").strip()
    return VALID_MODELS.get(key, "Unknown")


def main():
    # Read everything as text so other columns are saved exactly as they were
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False, na_values=[""])
    model_col = next(c for c in df.columns if c.strip() == "Model")  # raw header is " Model "

    # Rows without a case number are the spreadsheet's totals and notes, not crashes
    totals = df["aCase #"].isna()
    df = df[~totals].reset_index(drop=True)
    print(f"Dropped {totals.sum()} totals/notes rows")

    before = df[model_col].str.strip().fillna("<blank>")
    df[model_col] = df[model_col].map(clean_model)

    changed = before[before != df[model_col]]
    print("Replaced with Unknown:", changed.value_counts().to_dict())
    print("Model counts after cleaning:", df[model_col].value_counts().to_dict())

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
