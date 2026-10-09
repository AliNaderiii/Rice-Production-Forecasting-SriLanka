"""Input loading and schema validation."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd

TARGET = "Production (*000  Mt.)"
YEAR = "Year_New"
SEASON = "Season"
SEASON_ORDER = {"Yala": 0, "Maha": 1}
REQUIRED_COLUMNS = {YEAR, SEASON, TARGET}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the workbook and return a validated, chronological seasonal frame.

    The returned index is a zero-based observation index, not a fabricated
    calendar frequency. The supplied workbook ends in Yala 2024 and therefore
    does not contain a complete pair of seasons for its final year.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_excel(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame[YEAR] = pd.to_numeric(frame[YEAR], errors="raise").astype(int)
    frame[SEASON] = frame[SEASON].astype(str).str.strip()
    unknown = sorted(set(frame[SEASON]) - set(SEASON_ORDER))
    if unknown:
        raise ValueError(f"Unexpected season labels: {unknown}")

    frame["_season_order"] = frame[SEASON].map(SEASON_ORDER)
    frame = frame.sort_values([YEAR, "_season_order"], kind="stable").drop(columns="_season_order")
    frame = frame.reset_index(drop=True)

    if frame.duplicated([YEAR, SEASON]).any():
        duplicates = frame.loc[frame.duplicated([YEAR, SEASON], keep=False), [YEAR, SEASON]]
        raise ValueError(f"Duplicate year/season rows found:\n{duplicates.to_string(index=False)}")
    if frame[TARGET].isna().any() or (frame[TARGET] <= 0).any():
        raise ValueError("Production must be complete and strictly positive for log-scale models.")

    # Rows must follow Yala, Maha, Yala, Maha...; a missing season is not silently
    # treated as a regular time step.
    rows = frame[[YEAR, SEASON]].to_dict("records")
    for previous, current in pairwise(rows):
        if previous[SEASON] == "Yala":
            valid_transition = current[SEASON] == "Maha" and current[YEAR] == previous[YEAR]
        else:
            valid_transition = current[SEASON] == "Yala" and current[YEAR] == previous[YEAR] + 1
        if not valid_transition:
            raise ValueError("Season sequence has a gap or unexpected year/season ordering.")

    return frame


def season_label(frame: pd.DataFrame, position: int) -> str:
    """Return a human-readable label for a row position."""
    row = frame.iloc[position]
    year = int(row[YEAR])
    season = str(row[SEASON])
    if season == "Yala":
        return f"Yala {year}"
    return f"Maha {year}/{year + 1}"
