from pathlib import Path

import pytest

from rice_forecasting.data import TARGET, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_workbook_is_validated_and_ordered() -> None:
    frame = load_dataset(ROOT / "rice new one.xlsx")
    assert len(frame) == 149
    assert frame.loc[0, "Season"] == "Yala"
    assert frame.loc[1, "Season"] == "Maha"
    assert frame.loc[len(frame) - 1, "Season"] == "Yala"
    assert frame[TARGET].gt(0).all()


def test_missing_workbook_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(ROOT / "does-not-exist.xlsx")
