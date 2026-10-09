from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rice_forecasting import backtest
from rice_forecasting.backtest import BacktestConfig, run_backtest


def test_config_rejects_impossible_evaluation_window() -> None:
    frame = pd.DataFrame(
        {
            "Year_New": range(30),
            "Season": ["Yala", "Maha"] * 15,
            "Production (*000  Mt.)": np.arange(30) + 100.0,
        }
    )
    with pytest.raises(ValueError, match="Not enough observations"):
        run_backtest(frame, BacktestConfig(max_horizon=6, evaluation_start=29))


def test_backtest_smoke_with_stubbed_sarimax(monkeypatch) -> None:
    def fake_sarimax(history, horizon):
        point = np.repeat(float(history[-1]), horizon)
        return SimpleNamespace(point=point, lower_95=point * 0.8, upper_95=point * 1.2)

    monkeypatch.setattr(backtest, "fit_sarimax_forecast", fake_sarimax)
    n = 60
    frame = pd.DataFrame(
        {
            "Year_New": np.repeat(np.arange(1990, 2020), 2)[:n],
            "Season": ["Yala", "Maha"] * (n // 2),
            "Production (*000  Mt.)": 500
            + 2 * np.arange(n)
            + 15 * np.sin(np.arange(n) * np.pi / 2),
        }
    )
    predictions, summary = run_backtest(
        frame,
        BacktestConfig(
            max_horizon=2,
            evaluation_start=40,
            origin_step=4,
            residual_training_start=30,
            minimum_ml_samples=2,
        ),
    )
    assert {"Seasonal naive", "SARIMAX", "Direct RF (lags)", "SARIMAX + RF residual"}.issubset(
        set(predictions["model"])
    )
    assert set(predictions["horizon"]) == {1, 2}
    assert not summary.empty
    assert predictions["target_idx"].gt(predictions["origin_idx"]).all()
