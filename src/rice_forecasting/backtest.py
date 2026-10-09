"""Expanding-window, multi-horizon evaluation without realized future features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SEASON, TARGET, YEAR, season_label
from .models import (
    fit_sarimax_forecast,
    lag_features,
    make_random_forest,
    residual_features,
    seasonal_naive_forecast,
)


@dataclass(frozen=True)
class BacktestConfig:
    max_horizon: int = 6
    evaluation_start: int = 80
    origin_step: int = 2
    residual_training_start: int = 50
    minimum_ml_samples: int = 20

    def validate(self, n_observations: int) -> None:
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        if self.origin_step < 1:
            raise ValueError("origin_step must be >= 1")
        if self.evaluation_start < 24:
            raise ValueError("evaluation_start must be >= 24 for stable seasonal SARIMAX fitting")
        if self.evaluation_start >= n_observations - self.max_horizon:
            raise ValueError(
                "Not enough observations for the requested evaluation_start and horizon"
            )
        if self.residual_training_start < 24:
            raise ValueError("residual_training_start must be >= 24")


def _mase_scale(history: np.ndarray, period: int = 2) -> float:
    differences = np.abs(history[period:] - history[:-period])
    scale = float(differences.mean()) if len(differences) else 0.0
    return scale if scale > 1e-12 else 1.0


def _build_oof_residuals(
    frame: pd.DataFrame, values: np.ndarray, config: BacktestConfig
) -> pd.DataFrame:
    """Create historical forecast errors from genuine expanding-window forecasts.

    A row is generated at historical origin t using only observations through t.
    The outer backtest later filters rows to labels already observed at its origin.
    """
    n = len(frame)
    records: list[dict[str, float | int]] = []
    first_origin = config.residual_training_start - 1
    last_origin = n - config.max_horizon - 1
    for origin in range(first_origin, last_origin + 1):
        history = values[: origin + 1]
        base = fit_sarimax_forecast(history, config.max_horizon).point
        for horizon in range(1, config.max_horizon + 1):
            target_idx = origin + horizon
            target_season = str(frame.iloc[target_idx][SEASON])
            row: dict[str, float | int] = {
                "origin_idx": origin,
                "target_idx": target_idx,
                "horizon": horizon,
                "residual": float(values[target_idx] - base[horizon - 1]),
            }
            row.update(
                residual_features(
                    history,
                    horizon,
                    target_is_maha=(target_season == "Maha"),
                    base_forecast=float(base[horizon - 1]),
                )
            )
            records.append(row)
    return pd.DataFrame.from_records(records)


def _build_direct_training_rows(
    frame: pd.DataFrame, values: np.ndarray, max_horizon: int
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for origin in range(3, len(frame) - max_horizon):
        history = values[: origin + 1]
        for horizon in range(1, max_horizon + 1):
            target_idx = origin + horizon
            row: dict[str, float | int] = {
                "origin_idx": origin,
                "target_idx": target_idx,
                "horizon": horizon,
                "target": float(values[target_idx]),
            }
            row.update(
                lag_features(
                    history,
                    horizon,
                    target_is_maha=(str(frame.iloc[target_idx][SEASON]) == "Maha"),
                )
            )
            records.append(row)
    return pd.DataFrame.from_records(records)


def _append_prediction(
    records: list[dict[str, float | int | str]],
    frame: pd.DataFrame,
    values: np.ndarray,
    origin: int,
    horizon: int,
    model: str,
    forecast: float,
    mase_scale: float,
    lower_95: float | None = None,
    upper_95: float | None = None,
) -> None:
    target_idx = origin + horizon
    origin_label = season_label(frame, origin)
    target_label = season_label(frame, target_idx)
    records.append(
        {
            "model": model,
            "origin_idx": origin,
            "origin_year": int(frame.iloc[origin][YEAR]),
            "origin_season": str(frame.iloc[origin][SEASON]),
            "origin_label": origin_label,
            "target_idx": target_idx,
            "target_year": int(frame.iloc[target_idx][YEAR]),
            "target_season": str(frame.iloc[target_idx][SEASON]),
            "target_label": target_label,
            "horizon": horizon,
            "actual": float(values[target_idx]),
            "forecast": float(forecast),
            "error": float(forecast - values[target_idx]),
            "mase_scale": mase_scale,
            "lower_95": lower_95,
            "upper_95": upper_95,
        }
    )


def run_backtest(
    frame: pd.DataFrame, config: BacktestConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run seasonal-naive, SARIMAX, direct RF, and residual-corrected SARIMAX.

    Forecast inputs are limited to past production, the known future season, and
    the model's own forecast. Realized test-period weather/economic/acreage data
    are deliberately not used. RF residual models are trained only on historical
    rolling-origin forecast errors whose outcomes were known at each outer origin.
    """
    config = config or BacktestConfig()
    if TARGET not in frame or SEASON not in frame or YEAR not in frame:
        raise ValueError(f"frame must contain {YEAR!r}, {SEASON!r}, and {TARGET!r}")
    config.validate(len(frame))
    values = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Target values must be finite and strictly positive")

    residual_rows = _build_oof_residuals(frame, values, config)
    direct_rows = _build_direct_training_rows(frame, values, config.max_horizon)
    prediction_records: list[dict[str, float | int | str]] = []

    first_eval_origin = config.evaluation_start - 1
    last_eval_origin = len(frame) - config.max_horizon - 1
    evaluation_origins = list(range(first_eval_origin, last_eval_origin + 1, config.origin_step))
    if evaluation_origins[-1] != last_eval_origin:
        evaluation_origins.append(last_eval_origin)

    for origin in evaluation_origins:
        history = values[: origin + 1]
        mase_scale = _mase_scale(history)
        sarimax = fit_sarimax_forecast(history, config.max_horizon)
        naive = seasonal_naive_forecast(history, config.max_horizon)

        for horizon in range(1, config.max_horizon + 1):
            target_idx = origin + horizon
            target_is_maha = str(frame.iloc[target_idx][SEASON]) == "Maha"
            sarimax_point = float(sarimax.point[horizon - 1])

            _append_prediction(
                prediction_records,
                frame,
                values,
                origin,
                horizon,
                "Seasonal naive",
                float(naive[horizon - 1]),
                mase_scale,
            )
            _append_prediction(
                prediction_records,
                frame,
                values,
                origin,
                horizon,
                "SARIMAX",
                sarimax_point,
                mase_scale,
                float(sarimax.lower_95[horizon - 1]),
                float(sarimax.upper_95[horizon - 1]),
            )

            direct_train = direct_rows[
                (direct_rows["horizon"] == horizon)
                & (direct_rows["target_idx"] <= origin)
                & (direct_rows["origin_idx"] < origin)
            ]
            if len(direct_train) >= config.minimum_ml_samples:
                model = make_random_forest()
                feature_columns = list(lag_features(history, horizon, target_is_maha).keys())
                model.fit(direct_train[feature_columns], direct_train["target"])
                direct_x = pd.DataFrame(
                    [lag_features(history, horizon, target_is_maha)], columns=feature_columns
                )
                direct_prediction = float(model.predict(direct_x)[0])
                _append_prediction(
                    prediction_records,
                    frame,
                    values,
                    origin,
                    horizon,
                    "Direct RF (lags)",
                    direct_prediction,
                    mase_scale,
                )

            residual_train = residual_rows[
                (residual_rows["horizon"] == horizon)
                & (residual_rows["target_idx"] <= origin)
                & (residual_rows["origin_idx"] < origin)
            ]
            if len(residual_train) >= config.minimum_ml_samples:
                feature_columns = [
                    column
                    for column in residual_train.columns
                    if column not in {"origin_idx", "target_idx", "horizon", "residual"}
                ]
                correction_model = make_random_forest()
                correction_model.fit(residual_train[feature_columns], residual_train["residual"])
                current_features = residual_features(
                    history, horizon, target_is_maha, sarimax_point
                )
                correction = float(
                    correction_model.predict(
                        pd.DataFrame([current_features], columns=feature_columns)
                    )[0]
                )
                _append_prediction(
                    prediction_records,
                    frame,
                    values,
                    origin,
                    horizon,
                    "SARIMAX + RF residual",
                    sarimax_point + correction,
                    mase_scale,
                )

    predictions = pd.DataFrame.from_records(prediction_records)
    from .metrics import summarize_predictions

    summary = summarize_predictions(predictions)
    return predictions, summary


def write_outputs(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_file = output_path / "backtest_predictions.csv"
    summary_file = output_path / "backtest_summary.csv"
    predictions.to_csv(prediction_file, index=False, float_format="%.6f")
    summary.to_csv(summary_file, index=False, float_format="%.6f")
    return prediction_file, summary_file
