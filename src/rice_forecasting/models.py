"""Past-only forecasting models and residual-correction helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class SarimaxForecast:
    point: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray


def fit_sarimax_forecast(history: np.ndarray, horizon: int) -> SarimaxForecast:
    """Fit a univariate seasonal SARIMAX and forecast the requested horizon.

    No realized test-period exogenous variables are used. The point estimate is
    lognormal-mean adjusted; confidence limits are transformed from log space.
    """
    values = np.asarray(history, dtype=float)
    if horizon < 1 or len(values) < 24 or np.any(values <= 0):
        raise ValueError(
            "SARIMAX requires positive values, at least 24 observations, and horizon >= 1."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = SARIMAX(
            np.log(values),
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 2),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        forecast = result.get_forecast(steps=horizon)

    mean_log = np.asarray(forecast.predicted_mean, dtype=float)
    variance_log = np.asarray(forecast.var_pred_mean, dtype=float)
    std_log = np.sqrt(np.maximum(variance_log, 0.0))
    point = np.exp(mean_log + 0.5 * variance_log)
    lower = np.exp(mean_log - 1.959963984540054 * std_log)
    upper = np.exp(mean_log + 1.959963984540054 * std_log)
    return SarimaxForecast(point=point, lower_95=lower, upper_95=upper)


def seasonal_naive_forecast(history: np.ndarray, horizon: int, period: int = 2) -> np.ndarray:
    """Repeat the latest observed value for the matching seasonal position."""
    values = list(np.asarray(history, dtype=float))
    if len(values) < period or horizon < 1:
        raise ValueError("Seasonal naive requires at least one full season and a positive horizon.")
    forecasts: list[float] = []
    for h in range(1, horizon + 1):
        source_index = len(values) - period + (h - 1)
        if source_index < len(values):
            forecasts.append(float(values[source_index]))
        else:
            forecasts.append(float(forecasts[source_index - len(values)]))
    return np.asarray(forecasts, dtype=float)


def lag_features(history: np.ndarray, horizon: int, target_is_maha: bool) -> dict[str, float]:
    """Build predictors available at the forecast origin and calendar target."""
    values = np.asarray(history, dtype=float)
    if len(values) < 4:
        raise ValueError("At least four historical observations are required for lag features.")
    recent = values[-4:]
    return {
        "horizon": float(horizon),
        "target_is_maha": float(target_is_maha),
        "last_value": float(values[-1]),
        "lag_2": float(values[-2]),
        "lag_3": float(values[-3]),
        "lag_4": float(values[-4]),
        "change_1": float(values[-1] - values[-2]),
        "change_2": float(values[-1] - values[-3]),
        "recent_mean_4": float(recent.mean()),
        "recent_std_4": float(recent.std(ddof=0)),
    }


def residual_features(
    history: np.ndarray, horizon: int, target_is_maha: bool, base_forecast: float
) -> dict[str, float]:
    features = lag_features(history, horizon, target_is_maha)
    features["sarimax_forecast"] = float(base_forecast)
    features["forecast_minus_last"] = float(base_forecast - history[-1])
    return features


def make_random_forest() -> RandomForestRegressor:
    """Conservative tree model suitable for a small seasonal dataset."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=4,
        max_features=0.8,
        random_state=42,
        n_jobs=1,
    )
