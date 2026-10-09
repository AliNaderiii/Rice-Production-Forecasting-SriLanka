"""Forecast metrics with explicit seasonal scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize scores by model and forecast horizon.

    MASE is scaled using the seasonal-naive in-sample error at each forecast
    origin. MAPE is reported for continuity, but is not the sole selection metric.
    """
    records: list[dict[str, float | int | str]] = []
    for (model, horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        error = forecast - actual
        denom = group["mase_scale"].to_numpy(dtype=float)
        nonzero = actual != 0
        record: dict[str, float | int | str] = {
            "model": str(model),
            "horizon": int(horizon),
            "n_forecasts": len(group),
            "mae": float(mean_absolute_error(actual, forecast)),
            "rmse": float(np.sqrt(mean_squared_error(actual, forecast))),
            "wape_pct": float(100 * np.abs(error).sum() / np.abs(actual).sum()),
            "mape_pct": float(100 * np.abs(error[nonzero] / actual[nonzero]).mean()),
            "mean_error": float(error.mean()),
            "mase": float(np.mean(np.abs(error) / denom)),
            "r2": float(r2_score(actual, forecast)) if len(actual) > 1 else float("nan"),
        }
        if "lower_95" in group and group["lower_95"].notna().any():
            valid = group["lower_95"].notna() & group["upper_95"].notna()
            record["interval_coverage_95_pct"] = float(
                100
                * (
                    (group.loc[valid, "actual"] >= group.loc[valid, "lower_95"])
                    & (group.loc[valid, "actual"] <= group.loc[valid, "upper_95"])
                ).mean()
            )
        records.append(record)
    return pd.DataFrame.from_records(records)
