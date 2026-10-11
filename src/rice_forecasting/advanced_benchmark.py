"""Leakage-aware small-sample benchmark for additional classical/ML models."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from rice_forecasting.data import SEASON, TARGET, YEAR, load_dataset, season_label
from rice_forecasting.metrics import summarize_predictions


def features(values: np.ndarray, horizon: int, target_season: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    result = {f"lag_{i}": float(x[-i]) for i in range(1, min(8, len(x)) + 1)}
    result.update({
        "horizon": float(horizon),
        "target_is_maha": float(target_season == "Maha"),
        "mean_2": float(np.mean(x[-2:])),
        "mean_4": float(np.mean(x[-4:])),
        "mean_8": float(np.mean(x[-8:])),
        "std_4": float(np.std(x[-4:])),
        "diff_1": float(x[-1] - x[-2]),
        "diff_2": float(x[-1] - x[-3]),
    })
    return result


def mase_scale(x: np.ndarray, period: int = 2) -> float:
    return max(float(np.mean(np.abs(x[period:] - x[:-period]))), 1e-12)


def run(frame: pd.DataFrame, start: int = 80, step: int = 2, max_h: int = 6) -> pd.DataFrame:
    y = frame[TARGET].to_numpy(float)
    final_origin = len(frame) - max_h - 1
    origins = list(range(start - 1, final_origin + 1, step))
    if origins[-1] != final_origin:
        origins.append(final_origin)
    out: list[dict] = []
    for origin in origins:
        hist = y[:origin + 1]
        log_hist = np.log(hist)
        try:
            ets = ExponentialSmoothing(log_hist, trend="add", damped_trend=True,
                seasonal="add", seasonal_periods=2, initialization_method="estimated").fit(optimized=True)
            ets_fc = np.exp(np.asarray(ets.forecast(max_h), dtype=float))
        except (ValueError, np.linalg.LinAlgError):
            ets_fc = np.repeat(hist[-1], max_h)
        try:
            theta = ThetaModel(log_hist, period=2, deseasonalize=True, use_test=False).fit()
            theta_fc = np.exp(np.asarray(theta.forecast(max_h), dtype=float))
        except (ValueError, np.linalg.LinAlgError):
            theta_fc = np.repeat(hist[-1], max_h)

        for h in range(1, max_h + 1):
            target_idx = origin + h
            target_season = str(frame.iloc[target_idx][SEASON])
            train_x, train_y = [], []
            # Each supervised row is an historical forecast origin; label must
            # already be observed at the current outer origin.
            for past_origin in range(7, origin):
                past_target = past_origin + h
                if past_target > origin:
                    continue
                train_x.append(features(y[:past_origin + 1], h, str(frame.iloc[past_target][SEASON])))
                train_y.append(np.log(y[past_target]))
            if len(train_x) >= 18:
                columns = list(train_x[0])
                ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                ridge.fit(pd.DataFrame(train_x)[columns], train_y)
                pred_ridge = float(np.exp(ridge.predict(pd.DataFrame([features(hist, h, target_season)], columns=columns))[0]))
            else:
                pred_ridge = float(hist[-1])
            forecasts = {
                "ETS damped trend seasonal": float(ets_fc[h - 1]),
                "Theta seasonal": float(theta_fc[h - 1]),
                "Ridge direct log": pred_ridge,
            }
            naive = float(hist[-2] if h % 2 == 1 else hist[-1])
            forecasts["Seasonal naive"] = naive
            for name, forecast in forecasts.items():
                out.append({
                    "model": name, "origin_idx": origin,
                    "origin_year": int(frame.iloc[origin][YEAR]),
                    "origin_season": str(frame.iloc[origin][SEASON]),
                    "origin_label": season_label(frame, origin),
                    "target_idx": target_idx,
                    "target_year": int(frame.iloc[target_idx][YEAR]),
                    "target_season": target_season,
                    "target_label": season_label(frame, target_idx),
                    "horizon": h, "actual": float(y[target_idx]),
                    "forecast": forecast, "error": forecast - float(y[target_idx]),
                    "mase_scale": mase_scale(hist), "lower_95": np.nan, "upper_95": np.nan,
                })
    return pd.DataFrame(out)


def create_figures(all_predictions: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_predictions(all_predictions)
    summary.to_csv(output / "advanced_benchmark_summary.csv", index=False, float_format="%.6f")
    all_predictions.to_csv(output / "advanced_benchmark_predictions.csv", index=False, float_format="%.6f")
    selected = summary[summary.model.isin(["Seasonal naive", "SARIMAX", "SARIMAX + RF residual",
        "ETS damped trend seasonal", "Theta seasonal", "Ridge direct log", "Direct RF (lags)"])].copy()
    order = ["Seasonal naive", "SARIMAX", "Direct RF (lags)", "SARIMAX + RF residual",
        "ETS damped trend seasonal", "Theta seasonal", "Ridge direct log"]
    colors = {m: c for m, c in zip(order, ["#94a3b8", "#2563eb", "#f97316", "#16a34a", "#8b5cf6", "#0891b2", "#db2777"])}
    fig, ax = plt.subplots(figsize=(11, 6.2), layout="constrained")
    for model in order:
        rows = selected[selected.model == model].sort_values("horizon")
        if rows.empty: continue
        ax.plot(rows.horizon, rows.mae, marker="o", linewidth=2.2, markersize=5,
                label=model, color=colors[model])
    ax.set(title="Rice production forecasting | Rolling-origin MAE by horizon",
           xlabel="Forecast horizon (seasonal observations)", ylabel="MAE (thousand metric tonnes)")
    ax.set_xticks(range(1, 7)); ax.grid(True, alpha=.25); ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.savefig(output / "mae_by_horizon.png", dpi=220, facecolor="white"); plt.close(fig)

    overall = selected.groupby("model", as_index=False).agg(mean_horizon_mae=("mae", "mean"))
    overall["model"] = pd.Categorical(overall.model, categories=order, ordered=True)
    overall = overall.sort_values("mean_horizon_mae", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    ax.barh(overall.model.astype(str), overall.mean_horizon_mae,
            color=[colors[str(x)] for x in overall.model])
    ax.set(title="Average MAE across forecast horizons (descriptive; not a pooled test)",
           xlabel="Mean of horizon-specific MAE values (thousand metric tonnes)")
    ax.grid(axis="x", alpha=.25); ax.invert_yaxis()
    fig.savefig(output / "mean_mae_models.png", dpi=220, facecolor="white"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="rice new one.xlsx")
    parser.add_argument("--reference-predictions", default="outputs/backtest_predictions.csv")
    parser.add_argument("--output", default="outputs/advanced")
    parser.add_argument("--evaluation-start", type=int, default=80)
    parser.add_argument("--origin-step", type=int, default=2)
    parser.add_argument("--max-horizon", type=int, default=6)
    args = parser.parse_args()
    frame = load_dataset(args.data)
    candidates = run(frame, args.evaluation_start, args.origin_step, args.max_horizon)
    reference_path = Path(args.reference_predictions)
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        combined = pd.concat([reference, candidates], ignore_index=True)
    else:
        combined = candidates
    create_figures(combined, Path(args.output))
    print(f"Saved rolling-origin results and figures to {args.output}")
    print(summarize_predictions(candidates).to_string(index=False))


if __name__ == "__main__":
    main()
