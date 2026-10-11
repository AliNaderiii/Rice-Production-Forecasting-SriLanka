"""Separate operational forecasts from within-season/post-season estimates."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import SEASON, TARGET, YEAR, load_dataset

FEATURES = ["Sown (*000  Acres)", "Harvested (*000  Acres)", "Rainfall(mm)",
            "Temperature(°C)", "GDP B$", "Inflation(%)"]


def evaluate_nowcast(frame: pd.DataFrame, start: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-origin nowcast using same-season realized covariates.

    Inputs are assumed to be available only after the season closes. This is not
    an advance/pre-season forecast. Hyperparameters are fixed, not selected on
    the evaluation segment.
    """
    y = frame[TARGET].to_numpy(float)
    x = frame[FEATURES].astype(float).copy()
    x["season_is_maha"] = (frame[SEASON] == "Maha").astype(float).to_numpy()
    models = {
        "Nowcast Ridge (realized season data)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Nowcast Random Forest (realized season data)": lambda: RandomForestRegressor(
            n_estimators=400, max_depth=4, min_samples_leaf=4, max_features=0.8,
            random_state=42, n_jobs=1),
        "Nowcast HistGradientBoosting (realized season data)": lambda: HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.04, max_leaf_nodes=7, l2_regularization=5.0,
            random_state=42),
    }
    rows = []
    for target_idx in range(start, len(frame)):
        train_x, test_x = x.iloc[:target_idx], x.iloc[[target_idx]]
        for name, factory in models.items():
            model = factory()
            model.fit(train_x, y[:target_idx])
            pred = float(model.predict(test_x)[0])
            rows.append({"scenario": "within_season_or_post_season_nowcast",
                "model": name, "target_idx": target_idx,
                "year": int(frame.iloc[target_idx][YEAR]),
                "season": str(frame.iloc[target_idx][SEASON]),
                "actual": float(y[target_idx]), "forecast": pred,
                "error": pred - float(y[target_idx]),
                "feature_availability": "realized same-season covariates; after season close"})
    predictions = pd.DataFrame(rows)
    summary = []
    for name, g in predictions.groupby("model", sort=False):
        err = g.forecast.to_numpy() - g.actual.to_numpy()
        summary.append({"scenario": "within_season_or_post_season_nowcast", "model": name,
            "n_forecasts": len(g), "mae": float(mean_absolute_error(g.actual, g.forecast)),
            "rmse": float(np.sqrt(mean_squared_error(g.actual, g.forecast))),
            "mape_pct": float(np.mean(np.abs(err / g.actual.to_numpy())) * 100),
            "r2": float(r2_score(g.actual, g.forecast)),
            "evaluation_start_index": start,
            "note": "Exploratory expanding-origin evaluation; same-season actual covariates; not a pre-season forecast."})
    return predictions, pd.DataFrame(summary)


def plot_results(predictions: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "scenario_benchmark_summary.csv", index=False, float_format="%.6f")
    predictions.to_csv(output / "scenario_benchmark_predictions.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    labels = {
        "Nowcast HistGradientBoosting (realized season data)": "Histogram gradient boosting",
        "Nowcast Ridge (realized season data)": "Ridge regression",
        "Nowcast Random Forest (realized season data)": "Random Forest",
    }
    ordered = summary.sort_values("r2")
    colors = {"Nowcast HistGradientBoosting (realized season data)": "#0f766e",
              "Nowcast Ridge (realized season data)": "#2563eb",
              "Nowcast Random Forest (realized season data)": "#d97706"}
    ticklabels = [labels[m] for m in ordered.model]
    axes[0].barh(ticklabels, ordered.r2, color=[colors[m] for m in ordered.model])
    axes[0].axvline(.93, color="#dc2626", linestyle="--", linewidth=1.6, label="Reference target: 0.93")
    axes[0].set(title="Rolling-origin R²", xlabel="R²", xlim=(0, 1))
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].barh(ticklabels, ordered.mape_pct, color=[colors[m] for m in ordered.model])
    axes[1].set(title="Rolling-origin MAPE", xlabel="MAPE (%)")
    for ax in axes: ax.grid(axis="x", alpha=.22)
    fig.suptitle("Nowcast with realized same-season inputs | available only after season close", fontsize=13)
    fig.savefig(output / "scenario_nowcast_comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="rice new one.xlsx")
    parser.add_argument("--output", default="outputs/scenarios")
    parser.add_argument("--evaluation-start", type=int, default=80)
    args = parser.parse_args()
    frame = load_dataset(args.data)
    preds, summary = evaluate_nowcast(frame, args.evaluation_start)
    plot_results(preds, summary, Path(args.output))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
