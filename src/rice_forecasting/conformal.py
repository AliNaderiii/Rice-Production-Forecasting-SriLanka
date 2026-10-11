"""Prequential, horizon-specific conformal intervals for the ETS benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite_sample_quantile(scores: np.ndarray, coverage: float = 0.95) -> float:
    """Split-conformal order statistic with the finite-sample correction."""
    values = np.sort(np.asarray(scores, dtype=float))
    if len(values) == 0:
        return float("nan")
    rank = int(np.ceil((len(values) + 1) * coverage))
    if rank > len(values):
        return float("inf")
    return float(values[rank - 1])


def calibrate(predictions: pd.DataFrame, coverage: float = 0.95,
              min_calibration: int = 8) -> pd.DataFrame:
    """Calibrate each forecast only from earlier errors known at its origin.

    For each horizon, the calibration pool excludes any forecast whose target
    observation would not yet have been available at the current origin.
    """
    data = predictions[predictions.model == "ETS damped trend seasonal"].copy()
    data = data.sort_values(["origin_idx", "horizon"]).reset_index(drop=True)
    lower, upper, n_cal = [], [], []
    for row in data.itertuples(index=False):
        prior = data[(data.horizon == row.horizon)
                     & (data.target_idx <= row.origin_idx)
                     & (data.origin_idx < row.origin_idx)]
        scores = np.abs(prior.error.to_numpy(float))
        n_cal.append(len(scores))
        if len(scores) < min_calibration:
            lower.append(np.nan); upper.append(np.nan)
            continue
        q = finite_sample_quantile(scores, coverage)
        lower.append(max(0.0, float(row.forecast) - q))
        upper.append(float(row.forecast) + q)
    data["lower"] = lower
    data["upper"] = upper
    data["n_calibration"] = n_cal
    data["covered"] = (data.actual >= data.lower) & (data.actual <= data.upper)
    data["interval_width"] = data.upper - data.lower
    return data


def summarize(intervals: pd.DataFrame) -> pd.DataFrame:
    usable = intervals.dropna(subset=["lower", "upper"])
    rows = []
    for h, group in usable.groupby("horizon", sort=True):
        rows.append({"horizon": int(h), "n_calibrated": len(group),
            "empirical_coverage_pct": float(100 * group.covered.mean()),
            "mean_interval_width": float(group.interval_width.mean()),
            "median_calibration_count": float(group.n_calibration.median())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="outputs/advanced/advanced_benchmark_predictions.csv")
    parser.add_argument("--output", default="outputs/advanced")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--min-calibration", type=int, default=10)
    args = parser.parse_args()
    source = pd.read_csv(args.predictions)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    intervals = calibrate(source, args.coverage, args.min_calibration)
    metrics = summarize(intervals)
    intervals.to_csv(output / "ets_conformal_intervals.csv", index=False, float_format="%.6f")
    metrics.to_csv(output / "ets_conformal_summary.csv", index=False, float_format="%.6f")
    if not metrics.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
        axes[0].bar(metrics.horizon, metrics.empirical_coverage_pct, color="#2563eb", width=.66)
        axes[0].axhline(args.coverage * 100, color="#dc2626", linestyle="--", linewidth=1.8,
                        label=f"Nominal {args.coverage:.0%}")
        axes[0].set(title="Empirical interval coverage", xlabel="Horizon (seasons)", ylabel="Coverage (%)")
        axes[0].set_xticks(metrics.horizon); axes[0].set_ylim(0, 105); axes[0].legend(frameon=False)
        axes[1].bar(metrics.horizon, metrics.mean_interval_width, color="#0f766e", width=.66)
        axes[1].set(title="Average interval width", xlabel="Horizon (seasons)",
                    ylabel="Width (thousand metric tonnes)")
        axes[1].set_xticks(metrics.horizon)
        for ax in axes: ax.grid(axis="y", alpha=.22)
        fig.suptitle("ETS forecast uncertainty | prequential conformal calibration", fontsize=14)
        fig.savefig(output / "ets_conformal_diagnostics.png", dpi=220, facecolor="white")
        plt.close(fig)
    print(metrics.to_string(index=False))
    print(f"Saved calibrated predictions and diagnostics to {output}")


if __name__ == "__main__":
    main()
