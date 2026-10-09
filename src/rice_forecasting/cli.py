"""Command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path

from . import __version__
from .backtest import BacktestConfig, run_backtest, write_outputs
from .data import load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a leakage-aware seasonal rice forecast backtest."
    )
    parser.add_argument(
        "--data", default="rice new one.xlsx", help="Path to the source Excel workbook"
    )
    parser.add_argument("--output", default="outputs", help="Directory for CSV results")
    parser.add_argument(
        "--max-horizon", type=int, default=6, help="Maximum horizon in seasonal observations"
    )
    parser.add_argument(
        "--evaluation-start", type=int, default=80, help="First zero-based training endpoint + 1"
    )
    parser.add_argument("--origin-step", type=int, default=2, help="Evaluate every N observations")
    parser.add_argument("--residual-training-start", type=int, default=50)
    parser.add_argument("--minimum-ml-samples", type=int, default=20)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    frame = load_dataset(args.data)
    config = BacktestConfig(
        max_horizon=args.max_horizon,
        evaluation_start=args.evaluation_start,
        origin_step=args.origin_step,
        residual_training_start=args.residual_training_start,
        minimum_ml_samples=args.minimum_ml_samples,
    )
    predictions, summary = run_backtest(frame, config)
    prediction_file, summary_file = write_outputs(predictions, summary, args.output)
    print(
        f"Validated {len(frame)} seasonal observations: {frame.iloc[0]['Year_New']} {frame.iloc[0]['Season']}–"
        f"{frame.iloc[-1]['Year_New']} {frame.iloc[-1]['Season']}"
    )
    print(f"Forecast rows: {len(predictions)} | origins: {predictions['origin_idx'].nunique()}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved: {prediction_file}")
    print(f"Saved: {summary_file}")
    dataset_path = Path(args.data)
    package_versions = {}
    for package_name in ("numpy", "pandas", "scikit-learn", "statsmodels", "openpyxl"):
        package_versions[package_name] = importlib.metadata.version(package_name)
    metadata = {
        "pipeline_version": __version__,
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "n_observations": len(frame),
        "target": "Rice production (thousand metric tonnes)",
        "config": vars(config),
        "prediction_file": str(prediction_file),
        "summary_file": str(summary_file),
        "models": sorted(predictions["model"].unique().tolist()),
        "note": "Historical realized exogenous features are excluded from forecast predictors.",
    }
    Path(args.output, "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
