# Sri Lanka Rice Production Forecasting

An auditable, leakage-aware benchmark for seasonal rice production in Sri Lanka. The current reference pipeline compares seasonal naive, univariate SARIMAX, direct Random Forest, and SARIMAX with rolling-origin residual correction.

> **Results note:** Earlier repository and LinkedIn materials reported a single hybrid score (`R² = 0.9254`, `MAPE = 5.12%`). Those figures are not reproduced by the original saved modeling notebook. They are not used as current results here. The current benchmark below is generated from the checked-in code and workbook. It does **not** establish that the hybrid is universally best or that its residual correction identifies climate shocks.

## Current benchmark

Run configuration: 149 seasonal observations; expanding training window with at least 80 observations; 33 forecast origins; horizons 1–6 seasons; origins advance by two observations, with the last eligible origin included. All models use the same origins. Predictors are limited to past production and the known future season; realized future weather, acreage, and economic values are excluded.

The table reports **MAE (WAPE)** for each horizon. MAE is in the workbook's target units: thousand metric tonnes. Lower is better.

| Horizon | Seasonal naive | SARIMAX | Direct RF (lags) | SARIMAX + RF residual |
|---:|---:|---:|---:|---:|
| 1 | 303.4 (23.90%) | 252.3 (19.87%) | 246.4 (19.41%) | **245.4 (19.33%)** |
| 2 | 335.8 (16.11%) | 333.7 (16.01%) | 390.0 (18.72%) | **328.8 (15.78%)** |
| 3 | 264.5 (20.19%) | 254.0 (19.39%) | **234.0 (17.87%)** | 253.7 (19.37%) |
| 4 | 376.7 (17.69%) | 402.7 (18.91%) | 374.9 (17.61%) | **370.1 (17.38%)** |
| 5 | 260.2 (19.42%) | 285.6 (21.31%) | **252.5 (18.84%)** | 269.0 (20.08%) |
| 6 | **326.8 (15.07%)** | 366.5 (16.90%) | 373.8 (17.24%) | 345.5 (15.93%) |

The hybrid has the lowest MAE at three of the six horizons in this run, but it does not dominate at every horizon. These rolling-origin forecasts overlap, the dataset is small, and the evaluated years are not a separate prospective holdout. Treat the table as a reproducible benchmark—not proof of robust future superiority.

The complete scores, including RMSE, MAPE, mean error, MASE, R², and SARIMAX interval coverage, are in [`outputs/backtest_summary.csv`](outputs/backtest_summary.csv). Forecast-level results are in [`outputs/backtest_predictions.csv`](outputs/backtest_predictions.csv). The static dashboard is [`index.html`](index.html).

## Dataset and target

- **Target:** `Production (*000  Mt.)`, rice production in thousand metric tonnes—not yield per unit area.
- **Coverage:** 149 seasonal records, from Yala 1950 through Yala 2024. The final year contains a Yala record only.
- **Season sequence:** Yala and Maha are treated as alternating observations.
- **Workbook:** [`rice new one.xlsx`](rice%20new%20one.xlsx).

The workbook includes acreage, GDP, inflation, rainfall, and temperature columns. They are **not predictors in the current reference backtest**: the file does not provide historical forecast-time vintages, issue-time weather forecasts, or publication-vintage metadata. Using realized future covariates would make a pre-season evaluation optimistic. See [`docs/data_dictionary.md`](docs/data_dictionary.md) before proposing an exogenous-variable model.

The original project materials name national statistical, meteorological, and central-bank sources, but the workbook does not supply a source URL, retrieval date, revision history, or release vintage for each field. Verify and document these details before operational use.

## Models and evaluation

1. **Seasonal naive:** repeats the most recent observed value for the matching seasonal position.
2. **SARIMAX:** log production, order `(1, 1, 1)`, seasonal order `(1, 1, 0, 2)`, with no exogenous inputs.
3. **Direct RF (lags):** horizon-specific Random Forest using lagged production and the known target season.
4. **SARIMAX + RF residual:** horizon-specific Random Forest learns historical expanding-window SARIMAX errors. For each outer origin, only residual targets already observed at that origin are eligible for training.

The expanding-window evaluation has 33 origins per horizon. At every origin, model training uses only production observed by then. Exogenous actuals for the forecast period are not passed to the models. Model settings are fixed and conservative; no hyperparameter search is selected on the reported evaluation scores.

MAE, RMSE, WAPE, MAPE, mean error, MASE, and R² are reported by horizon. MASE uses the seasonal-naive in-sample error at each origin. SARIMAX 95% interval coverage is also included. Because origins and horizons overlap, metrics across origins are correlated; the reported number of forecasts is not an independent sample size.

The SARIMAX point forecast is lognormal-mean adjusted. Its interval is transformed from log space. No interval is presented for the hybrid because the SARIMAX interval alone would omit residual-model uncertainty.

See [`docs/methodology.md`](docs/methodology.md) for the full protocol and caveats.

## Reproduce the benchmark

The checked-in benchmark outputs were generated on Windows with Python 3.11.9. Installed package versions are recorded in `outputs/run_metadata.json`. The project supports Python 3.10+; small numerical differences may occur across platforms and dependency versions. The CI workflow uses its pinned Python 3.13 environment. From the repository root:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check src tests scripts
python -m rice_forecasting.cli --data ".\rice new one.xlsx" --output ".\outputs" --max-horizon 6 --evaluation-start 80 --origin-step 2 --residual-training-start 50 --minimum-ml-samples 20
python .\scripts\build_dashboard.py --summary .\outputs\backtest_summary.csv --output .\index.html
```

If PowerShell blocks environment activation, use `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell window, or call `.\.venv\Scripts\python.exe` directly.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check src tests scripts
python -m rice_forecasting.cli --data "rice new one.xlsx" --output outputs --max-horizon 6 --evaluation-start 80 --origin-step 2 --residual-training-start 50 --minimum-ml-samples 20
python scripts/build_dashboard.py --summary outputs/backtest_summary.csv --output index.html
```

The full default backtest re-fits SARIMAX repeatedly and may take a few minutes on some machines. Use `--help` to view options. `run_metadata.json` records the run settings and input hash.

## Repository layout

```text
src/rice_forecasting/   validated loader, model definitions, backtest, metrics, CLI
scripts/                dashboard builder
outputs/                reproducible prediction-level and summary CSVs
tests/                   automated checks
 docs/                   methodology, field definitions, release checklist
 legacy/                 original notebooks, HTML exports, and report, preserved for provenance
```

The notebooks and HTML exports under `legacy/` are historical artifacts with previously saved outputs; they are not the authoritative source of current metrics. The root pipeline and its outputs are authoritative for this release.

## Limitations and next steps

- Only 149 seasonal observations are available; performance uncertainty is material.
- The rolling backtest is retrospective and is not a separately held-out prospective test.
- No causal attribution to climate shocks is made. Such a claim requires a documented shock definition, feature ablations, and evaluation by event period.
- Climate, acreage, and economic regressors should be added only with a forecast-time availability policy and historical vintages or realistic proxy forecasts.
- Future operational forecasts and hybrid prediction intervals are not released in this benchmark.

Before publishing a model-performance claim, keep a final untouched time period, repeat the rolling-origin comparison, and report results by horizon and season. See [`docs/release_checklist.md`](docs/release_checklist.md).

## Project and license

**Project author:** Ali Naderi · [Portfolio](https://alinaderiii.github.io/) · [LinkedIn](https://www.linkedin.com/in/alinaderi-data-scientist) · [GitHub](https://github.com/AliNaderiii)

Historical source materials are preserved in `legacy/`. This project is distributed under the [MIT License](LICENSE). Source-data licensing and attribution should be verified independently before redistribution.
