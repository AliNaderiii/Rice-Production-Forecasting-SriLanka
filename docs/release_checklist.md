# Release checklist

- [ ] Create a clean virtual environment and install from `pyproject.toml`.
- [ ] Run `pytest` and `ruff check src tests scripts`.
- [ ] Run the CLI from the repository root against the checked-in workbook.
- [ ] Confirm `outputs/backtest_summary.csv`, `outputs/backtest_predictions.csv`, and `index.html` are generated from the same run.
- [ ] Check that all README metrics match the generated summary and are labeled by horizon.
- [ ] Verify that the forecast-time information set remains past-only.
- [ ] Review data-source attribution, licensing, and any revisions to the input workbook.
- [ ] Do not describe rolling-origin scores as independent samples or as a guarantee of future performance.
- [ ] Do not claim climate-shock attribution without a documented ablation and event definition.
