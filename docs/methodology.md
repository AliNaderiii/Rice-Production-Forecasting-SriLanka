# Evaluation protocol

## Forecast target and unit

The supplied workbook has 149 seasonal observations, from Yala 1950 through Yala 2024. The target is rice production in thousand metric tonnes (`Production (*000  Mt.)`), not yield per unit of land. Yala and Maha are treated as alternating seasonal observations. The final year has a Yala row only.

## Information set at forecast time

The reference backtest uses historical target values and the future season label only. The season label is known in advance. Realized test-period rainfall, temperature, GDP, inflation, sown area, and harvested area are not used as predictors. The workbook does not provide historical data vintages or issue-time forecasts for those covariates, so their realized values cannot be used to make an operationally honest pre-season backtest.

This is a deliberately conservative target-history benchmark. Exogenous features may be added only when a forecast-time data-availability policy is documented and those inputs can be reconstructed at every historical forecast origin.

## Expanding-window design

The default run uses an expanding training window with 80 or more observed seasonal rows at the first evaluation origin, advances by two rows, and forecasts 1 through 6 seasons ahead. The final eligible origin is included even if it does not align with the two-row step. At each origin, all models see the same observed target history. The end of the series is used for historical rolling-origin evaluation, so this is not a separately held-out prospective test after model selection.

1. **Seasonal naive:** repeats the latest available observation for the corresponding season.
2. **SARIMAX:** log target, order `(1, 1, 1)`, seasonal order `(1, 1, 0, 2)`; no exogenous regressors.
3. **Direct RF:** one horizon-specific random forest on lagged production and known target-season indicators.
4. **SARIMAX + RF residual:** the SARIMAX point forecast is corrected by a horizon-specific RF trained only on historical expanding-window SARIMAX forecast errors. At an evaluation origin, a residual label is eligible only when its target observation would already have been observed.

Random-forest settings are fixed and conservative; the current result is a comparison, not a tuned winner. There is no random shuffling and no hyperparameter search on evaluation outcomes.

## Metrics

Scores are reported separately by horizon: MAE, RMSE, WAPE, MAPE, mean error, MASE, and R-squared. MASE is scaled against the seasonal-naive in-sample error at each origin. The SARIMAX 95% interval coverage is also reported. Because forecast origins and horizons overlap, these scores are not independent observations and should not be interpreted as formal confidence intervals.

## Log-scale retransformation

The SARIMAX point forecast is transformed from log space using the lognormal mean adjustment `exp(mu + 0.5 * variance)`. Interval endpoints are transformed from the corresponding log-scale normal limits. The hybrid prediction interval is intentionally not published: the SARIMAX interval alone would omit residual-model and model-selection uncertainty.

## Limitations

- There are only 149 observations, with fewer observations available at earlier origins.
- Model comparison uncertainty is substantial; overlapping rolling origins are correlated.
- The current reference models do not test weather or macroeconomic covariates because historical forecast-time vintages are unavailable.
- No causal claim about climate shocks is made. Feature ablations and shock-period definitions require separate validation.
- Forecasts beyond the observed record are not presented as operational forecasts in this release.
