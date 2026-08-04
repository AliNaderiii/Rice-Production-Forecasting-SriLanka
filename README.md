# Rice Production Forecasting in Sri Lanka
Time series and machine learning forecasting of paddy production across the Yala and Maha seasons, 1950 to 2024.
[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-E85D4A?style=for-the-badge&logo=googlechrome&logoColor=white)](https://alinaderiii.github.io/Rice-Production-Forecasting-SriLanka/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4051B5?style=for-the-badge)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
### [Open the interactive dashboard](https://alinaderiii.github.io/Rice-Production-Forecasting-SriLanka/)
## Overview
Rice is the staple crop of Sri Lanka. Accurate production forecasts inform import planning, price policy and food security decisions, and the country is exposed to monsoon-driven volatility across two distinct growing seasons.
This project combines classical time series analysis with machine learning, then couples the two: SARIMAX captures trend and seasonality, and a Random Forest learns the structure remaining in the residuals.
**Best model: Hybrid SARIMAX + Random Forest residuals, R2 = 0.9254, MAPE = 5.12%**
## Results
Chronological validation. Train 1950Q2 to 2018Q4, test 2019Q2 to 2024Q2. No random shuffling at any point.
| Model | Validation strategy | R2 | RMSE (Mt) | MAPE |
| :--- | :--- | ---: | ---: | ---: |
| SARIMA baseline (1,1,1)(1,1,0,2) | Chronological split | 0.5401 | 238.45 | 8.12% |
| SARIMAX with exogenous forecasts | Chronological + forecast exog | 0.8924 | 114.65 | 6.67% |
| Random Forest | RobustScaled features | 0.9134 | 210.33 | 19.25% |
| XGBoost | RobustScaled features | 0.8789 | 248.81 | 21.96% |
| **Hybrid SARIMAX + RF residuals** | Residual coupling | **0.9254** | **98.11** | **5.12%** |
**Reading the table.** Random Forest posts a respectable R2 of 0.9134 but a MAPE of 19.25%, nearly four times the hybrid. R2 rewards explaining variance across the whole series; MAPE penalises being wrong on individual seasons. A model can score well on one and badly on the other, which is why both are reported.
The hybrid wins because the two components fail in different places. SARIMAX handles the seasonal structure but cannot model non-linear climate effects. Random Forest captures those effects but has no notion of temporal ordering. Fitting RF to SARIMAX residuals lets each do what it is good at.
## Forecasts
Six seasons ahead, generated with SARIMAX using exponentially smoothed exogenous forecasts.
| Season | Year | Forecast (000 Mt) | 95% CI |
| :--- | :--- | ---: | :--- |
| Maha | 2024/2025 | 1,942.53 | 1,805.10 - 2,090.41 |
| Yala | 2025 | 2,003.28 | 1,822.14 - 2,202.61 |
| Maha | 2025/2026 | 2,105.59 | 1,889.30 - 2,346.70 |
| Yala | 2026 | 2,230.89 | 1,955.10 - 2,545.92 |
| Maha | 2026/2027 | 2,341.22 | 2,014.20 - 2,721.43 |
| Yala | 2027 | 2,488.10 | 2,102.50 - 2,944.51 |
Production is projected to exceed 2,400 thousand metric tons by Yala 2027 under normal weather assumptions. Forecasting the exogenous variables rather than holding them constant avoids the flatline projection error present in the baseline model, and widens the confidence intervals to something realistic.
## Dataset
| Property | Value |
| :--- | :--- |
| Sources | Department of Census and Statistics, Department of Meteorology, Central Bank of Sri Lanka |
| Period | 1950 to 2024, seasonal (Yala and Maha) |
| Records | 149 seasons |
| Target | Rice production (000 Mt) |
**Features**
| Feature | Type | Monotonic correlation |
| :--- | :--- | :--- |
| Harvested acres | Continuous | +0.904 |
| Sown acres | Continuous | +0.887 |
| GDP (billion USD) | Continuous | +0.835 |
| Rainfall (mm) | Continuous | +0.365 |
| Inflation (%) | Continuous | +0.337 |
| Temperature (C) | Continuous | -0.295 |
| Season | Categorical | Strong seasonal driver |
## Methodology
**Stationarity.** ADF testing confirms log(production) is non-stationary (p = 0.34) but stationary after first differencing (p < 0.0001), giving d = 1.
**Outliers kept, not dropped.** GDP, inflation and temperature contain 23 outliers between them. Rather than removing rows and breaking the time index, features are scaled with RobustScaler, which is insensitive to extreme values.
**Leakage prevention.** The train/test boundary is chronological, never random. Exogenous variables for the test period are forecast, not taken from actuals, so the model never sees future information.
**Seasonal structure.** Maha (northeast monsoon) yields nearly double Yala: mean 1,488.8 Mt against 857.8 Mt. Seasonal period s = 2.
## Repository contents
| File | Description |
| :--- | :--- |
| [Live dashboard](https://alinaderiii.github.io/Rice-Production-Forecasting-SriLanka/) | Interactive summary of the full analysis |
| [Rice_Production_Advanced_EDA.ipynb](Rice_Production_Advanced_EDA.ipynb) | Exploratory analysis, stationarity testing, correlation |
| [Rice_Production_Advanced_Modeling.ipynb](Rice_Production_Advanced_Modeling.ipynb) | SARIMAX, ML models, hybrid coupling, forecasts |
| [EDA HTML render](https://alinaderiii.github.io/Rice-Production-Forecasting-SriLanka/Rice_Production_Advanced_EDA.html) | Read the EDA without running anything |
| [Modeling HTML render](https://alinaderiii.github.io/Rice-Production-Forecasting-SriLanka/Rice_Production_Advanced_Modeling.html) | Read the modelling notebook in-browser |
| rice new one.xlsx | Source dataset |
## How to run
```bash
git clone https://github.com/AliNaderiii/Rice-Production-Forecasting-SriLanka.git
cd Rice-Production-Forecasting-SriLanka
```
```bash
pip install pandas numpy statsmodels scikit-learn xgboost matplotlib seaborn plotly openpyxl jupyter
jupyter notebook
```
Start with `Rice_Production_Advanced_EDA.ipynb`, then `Rice_Production_Advanced_Modeling.ipynb`.
## Hybrid model
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
# 1. Fit SARIMAX on log-scaled production
sarimax_fit = SARIMAX(
    y_train_log, exog=exog_train,
    order=(1, 1, 1), seasonal_order=(1, 1, 0, 2)
).fit(disp=False)
sarimax_fitted_train = np.exp(sarimax_fit.fittedvalues)
# 2. Extract residuals
train_residuals = train_df[TARGET] - sarimax_fitted_train
# 3. Train RF on the residuals
rf_residual_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_residual_model.fit(X_train_preped, train_residuals)
# 4. Combine
sarimax_test_pred = np.exp(sarimax_fit.forecast(steps=len(test_df), exog=exog_test))
rf_pred_residuals = rf_residual_model.predict(X_test_preped)
hybrid_pred = sarimax_test_pred + rf_pred_residuals
```
## Tech stack
| Layer | Tools |
| :--- | :--- |
| Time series | statsmodels (SARIMAX), Holt-Winters |
| Machine learning | scikit-learn, XGBoost |
| Data handling | pandas, NumPy, openpyxl |
| Visualisation | Matplotlib, Seaborn, Plotly |
## Limitations
The test set covers six seasons. That is enough to compare models but too few to establish confidence in any single forecast. Sri Lankan agriculture over this period was also affected by the 2021 fertiliser import ban and the 2022 economic crisis, neither of which is captured by the exogenous variables used here. Treat the 2025 to 2027 projections as a baseline under normal conditions, not a prediction.
## Attribution
This project reimplements and extends the analysis from a report by Ashfaq M (KIC-HNDDS-241-F-007). The SARIMAX and Random Forest residual coupling, the leakage-free validation design, the exogenous forecasting approach and all code in this repository are my own work.
Completed as a Higher National Diploma in Data Science project.
## Author
**Ali Naderi** - AI Research Engineer and Data Scientist, Dublin, Ireland
[Portfolio](https://alinaderiii.github.io/) | [LinkedIn](https://www.linkedin.com/in/alinaderi-data-scientist) | [GitHub](https://github.com/AliNaderiii) | [Kaggle](https://www.kaggle.com/alinaderi1) | [alinaderi119@gmail.com](mailto:alinaderi119@gmail.com)
## License
Released under the [MIT License](LICENSE).