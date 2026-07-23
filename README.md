# 🌾 Rice Production Forecasting in Sri Lanka

**Advanced Time Series & Machine Learning Forecasting**  
**1950 – 2024 | Yala & Maha Seasons**

![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-red)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-orange)

---

## 📋 Project Overview

Accurate rice production forecasting is essential for **food security**, agricultural planning, and policy-making in Sri Lanka.  

This project combines **classical time series analysis (SARIMAX)** with **Machine Learning** and a **Hybrid modeling approach** to deliver high-accuracy forecasts while strictly preserving temporal structure and preventing data leakage.

**🏆 Best Performing Model**:  
**Hybrid SARIMAX + Random Forest Residuals** → **MAPE = 5.12%** | **R² = 0.9254**

---

## ✨ Key Features

- Comprehensive Exploratory Data Analysis (EDA) with clear seasonal patterns (Yala vs Maha)
- Stationarity testing and transformation (ADF, Log, Differencing)
- SARIMAX modeling with exogenous variables (Rainfall & Harvested Acres)
- Machine Learning models (Random Forest & XGBoost)
- **Hybrid Model**: SARIMAX for trend/seasonality + RF on residuals
- Strict time-based train/test split (no data leakage)
- Future forecasting (2024–2027)
- Clean, well-documented, and reproducible Jupyter Notebooks

---

## 📊 Dataset

- **Source**: Department of Census & Statistics, Meteorology Department, Central Bank of Sri Lanka
- **Time Range**: 1950 – 2024 (Seasonal/Quarterly)
- **Target Variable**: Rice Production (`000 Mt`)
- **Key Features**: Sown Acres, Harvested Acres, Rainfall (mm), Temperature (°C), GDP, Inflation, Season (Yala/Maha)

---

## 🏆 Model Performance (Test Set)

| Model                        | R² Score | RMSE (Mt) | MAPE (%)   | Remark                      |
|-----------------------------|----------|-----------|------------|-----------------------------|
| SARIMA Baseline             | 0.5401   | 238.45    | 8.12%      | Basic seasonality           |
| **SARIMAX (Best Single)**   | **0.8924** | **114.65** | **6.67%**  | With exogenous variables    |
| Random Forest               | 0.9134   | 210.33    | 19.25%     | Strong non-linear capture   |
| XGBoost                     | 0.8789   | 248.81    | 21.96%     | Gradient Boosting           |
| **Hybrid SARIMAX + RF**     | **0.9254** | **98.11**  | **5.12%**  | **Best Overall**            |

---

## 📁 Repository Structure

Rice-Production-Forecasting-SriLanka/
├── data/                          # Raw & processed datasets
├── notebooks/
│   ├── 01_Rice_Production_Advanced_EDA.ipynb
│   └── 02_Rice_Production_Advanced_Modeling.ipynb
├── reports/
│   ├── Rice_Production_Advanced_EDA.html
│   └── Rice_Production_Advanced_Modeling.html
├── images/                        # Key visualizations & plots
├── src/                           # Python scripts (optional)
├── requirements.txt
├── portfolio.html                 # Beautiful one-page portfolio
└── README.md


---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Rice-Production-Forecasting-SriLanka.git
   cd Rice-Production-Forecasting-SriLanka
   Install dependencies:Bash
   pip install -r requirements.txt

   Launch Jupyter Notebook:Bash
   jupyter notebook
   Start with 01_Rice_Production_Advanced_EDA.ipynb, then proceed to the modeling notebook.

   🔍 Key Insights

Maha season (Northeast monsoon) produces significantly higher yields than Yala.
Harvested Acres is the strongest predictor (Spearman r > 0.90).
Rainfall and temperature have important but non-linear effects.
The Hybrid model effectively combines linear seasonality with non-linear residual patterns.


🛠 Technologies Used

Time Series: Statsmodels (SARIMAX)
Machine Learning: Scikit-learn, XGBoost
Visualization: Matplotlib, Seaborn, Plotly
Data Handling: Pandas, NumPy


📄 References & Inspiration

Original project report by Ashfaq M (KIC-HNDDS-241-F-007)
Department of Census and Statistics Sri Lanka
Academic literature on SARIMAX for agricultural forecasting


📬 Contact

LinkedIn: https://www.linkedin.com/in/alinaderi-data-scientist
Kaggle: www.kaggle.com/Alinaderi1
Email: Alinaderi119@gmail.com


⭐ If you find this project useful, please give it a star!
Made with ❤️ for Sri Lanka’s Agricultural Data Science
