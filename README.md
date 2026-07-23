# Bangladesh CPI & Inflation Forecasting

**Forecasting Consumer Prices in a Data-Scarce, Crisis-Prone Economy: A Leakage-Controlled Comparison of Statistical, Machine Learning, Deep Learning, and Hybrid Models for Bangladesh**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **Author:** Imran Hasan Sohan  
> **Supervisor:** Karim Mohammed Rezaul  
> **Affiliation:** Centre for Applied Research in Software and IT (CARSIT), Brit College of Engineering and Technology  
> **Contact:** imran173461@gmail.com | [@sohanever](https://github.com/sohanever)

---

## About

This repository accompanies an IEEE-format research manuscript studying how well different forecasting approaches can predict the Consumer Price Index (CPI) and inflation in Bangladesh. The study runs a leakage-controlled, head-to-head comparison of twelve models across four families — statistical, machine learning, deep learning, and hybrid — alongside four naive benchmarks, with Diebold–Mariano inference, ten-seed neural evaluation, and multi-horizon robustness checks.

The dataset covers **317 monthly observations (January 2000 – May 2026)**, pairing the harmonised CPI with ten macroeconomic drivers and three structural-break indicators. The 64-month test window (January 2021 – April 2026) deliberately contains the COVID-19 pandemic, the 2022–23 commodity shock, and the 2024 domestic unrest — stress-testing every model against exactly the conditions a policymaker cares about.

A separate leakage ablation study (Session 5) quantifies the information leakage premium in wavelet and VMD decomposition hybrids, showing that full-series decomposition — common in the literature — flatters published accuracy numbers.

---

## Models Compared

**Statistical**
- ARIMA(3,2,3)
- SARIMA(2,1,2)×(0,1,1,12)
- LASSO regression with macroeconomic exogenous variables
- MARS-Spline + LASSO on first differences

**Deep Learning**
- LSTM (12-month lookback, 12 macroeconomic input features)
- Transformer with multi-head attention (4 heads)

**Hybrid**
- ARIMA-LSTM — ARIMA handles trend, LSTM corrects residuals
- SARIMA-LSTM — same idea with seasonal ARIMA
- ARIMA-HMM-LSTM — adds Hidden Markov Model for regime detection
- SARIMA-HMM-LSTM
- Wavelet-SARIMA-LSTM — causal wavelet decomposition (db4, level 2)
- VMD-LSTM — Variational Mode Decomposition with 4 modes

**Naive benchmarks** — random walk, drift-adjusted random walk, seasonal naive, drift-adjusted seasonal naive

---

## Results

| Model | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| SARIMA(2,1,2)×(0,1,1,12) | **0.9985** | **1.3053** | **0.9370** | **0.384%** |
| SARIMA-LSTM | 0.9985 | 1.3106 | 0.9298 | 0.381% |
| SARIMA-HMM-LSTM | 0.9983 | 1.3984 | 0.9799 | 0.401% |
| Wavelet-SARIMA-LSTM | 0.9977 | 1.6227 | 1.2504 | 0.506% |
| ARIMA-LSTM | 0.9950 | 2.4047 | 1.8126 | 0.743% |
| ARIMA-HMM-LSTM | 0.9944 | 2.5321 | 1.8254 | 0.743% |
| ARIMA(3,2,3) | 0.9940 | 2.6239 | 1.9146 | 0.786% |
| VMD-LSTM | 0.9939 | 2.6349 | 2.0054 | 0.804% |
| LSTM | 0.9937 | 2.6889 | 2.0589 | 0.838% |
| Transformer | 0.9933 | 2.7714 | 2.1248 | 0.866% |
| MARS-Spline-LASSO | 0.9932 | 2.7923 | 2.1712 | 0.886% |
| LASSO | 0.9770 | 5.1332 | 4.3137 | 1.676% |

SARIMA(2,1,2)×(0,1,1,12) won decisively — significantly outperforming every competitor and every naive benchmark under Diebold–Mariano tests. Bangladesh CPI is dominated by trend plus a twelve-month seasonal rhythm driven by harvest cycles, and SARIMA captures that structure by specification rather than having to learn it from 252 training examples. The hybrid SARIMA-LSTM was a close runner-up but never significantly improved on pure SARIMA in ten seeds, and was significantly worse in three — adding seed risk without adding accuracy.

---

## Leakage Ablation (Session 5)

A key finding of this work is that decomposition gains reported in the literature partly reflect information leakage. When wavelet and VMD decompositions are computed causally (month by month, using only past data), they underperform their full-series counterparts:

| Model | Causal RMSE | Full-series RMSE | Premium |
|---|---|---|---|
| Wavelet-SARIMA-LSTM | 1.5928 | 1.4277 | +10.4% |
| VMD-LSTM | 2.4642 | 1.7587 | +28.6% |

The full-series version looks better only because it lets future observations shape components the model trains on. The ablation was validated across ten random seeds and the published values fall inside the causal seed distribution, confirming the result is not an artefact of GPU nondeterminism.

---

## Forecast: Jun 2026 – Dec 2027

Produced by refitting the champion SARIMA model on the full available series (January 2000 – May 2026):

| Month | CPI | 95% CI | Inflation (YoY) |
|---|---|---|---|
| Jun 2026 | 304.64 | [303.12 – 306.16] | 9.47% |
| Jul 2026 | 310.74 | [308.61 – 312.87] | 9.23% |
| Aug 2026 | 317.64 | [315.02 – 320.25] | 9.08% |
| Sep 2026 | 320.78 | [317.74 – 323.81] | 9.01% |
| Oct 2026 | 326.40 | [323.00 – 329.81] | 8.86% |
| Nov 2026 | 325.30 | [321.55 – 329.06] | 8.92% |
| Dec 2026 | 322.89 | [318.82 – 326.97] | 9.03% |
| Jun 2027 | 331.68 | [325.42 – 337.94] | 8.88% |
| Dec 2027 | 350.47 | [340.69 – 360.25] | 8.54% |

Inflation is projected to ease only from 9.47% to 8.54% by December 2027 — roughly half a percentage point per year — and at no point does the central projection reach the historical 5% comfort zone.

---

## Repository Layout

```
bangladesh-cpi-inflation-forecasting/
│
├── notebooks/
│   ├── 01_statistical_models.ipynb         # ARIMA, SARIMA, LASSO, MARS
│   ├── 02_deep_learning_models.ipynb       # LSTM, Transformer
│   ├── 03_hybrid_models_and_forecast.ipynb # Hybrid models + final forecast
│   ├── 04_verification_multiseed.ipynb     # Multi-seed verification
│   └── 05_leakage_ablation.ipynb          # Leakage ablation runbook
│
├── scripts/
│   └── session5_leakage_ablation/          # Standalone ablation scripts
│       ├── session5_leakage_ablation.py    # Initial causal vs full-series test
│       ├── session5b_warmup_diagnostic.py  # Warm-up sensitivity analysis
│       ├── session5c_ablation_matched.py   # Matched pipeline (reproduces Table VI)
│       └── session5d_multiseed_ablation.py # 10-seed leakage premium
│
├── data/
│   ├── README.md                           # Data sources and download instructions
│   ├── raw/                                # Source data files
│   └── processed/                          # Cleaned datasets + ablation results
│
├── figures/                                # All charts generated by the notebooks
│
├── LICENSE
└── README.md
```

**Note on large files:** The raw IMF datasets used to construct the master dataset are too large for GitHub (90 MB – 333 MB each). They are not included here. See [`data/README.md`](data/README.md) for download instructions.

---

## Notebooks

### `01_statistical_models.ipynb`

Covers data ingestion, master dataset construction, and exploratory analysis. Then fits the statistical model family: auto-ARIMA grid search (best order: `ARIMA(3,2,3)`), seasonal ARIMA grid search (best: `SARIMA(2,1,2)×(0,1,1,12)`), LASSO with macroeconomic regressors, and MARS-Spline on first differences. Results from all four models are saved to a shared results log that the next sessions append to.

### `02_deep_learning_models.ipynb`

Fits an LSTM and a Transformer on a sequence-to-point task: given the last 12 months of 12 macroeconomic features, predict next month's CPI change. The StandardScaler is deliberately fit on training data only to avoid leakage. Requires a GPU runtime (T4 or better recommended).

### `03_hybrid_models_and_forecast.ipynb`

Builds the hybrid models (ARIMA-LSTM, SARIMA-LSTM, HMM-augmented variants, Wavelet-SARIMA-LSTM, VMD-LSTM), produces the full 12-model scoreboard, then refits the champion SARIMA model on the complete series and generates the 2026–2027 forecast with 95% confidence intervals.

### `04_verification_multiseed.ipynb`

Re-estimates the neural and hybrid models over ten random seeds (42, 7, 13, 21, 99, 123, 256, 314, 777, 2024) to confirm the one-step tournament ranking is not a single-seed artefact. Reports mean, standard deviation, and per-seed Diebold–Mariano tests against SARIMA.

### `05_leakage_ablation.ipynb`

Colab runbook for the leakage ablation study. Executes four sub-sessions in order:

- **Session 5** — initial ablation comparing causal vs full-series wavelet and VMD decompositions
- **Session 5b** — warm-up sensitivity diagnostic for the causal VMD arm (identifies that training length does not explain the deviation from Table VI)
- **Session 5c** — matched pipeline ablation that lifts the Session 3 code verbatim, changing only the decomposition method, so the causal arms reproduce the published RMSE values
- **Session 5d** — multi-seed ablation over the same ten seeds, confirming the leakage premium holds across seeds and that published values fall inside the causal distribution

The standalone Python scripts live in `scripts/session5_leakage_ablation/` and can be run independently on Colab or locally.

---

## Data & Variables

The master dataset contains 317 monthly observations from January 2000 onward. Key variables:

| Variable | Description | Source |
|---|---|---|
| `CPI` | Bangladesh Consumer Price Index (2010=100) | IMF IFS / BBS |
| `Inflation_from_CPI` | Year-on-year % change in CPI | Derived |
| `ExchangeRate_BDT_USD` | BDT per USD | IMF |
| `Forex_Reserves_USDmn` | Bangladesh foreign exchange reserves (USD mn) | IMF |
| `BroadMoney_BDTmn` | M2 broad money supply (BDT mn) | IMF MFS |
| `Brent_Oil_USD` | Brent crude oil price (USD/barrel) | FRED |
| `Fed_Funds_Rate` | US Federal Funds Rate (%) | FRED |
| `Gold_Price_Index` | Gold price index | IMF IFS |
| `FAO_Food_Index` | FAO Food Price Index | FAO |
| `FAO_Cereals_Index` | FAO Cereals Price Index | FAO |
| `COVID_dummy` | 1 for March 2020 – June 2021 | Engineered |
| `UkraineWar_dummy` | 1 for February 2022 onward | Engineered |
| `BD_Unrest_dummy` | 1 for July–August 2024 | Engineered |

---

## Methodology Notes

**Train / test split**

```
Jan 2000 ─────────────── Dec 2020  |  Jan 2021 ────────── Apr 2026
        Training (252 months)      |     Test (64 months)
```

**Leakage prevention**

All scalers are fit on training data only. The VMD and wavelet decompositions use causal (rolling) implementations — at any point in time, only past observations are used in the decomposition. The HMM is fit on training residuals only. Exogenous variables are lagged one month. The SARIMA rolling forecast on the test set is strictly one-step-ahead.

**Inference**

Model rankings are tested with the Diebold–Mariano statistic on squared-error loss with the Harvey–Leybourne–Newbold small-sample correction. Neural results are reported over ten random seeds rather than single-seed point estimates.

---

## Reproducing the Results

All notebooks are designed to run on **Google Colab**. To reproduce:

1. Upload your data files to Google Drive under `MyDrive/CPI_Research/`
2. Open each notebook in Colab in order (Session 1 → 2 → 3 → 4 → 5)
3. For Sessions 2, 3, and 5: switch the runtime to **T4 GPU** before running
4. Each session saves its results to a shared CSV (`all_model_results.csv`) that the next session reads from
5. For Session 5: upload the Python scripts from `scripts/session5_leakage_ablation/` to the Colab runtime

If running locally instead of Colab:
```bash
git clone https://github.com/sohanever/bangladesh-cpi-inflation-forecasting.git
cd bangladesh-cpi-inflation-forecasting
pip install tensorflow statsmodels scikit-learn pandas numpy matplotlib seaborn hmmlearn PyWavelets vmdpy
jupyter notebook
```

---

## Citation

```bibtex
@misc{sohan2026bangladesh,
  author       = {Imran Hasan Sohan},
  title        = {Forecasting Consumer Prices in a Data-Scarce, Crisis-Prone Economy:
                  A Leakage-Controlled Comparison of Statistical, Machine Learning,
                  Deep Learning, and Hybrid Models for Bangladesh},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/sohanever/bangladesh-cpi-inflation-forecasting}}
}
```

---

## License

Code is released under the [MIT License](LICENSE). Data files are subject to the terms of their respective sources — see [`data/README.md`](data/README.md) for full attribution.
