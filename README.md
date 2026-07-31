# Bangladesh CPI & Inflation Forecasting

**Forecasting Consumer Prices in Data-Scarce, Crisis-Prone Economies: A Leakage-Controlled Comparison of Statistical, Machine Learning, Deep Learning, and Hybrid Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **Author:** Imran Hasan Sohan  
> **Supervisor:** Prof. Dr. Karim Mohammed Rezaul  
> **Affiliation:** Centre for Applied Research in Software and IT (CARSIT), Brit College of Engineering and Technology  
> **Contact:** imran173461@gmail.com | [@sohanever](https://github.com/sohanever)

---

## About

This repository accompanies an IEEE-format research manuscript that runs a leakage-controlled, head-to-head comparison of twelve forecasting models across four families -- statistical, machine learning, deep learning, and hybrid -- alongside four naive benchmarks, on Bangladeshi consumer prices. The evaluation includes Diebold-Mariano inference, ten-seed neural evaluation, multi-horizon robustness checks, a decomposition leakage ablation, a hyperparameter grid search, and replication on five further Asian economies.

The dataset covers **317 monthly observations (January 2000 - May 2026)**, pairing the harmonised CPI with ten macroeconomic drivers and three structural-break indicators. The 64-month test window (January 2021 - April 2026) deliberately contains the COVID-19 pandemic, the 2022-23 commodity shock, and the 2024 domestic unrest.

---

## Models Compared

**Statistical**
- ARIMA(3,2,3)
- SARIMA(2,1,2)x(0,1,1,12)
- LASSO regression with macroeconomic exogenous variables
- MARS-Spline + LASSO on first differences

**Deep Learning**
- LSTM (12-month lookback, 12 macroeconomic input features)
- Transformer with multi-head attention (4 heads)

**Hybrid**
- ARIMA-LSTM -- ARIMA handles trend, LSTM corrects residuals
- SARIMA-LSTM -- same idea with seasonal ARIMA
- ARIMA-HMM-LSTM -- adds Hidden Markov Model for regime detection
- SARIMA-HMM-LSTM
- Wavelet-SARIMA-LSTM -- causal wavelet decomposition (db4, level 2)
- VMD-LSTM -- Variational Mode Decomposition with 4 modes

**Naive benchmarks** -- random walk, drift-adjusted random walk, seasonal naive, drift-adjusted seasonal naive

---

## Results

| Model | R2 | RMSE | MAE | MAPE |
|---|---|---|---|---|
| SARIMA(2,1,2)x(0,1,1,12) | **0.9985** | **1.3053** | **0.9370** | **0.384%** |
| SARIMA-LSTM | 0.9985 | 1.3106 | 0.9298 | 0.381% |
| SARIMA-HMM-LSTM | 0.9983 | 1.3984 | 0.9799 | 0.401% |
| Wavelet-SARIMA-LSTM | 0.9977 | 1.6227 | 1.2504 | 0.506% |
| ARIMA-LSTM | 0.9950 | 2.4047 | 1.8126 | 0.743% |
| ARIMA-HMM-LSTM | 0.9944 | 2.5321 | 1.8254 | 0.743% |
| ARIMA(3,2,3) | 0.9940 | 2.6239 | 1.9146 | 0.786% |
| VMD-LSTM | 0.9939 | 2.6349 | 2.0054 | 0.804% |
| LSTM | 0.9937 | 2.6889 | 2.0589 | 0.838% |
| MARS-Spline-LASSO | 0.9932 | 2.7923 | 2.1712 | 0.886% |
| Transformer | 0.9923 | 2.9684 | 2.3559 | 0.971% |
| LASSO | 0.9770 | 5.1332 | 4.3137 | 1.676% |

SARIMA(2,1,2)x(0,1,1,12) won decisively -- significantly outperforming every competitor and every naive benchmark under Diebold-Mariano tests. Bangladesh CPI is dominated by trend plus a twelve-month seasonal rhythm driven by harvest cycles, and SARIMA captures that structure by specification rather than having to learn it from 252 training examples. The hybrid SARIMA-LSTM was a close runner-up but never significantly improved on pure SARIMA in ten seeds, and was significantly worse in three -- adding seed risk without adding accuracy.

The Model Confidence Set (Hansen, Lunde, Nason) retains only three models at both 90% and 95% confidence: SARIMA, SARIMA-LSTM, and SARIMA-HMM-LSTM. Every other model and all naive benchmarks are excluded with p < 0.005.

---

## Cross-Country Replication (Session 6)

The full protocol was replicated on five further Asian economies to test external validity. The statistical family supplies the champion in five of six economies; a univariate LSTM wins in none.

| Economy | Champion | Champion RMSE | SARIMA RMSE | ARIMA RMSE | LSTM RMSE |
|---|---|---|---|---|---|
| Bangladesh | SARIMA | **1.3053** | 1.3053 | 2.6239 | 2.751 |
| India | SARIMA | **0.7747** | 0.7747 | 0.8993 | 0.881 |
| Philippines | ARIMA | 0.7988 | 0.8413 | **0.7988** | 0.811 |
| Indonesia | ARIMA | **0.3830** | 0.4023 | 0.3830 | 0.450 |
| Pakistan | ARIMA | **2.3379** | 3.2671 | 2.3379 | 3.451 |
| Sri Lanka | ARIMA | **4.8652** | 9.1427 | 4.8652 | 6.103 |

The seasonal advantage is regime-dependent: Bangladesh's 50.2% SARIMA-over-ARIMA gain is three times India's and reverses entirely in Sri Lanka, where an inflation regime break (YoY inflation peaking at 72.5%) destroys the seasonal pattern the model was trained on.

---

## Leakage Ablation (Session 5)

Computing decompositions over the full sample instead of causally inflates accuracy, but the effect depends on the basis:

| Model | Causal RMSE | Full-series RMSE | Premium | Significant? |
|---|---|---|---|---|
| VMD-LSTM | 2.46 +/- 0.37 | 1.76 +/- 0.13 | **29.8 +/- 5.7%** | Yes (all 10 seeds) |
| Wavelet-SARIMA-LSTM | 1.55 +/- 0.17 | 1.55 +/- 0.19 | 4.8 +/- 11.0% | No (sign reverses in 3 seeds) |

The risk is concentrated in decompositions whose basis is estimated from the data (VMD) rather than fixed in advance (wavelet). Reports of large accuracy gains from adaptive decomposition that do not document causal computation should be read as upper bounds.

---

## Hyperparameter Grid Search (Session 6)

To rule out under-tuning as an explanation for the deep networks' performance, 27 LSTM and 6 Transformer configurations were each trained 3 times and ranked on a validation partition drawn entirely from the training period.

| Network | Grid best RMSE (10-seed) | Paper config RMSE (10-seed) | Gap |
|---|---|---|---|
| LSTM | 2.747 | 2.751 | 0.004 |
| Transformer | 2.808 | 2.818 | 0.010 |

Exhaustive selection buys less than one standard deviation of the seed distribution. The best configuration the grid can find still records 2.747 against the champion's 1.3053 -- the deep-learning verdict does not rest on tuning.

---

## Forecast: Jun 2026 - Dec 2027

Produced by refitting the champion SARIMA model on the full available series (January 2000 - May 2026):

| Month | CPI | 95% CI | Inflation (YoY) |
|---|---|---|---|
| Jun 2026 | 304.64 | [303.12 - 306.16] | 9.47% |
| Jul 2026 | 310.74 | [308.61 - 312.87] | 9.23% |
| Aug 2026 | 317.64 | [315.02 - 320.25] | 9.08% |
| Sep 2026 | 320.78 | [317.74 - 323.81] | 9.01% |
| Oct 2026 | 326.40 | [323.00 - 329.81] | 8.86% |
| Nov 2026 | 325.30 | [321.55 - 329.06] | 8.92% |
| Dec 2026 | 322.89 | [318.82 - 326.97] | 9.03% |
| Jun 2027 | 331.68 | [325.42 - 337.94] | 8.88% |
| Dec 2027 | 350.47 | [340.69 - 360.25] | 8.54% |

Inflation is projected to ease only from 9.47% to 8.54% by December 2027 -- roughly half a percentage point per year -- and at no point does the central projection reach the historical 5% comfort zone.

---

## Repository Layout

```
bangladesh-cpi-inflation-forecasting/
|
+-- notebooks/
|   +-- 01_statistical_models.ipynb         # ARIMA, SARIMA, LASSO, MARS
|   +-- 02_deep_learning_models.ipynb       # LSTM, Transformer
|   +-- 03_hybrid_models_and_forecast.ipynb # Hybrid models + final forecast
|   +-- 04_verification_multiseed.ipynb     # Multi-seed verification
|   +-- 05_leakage_ablation.ipynb          # Leakage ablation runbook
|   +-- 06_cross_country_replication.ipynb  # 6-economy replication
|   +-- 06b_hyperparameter_grid.ipynb      # Deep network grid search
|
+-- scripts/
|   +-- session5_leakage_ablation/          # Standalone ablation scripts
|   |   +-- session5_leakage_ablation.py
|   |   +-- session5b_warmup_diagnostic.py
|   |   +-- session5c_ablation_matched.py
|   |   +-- session5d_multiseed_ablation.py
|   +-- session6_hparam_grid/
|       +-- session6_hparam_grid.py         # Standalone grid search script
|
+-- data/
|   +-- README.md
|   +-- raw/
|   +-- processed/
|       +-- bangladesh_MASTER_dataset.csv
|       +-- all_model_results.csv           # 12-model scoreboard
|       +-- FINAL_scoreboard.csv            # Ranked scoreboard
|       +-- FINAL_forecast_2026_2027.csv    # Deployment forecast
|       +-- lasso_selected_features.csv     # Feature selection results
|       +-- multiseed_results.csv           # 10-seed neural results
|       +-- session6_grid_LSTM.csv          # LSTM grid search results
|       +-- session6_grid_Transformer.csv   # Transformer grid results
|       +-- session6_grid_summary.json      # Grid search summary
|       +-- errors_s1.npz, errors_s2.npz, errors_s3.npz
|       +-- session5_*.csv, session5_*.npz  # Ablation results
|
+-- figures/                                # All charts from the notebooks
|
+-- LICENSE
+-- README.md
```

**Note on large files:** The raw IMF datasets used to construct the master dataset are too large for GitHub (90 MB - 333 MB each). They are not included here. See [`data/README.md`](data/README.md) for download instructions.

---

## Notebooks

### `01_statistical_models.ipynb`

Covers data ingestion, master dataset construction, and exploratory analysis (EDA series plots, correlation matrix, STL decomposition, ADF unit root tests). Then fits the statistical model family: auto-ARIMA grid search (best order: `ARIMA(3,2,3)`), seasonal ARIMA grid search (best: `SARIMA(2,1,2)x(0,1,1,12)`), LASSO with macroeconomic regressors, and MARS-Spline on first differences. Includes Diebold-Mariano tests and naive benchmark evaluation. Saves per-model error vectors for downstream inference.

### `02_deep_learning_models.ipynb`

Fits an LSTM and a Transformer on a sequence-to-point task: given the last 12 months of 12 macroeconomic features, predict next month's CPI change. The StandardScaler is deliberately fit on training data only to avoid leakage. Requires a GPU runtime (T4 or better recommended). Saves error vectors for DM testing.

### `03_hybrid_models_and_forecast.ipynb`

Builds the hybrid models (ARIMA-LSTM, SARIMA-LSTM, HMM-augmented variants, Wavelet-SARIMA-LSTM, VMD-LSTM), produces the full 12-model scoreboard, then refits the champion SARIMA model on the complete series and generates the 2026-2027 forecast with 95% confidence intervals. Includes residual diagnostics (ACF, PACF, Q-Q plot) showing SARIMA residuals approach white noise.

### `04_verification_multiseed.ipynb`

Re-estimates the neural and hybrid models over ten random seeds (42, 7, 13, 21, 99, 123, 256, 314, 777, 2024) to confirm the one-step tournament ranking is not a single-seed artefact. Reports mean, standard deviation, and per-seed Diebold-Mariano tests against SARIMA. Includes multi-horizon evaluation (3, 6, 12 months) and inflation-rate scoring.

### `05_leakage_ablation.ipynb`

Colab runbook for the leakage ablation study. Executes four sub-sessions: initial causal vs full-series test, warm-up sensitivity diagnostic, matched pipeline ablation reproducing the published Table VI values, and 10-seed confirmation. Shows that VMD leakage premium (29.8%) is systematic while wavelet premium (4.8%) is indistinguishable from seed noise.

### `06_cross_country_replication.ipynb`

Replicates the full protocol on India, Philippines, Indonesia, Pakistan, and Sri Lanka using IMF CPI data. Model orders are selected independently per country by the same AIC grids. Confirms the statistical family wins in 5 of 6 economies, the seasonal advantage is regime-dependent, and the LSTM clears the drift benchmark in only 2 of 6.

### `06b_hyperparameter_grid.ipynb`

Trains 27 LSTM configurations (3 unit sizes x 3 dropout rates x 3 lookback windows) and 6 Transformer configurations (3 head counts x 2 key dimensions), each 3 times, ranked on a training-period validation partition. Confirms the paper's architecture choice is within 0.004-0.010 index points of the grid optimum.

---

## Data & Variables

The master dataset contains 317 monthly observations from January 2000 onward. Key variables:

| Variable | Description | Source |
|---|---|---|
| `CPI` | Bangladesh Consumer Price Index (2010=100) | IMF IFS / BBS |
| `CPI_Food` | Food sub-index (from 2010) | IMF IFS / BBS |
| `Inflation_YoY` | Year-on-year % change in CPI | Derived |
| `ExchangeRate_BDT_USD` | BDT per USD | IMF |
| `Forex_Reserves_USDmn` | Bangladesh foreign exchange reserves (USD mn) | IMF |
| `BroadMoney_BDTmn` | M2 broad money supply (BDT mn) | IMF MFS |
| `Brent_Oil_USD` | Brent crude oil price (USD/barrel) | FRED |
| `Fed_Funds_Rate` | US Federal Funds Rate (%) | FRED |
| `Gold_Price_Index` | Gold price index | IMF IFS |
| `FAO_Food_Index` | FAO Food Price Index | FAO |
| `FAO_Cereals_Index` | FAO Cereals Price Index | FAO |
| `COVID_dummy` | 1 for March 2020 - June 2021 | Engineered |
| `UkraineWar_dummy` | 1 for February 2022 onward | Engineered |
| `BD_Unrest_dummy` | 1 for July-August 2024 | Engineered |

---

## Methodology Notes

**Train / test split**

```
Jan 2000 -------------- Dec 2020  |  Jan 2021 ---------- Apr 2026
        Training (252 months)      |     Test (64 months)
```

**Leakage prevention checklist**

1. Feature scalers fitted on training data only
2. Wavelet and VMD decompositions computed causally (rolling, past-only)
3. HMM fitted on training residuals only
4. Exogenous variables lagged one month
5. Chronological split, no shuffling
6. Every random seed fixed for archived runs

**Inference**

Model rankings tested with the Diebold-Mariano statistic (squared-error loss, HLN small-sample correction). Neural results reported over ten seeds. Model Confidence Set (Hansen-Lunde-Nason, 5000 bootstrap replications) confirms only SARIMA-based models survive at 90% and 95% confidence. Holm step-down multiplicity correction applied to the eight pairwise comparisons.

---

## Reproducing the Results

All notebooks are designed to run on **Google Colab**. To reproduce:

1. Upload your data files to Google Drive under `MyDrive/CPI_Research/`
2. Open each notebook in Colab in order (Session 1 -> 2 -> 3 -> 4 -> 5 -> 6)
3. For Sessions 2, 3, 5, and 6: switch the runtime to **T4 GPU** before running
4. Each session saves its results to shared files that later sessions read from
5. For Sessions 5 and 6: upload the Python scripts from `scripts/` to the Colab runtime

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
  title        = {Forecasting Consumer Prices in Data-Scarce, Crisis-Prone Economies:
                  A Leakage-Controlled Comparison of Statistical, Machine Learning,
                  Deep Learning, and Hybrid Models},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/sohanever/bangladesh-cpi-inflation-forecasting}}
}
```

---

## License

Code is released under the [MIT License](LICENSE). Data files are subject to the terms of their respective sources -- see [`data/README.md`](data/README.md) for full attribution.
