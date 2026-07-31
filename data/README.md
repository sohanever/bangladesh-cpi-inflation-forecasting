# Data Sources & Attribution

## Processed Data

### Core datasets

| File | Description | Source |
|------|-------------|--------|
| `processed/bangladesh_MASTER_dataset.csv` | Master dataset: monthly CPI + 10 macroeconomic drivers + 3 structural-break dummies, January 2000 - May 2026 (317 obs) | Compiled from sources below |
| `processed/bangladesh_cpi_monthly_RESEARCH_READY.csv` | Monthly Bangladesh CPI only (2000-2026), research-ready format | IMF IFS / BBS |

### Model results (produced by Sessions 1-4)

| File | Description | Produced by |
|------|-------------|-------------|
| `processed/all_model_results.csv` | Full 12-model scoreboard with R2, RMSE, MAE, MAPE (Table VI in the paper) | Sessions 1-3 |
| `processed/FINAL_scoreboard.csv` | Same 12-model results, ranked by RMSE | Session 3 |
| `processed/FINAL_forecast_2026_2027.csv` | 19-month deployment forecast with 95% CIs and implied inflation (Table XIV) | Session 3 |
| `processed/lasso_selected_features.csv` | LASSO-retained features with standardised coefficients (Table XIII) | Session 1 |
| `processed/session1_orders.json` | AIC-selected ARIMA and SARIMA orders | Session 1 |
| `processed/multiseed_results.csv` | 10-seed RMSE for LSTM, Transformer, SARIMA-LSTM (Tables VIII-IX) | Session 4 |

### Error vectors (inputs for Diebold-Mariano tests)

| File | Description |
|------|-------------|
| `processed/errors_s1.npz` | Forecast error vectors from Session 1 (statistical + ML models) |
| `processed/errors_s2.npz` | Forecast error vectors from Session 2 (deep learning models) |
| `processed/errors_s3.npz` | Forecast error vectors from Session 3 (hybrid models) |

### Session 5: Leakage ablation results

| File | Description |
|------|-------------|
| `processed/session5_leakage_ablation_results.csv` | Initial causal vs full-series comparison |
| `processed/session5b_warmup_sensitivity.csv` | VMD warm-up sensitivity sweep |
| `processed/session5c_ablation_results.csv` | Matched pipeline ablation (reproduces Table VI) |
| `processed/session5d_multiseed_ablation.csv` | 10-seed leakage premium (Table XVII) |
| `processed/session5_error_vectors.npz` | Error vectors for Session 5 DM tests |
| `processed/session5c_error_vectors.npz` | Error vectors for Session 5c DM tests |

### Session 6: Cross-country replication and hyperparameter grid

| File | Description |
|------|-------------|
| `processed/session6_grid_LSTM.csv` | 27-configuration LSTM grid search results (Table XVIII) |
| `processed/session6_grid_Transformer.csv` | 6-configuration Transformer grid search results |
| `processed/session6_grid_summary.json` | Best vs paper configuration comparison |

---

## Raw Data (included)

| File | Description | Source |
|------|-------------|--------|
| `raw/DFF.csv` | US Federal Funds Rate (daily) | [FRED](https://fred.stlouisfed.org/series/DFF) |
| `raw/POILBREUSDM.csv` | Brent Crude Oil Price (USD/barrel, monthly) | [FRED](https://fred.stlouisfed.org/series/POILBREUSDM) |
| `raw/IQ12260.csv` | Exchange Rate data | IMF International Financial Statistics |
| `raw/food_price_indices_data.csv` | FAO Food Price Indices | [FAO](https://www.fao.org/worldfoodsituation/foodpricesindex/en/) |
| `raw/time_series_data1972-2024.xlsx` | Bangladesh historical time series (1972-2024) | Bangladesh Bureau of Statistics |

---

## Large Raw IMF Datasets (NOT included)

The following large raw datasets were used to construct the master dataset
but are excluded from the repository due to file size constraints (90 MB - 333 MB each).

Download them from the IMF Data Portal: [https://data.imf.org/](https://data.imf.org/)

| Dataset | IMF Code | Description |
|---------|----------|-------------|
| CPI dataset | `IMF.STA_CPI_5.0.0` | Consumer Price Index (all countries) |
| Exchange Rate dataset | `IMF.STA_ER_4.0.1` | Exchange Rate Statistics |
| Monetary & Financial Statistics | `IMF.STA_MFS_DC_8.0.0` | Broad Money, Reserves, etc. |
| Interest Rate dataset | `IMF.STA_IL_13.0.1` | Interest & Lending Rates |

### Download Instructions
1. Go to [https://data.imf.org/](https://data.imf.org/)
2. Navigate to **Data** and search for the dataset by name or code above
3. Filter for **Bangladesh** (Country Code: `BD`)
4. Select the date range: **January 2000 - May 2026** (or latest available)
5. Download as CSV

---

## Data Construction Pipeline

The master dataset (`bangladesh_MASTER_dataset.csv`) was built by:
1. Extracting Bangladesh-specific rows from each raw IMF dataset
2. Aligning all series to monthly frequency (January 2000 base)
3. Engineering dummy variables: `COVID_dummy` (Mar 2020 - Jun 2021), `UkraineWar_dummy` (Feb 2022 onward), `BD_Unrest_dummy` (Jul-Aug 2024)
4. Forward-filling minor gaps (<3 months) for exogenous variables
5. Computing `Inflation_YoY` as 12-month percentage change in CPI

See `notebooks/01_statistical_models.ipynb` for the full data loading and validation pipeline.

---

## License

The **code** in this repository is licensed under the [MIT License](../LICENSE).

The **data** files are subject to the original data providers' terms:
- IMF data: [IMF Terms and Conditions](https://www.imf.org/external/terms.htm)
- FRED data: [FRED Terms of Use](https://fred.stlouisfed.org/legal/)
- FAO data: [FAO Open Data License (CC BY-NC-SA 3.0 IGO)](http://www.fao.org/contact-us/terms/en/)
