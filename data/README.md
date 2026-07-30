# Data Sources & Attribution

## Processed Data (Included in Repository)

| File | Description | Source |
|------|-------------|--------|
| `processed/bangladesh_cpi_monthly_RESEARCH_READY.csv` | Monthly Bangladesh CPI (2000–2026), research-ready | IMF IFS + BBS |
| `processed/bangladesh_MASTER_dataset.csv` | Master dataset combining CPI + all macroeconomic variables | Compiled from multiple sources |
| `raw/DFF.csv` | US Federal Funds Rate (daily) | [FRED, Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/DFF) |
| `raw/POILBREUSDM.csv` | Brent Crude Oil Price (USD/barrel, monthly) | [FRED, Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/POILBREUSDM) |
| `raw/IQ12260.csv` | Exchange Rate data | IMF International Financial Statistics |
| `raw/food_price_indices_data.csv` | FAO Food Price Indices | [FAO Food Price Index](https://www.fao.org/worldfoodsituation/foodpricesindex/en/) |
| `raw/time_series_data1972-2024.xlsx` | Bangladesh long-run historical time series (1972–2024) | Bangladesh Bureau of Statistics (BBS) |

---

## Large Raw IMF Datasets (NOT included — too large for GitHub)

The following large raw datasets were used to construct the master dataset
but are excluded from the repository due to file size constraints (90MB–333MB each).

You can download them directly from the **IMF Data Portal**:
👉 [https://data.imf.org/](https://data.imf.org/)

| Dataset | IMF Code | Description |
|---------|----------|-------------|
| CPI dataset | `IMF.STA_CPI_5.0.0` | Consumer Price Index (all countries) |
| Exchange Rate dataset | `IMF.STA_ER_4.0.1` | Exchange Rate Statistics |
| Monetary & Financial Statistics | `IMF.STA_MFS_DC_8.0.0` | Broad Money, Reserves, etc. |
| Interest Rate dataset | `IMF.STA_IL_13.0.1` | Interest & Lending Rates |

### Download Instructions
1. Go to [https://data.imf.org/](https://data.imf.org/)
2. Navigate to **Data** → search for the dataset by name or code above
3. Filter for **Bangladesh** (Country Code: `BD`)
4. Select the date range: **January 2000 – May 2026** (or latest available)
5. Download as CSV

---

## Data Construction Pipeline

The master dataset (`bangladesh_MASTER_dataset.csv`) was built by:
1. Extracting Bangladesh-specific rows from each raw IMF dataset
2. Aligning all series to monthly frequency (January 2000 base)
3. Engineering dummy variables: `COVID_dummy`, `UkraineWar_dummy`, `BD_Unrest_dummy`
4. Forward-filling minor gaps (<3 months) for exogenous variables
5. Computing `Inflation_from_CPI` as 12-month percentage change in CPI

See `notebooks/01_statistical_models.ipynb` for the full data loading and validation pipeline.

---

## License

The **code** in this repository is licensed under the [MIT License](../LICENSE).

The **data** files are subject to the original data providers' terms:
- IMF data: [IMF Terms and Conditions](https://www.imf.org/external/terms.htm)
- FRED data: [FRED Terms of Use](https://fred.stlouisfed.org/legal/)
- FAO data: [FAO Open Data License (CC BY-NC-SA 3.0 IGO)](http://www.fao.org/contact-us/terms/en/)
