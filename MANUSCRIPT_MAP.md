# Manuscript to repository map

| Manuscript item | Produced by | Result file |
| --- | --- | --- |
| Table 6 (main tournament) | `notebooks/01`, `02`, `03` | `data/processed/all_model_results.csv` |
| Table 7 (Diebold-Mariano) | `notebooks/01`, `04` | `data/processed/errors_s1.npz`, `errors_s2.npz`, `errors_s3.npz` |
| Tables 8-9 (multi-seed) | `notebooks/04` | `data/processed/multiseed_results.csv` |
| Table 16 (six economies) | `notebooks/06` | `data/processed/move2_results.json`, `move2_results.csv` |
| Table 17 (leakage ablation) | `scripts/session5_leakage_ablation/session5d_multiseed_ablation.py` | `data/processed/session5d_multiseed_ablation.csv` |
| Table 18 (hyperparameter grid) | `scripts/session6_hparam_grid/session6_hparam_grid.py` | `data/processed/session6_grid_LSTM.csv`, `session6_grid_Transformer.csv`, `session6_grid_summary.json` |
| Table 19 (model confidence set) | `notebooks/04` + `arch` package | computed from the three `errors_s*.npz` files |
| Deployment forecast | `notebooks/03` | `data/processed/FINAL_forecast_2026_2027.csv` |
