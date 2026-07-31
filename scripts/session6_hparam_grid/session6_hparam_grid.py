"""
SESSION 6 - DOCUMENTED HYPERPARAMETER SEARCH  (Move 4)

WHY
---
Section VIII concedes that the deep architectures were "tuned by informal search
under compute constraints rather than exhaustive optimisation". That concession
is the single remaining opening for the objection that the deep models lost
because they were under-tuned. This session closes it by running a bounded,
fully documented grid and reporting the result whichever way it falls.

LEAKAGE DISCIPLINE (the point that matters)
-------------------------------------------
Selection uses ONLY training data. The training windows are split
chronologically into a fit portion (first 85%) and a validation portion (last
15%, ending 2020-12-01). Configurations are ranked by validation RMSE. The test
window (2021-01 to 2026-04) is touched exactly once, by the single selected
configuration, after selection is complete. A grid that scored on the test set
would invalidate the very claim this paper makes.

OUTPUT
------
Full grid with validation RMSE per configuration, the selected configuration,
and its ten-seed test RMSE beside the published value.
"""
import os, random, warnings, time, itertools, json
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

FOLDER    = "/content/drive/MyDrive/CPI_Research"
DATA_PATH = f"{FOLDER}/bangladesh_MASTER_dataset.csv"
TRAIN_END, TEST_START, TEST_END = "2020-12-01", "2021-01-01", "2026-04-01"
FEATS = ['CPI','ExchangeRate_BDT_USD','Forex_Reserves_USDmn','BroadMoney_BDTmn',
         'Brent_Oil_USD','Fed_Funds_Rate','Gold_Price_Index','FAO_Food_Index',
         'FAO_Cereals_Index','COVID_dummy','UkraineWar_dummy','BD_Unrest_dummy']

SEL_SEEDS   = [42, 7, 13]                                    # for ranking configs
FINAL_SEEDS = [42, 7, 13, 21, 99, 123, 256, 314, 777, 2024]  # Table IX battery
VAL_FRAC    = 0.15
PUBLISHED   = {"LSTM": 2.6889, "Transformer": 2.7714}
PAPER_CFG   = {"LSTM": dict(units=64, dropout=0.2, lookback=12),
               "Transformer": dict(heads=4, key_dim=8, lookback=12)}

LSTM_GRID = [dict(units=u, dropout=d, lookback=L)
             for u, d, L in itertools.product([32,64,128], [0.1,0.2,0.3], [6,12,24])]
TRF_GRID  = [dict(heads=h, key_dim=k, lookback=L)
             for h, k, L in itertools.product([2,4,8], [8,16], [12])]

def seed_all(s):
    os.environ["PYTHONHASHSEED"]=str(s); random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def build_windows(df, lookback):
    data = df.loc['2002-01-01':TEST_END, FEATS].dropna()
    target, base_s = data['CPI'].diff(), data['CPI'].shift(1)
    sc = StandardScaler().fit(data.loc[:TRAIN_END])          # TRAIN ONLY
    ds = pd.DataFrame(sc.transform(data), index=data.index, columns=FEATS)
    X, y, base, dates = [], [], [], []
    for i in range(lookback, len(ds)):
        if pd.isna(target.iloc[i]): continue
        X.append(ds.iloc[i-lookback:i].values); y.append(target.iloc[i])
        base.append(base_s.iloc[i]); dates.append(ds.index[i])
    X, y, base = np.array(X), np.array(y), np.array(base)
    dates = pd.DatetimeIndex(dates); tr = dates <= pd.Timestamp(TRAIN_END)
    n_tr = int(tr.sum()); n_fit = int(n_tr*(1-VAL_FRAC))
    idx = np.where(tr)[0]
    return dict(X=X, y=y, base=base, dates=dates, tr=tr,
                fit=idx[:n_fit], val=idx[n_fit:], te=np.where(~tr)[0])

def make_lstm(cfg, nf):
    m = keras.Sequential([layers.Input(shape=(cfg['lookback'], nf)),
        layers.LSTM(cfg['units']), layers.Dropout(cfg['dropout']),
        layers.Dense(max(16, cfg['units']//2), activation='relu'), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse'); return m

def make_trf(cfg, nf):
    inp = layers.Input(shape=(cfg['lookback'], nf))
    h = layers.Dense(32)(inp)
    a = layers.MultiHeadAttention(num_heads=cfg['heads'], key_dim=cfg['key_dim'])(h, h)
    h = layers.LayerNormalization()(layers.Add()([h, a]))
    f = layers.Dense(32, activation='relu')(h); f = layers.Dense(32)(f)
    h = layers.LayerNormalization()(layers.Add()([h, f]))
    h = layers.GlobalAveragePooling1D()(h); h = layers.Dropout(0.2)(h)
    m = keras.Model(inp, layers.Dense(1)(h))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse'); return m

def fit_eval(builder, cfg, W, seed, train_idx, eval_idx):
    keras.backend.clear_session(); seed_all(seed)
    m = builder(cfg, len(FEATS))
    m.fit(W['X'][train_idx], W['y'][train_idx], validation_split=0.15,
          epochs=300, batch_size=16, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)])
    pred = W['base'][eval_idx] + m.predict(W['X'][eval_idx], verbose=0).ravel()
    act  = W['base'][eval_idx] + W['y'][eval_idx]
    return float(np.sqrt(np.mean((act-pred)**2)))

def search(name, grid, builder, df):
    print(f"\n{'='*70}\n{name}: {len(grid)} configurations x {len(SEL_SEEDS)} seeds "
          f"(validation only)\n{'='*70}")
    Wc, rows = {}, []
    t0 = time.time()
    for gi, cfg in enumerate(grid, 1):
        L = cfg['lookback']
        if L not in Wc: Wc[L] = build_windows(df, L)
        W = Wc[L]
        v = [fit_eval(builder, cfg, W, s, W['fit'], W['val']) for s in SEL_SEEDS]
        rows.append({**cfg, 'val_rmse_mean': float(np.mean(v)), 'val_rmse_sd': float(np.std(v, ddof=1))})
        print(f"  [{gi:2d}/{len(grid)}] {cfg}  val={np.mean(v):.4f}")
    tab = pd.DataFrame(rows).sort_values('val_rmse_mean').reset_index(drop=True)
    print(f"\nsearch time {(time.time()-t0)/60:.1f} min")
    print(tab.head(5).to_string(index=False))

    best = {k: tab.iloc[0][k] for k in grid[0]}
    for k in ('units','lookback','heads','key_dim'):
        if k in best: best[k] = int(best[k])
    paper = PAPER_CFG[name]
    print(f"\nselected : {best}")
    print(f"paper    : {paper}")

    W = Wc[best['lookback']] if best['lookback'] in Wc else build_windows(df, best['lookback'])
    sel = [fit_eval(builder, best, W, s, np.where(W['tr'])[0], W['te']) for s in FINAL_SEEDS]
    Wp = Wc.get(paper['lookback']) or build_windows(df, paper['lookback'])
    pap = [fit_eval(builder, paper, Wp, s, np.where(Wp['tr'])[0], Wp['te']) for s in FINAL_SEEDS]
    print(f"\nTEST RMSE over {len(FINAL_SEEDS)} seeds")
    print(f"  grid-selected : {np.mean(sel):.4f} +/- {np.std(sel,ddof=1):.4f}")
    print(f"  paper config  : {np.mean(pap):.4f} +/- {np.std(pap,ddof=1):.4f}  "
          f"(published single seed {PUBLISHED[name]})")
    print(f"  SARIMA champion: 1.3053")
    tab.to_csv(f"{FOLDER}/session6_grid_{name}.csv", index=False)
    return dict(model=name, best=best, paper=paper,
                sel_mean=float(np.mean(sel)), sel_sd=float(np.std(sel,ddof=1)),
                pap_mean=float(np.mean(pap)), pap_sd=float(np.std(pap,ddof=1)))

def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    W = build_windows(df, 12)
    assert len(W['te'])==64, f"test window must be 64, got {len(W['te'])}"
    assert W['dates'][W['val']][-1] < W['dates'][W['te']][0], "VALIDATION LEAKS INTO TEST"
    print(f"leakage check passed: validation ends {W['dates'][W['val']][-1].date()}, "
          f"test starts {W['dates'][W['te']][0].date()}")
    out = [search('LSTM', LSTM_GRID, make_lstm, df),
           search('Transformer', TRF_GRID, make_trf, df)]
    json.dump(out, open(f"{FOLDER}/session6_grid_summary.json","w"), indent=1)
    print(f"\nsaved session6_grid_summary.json and per-model CSVs")

if __name__ == "__main__":
    main()
