"""
SESSION 5d - MULTI-SEED LEAKAGE ABLATION

WHY
---
Session 5c matched the Session 3 pipeline and produced a leakage premium of
+10.4% (wavelet) and +28.6% (VMD). Two questions remain open:

  1. Is the premium an effect or seed noise? A single seed cannot say.
  2. The causal arms landed 1.8% and 6.5% below the published values. Is that
     the GPU nondeterminism already documented in Section VIII, or something
     systematic?

Both are answered by running the ablation over the SAME ten seeds the paper
already uses for its neural battery (Table IX): 42, 7, 13, 21, 99, 123, 256,
314, 777, 2024. If the published value falls inside the causal seed
distribution, question 2 is closed. If the premium holds across seeds, so is 1.

EFFICIENCY
----------
Decompositions and the SARIMA-on-smooth stage do not depend on the seed, so
they are computed once and reused across all ten runs. Only the LSTM stages
are repeated.

OUTPUT
------
Per-seed RMSE for all four arms, mean +/- SD, the premium distribution, a
containment check against Table VI, and a DM test at every seed.
"""

import os, random, warnings, time
import numpy as np
import pandas as pd
import pywt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from vmdpy import VMD
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

FOLDER    = "/content/drive/MyDrive/CPI_Research"
DATA_PATH = f"{FOLDER}/bangladesh_MASTER_dataset.csv"
DATE_COL, CPI_COL = "Date", "CPI"

TRAIN_END  = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
TEST_END   = pd.Timestamp("2026-04-30")

SEEDS = [42, 7, 13, 21, 99, 123, 256, 314, 777, 2024]   # Table IX seed list
LOOKBACK = 12
SARIMA_ORDER, SARIMA_SORDER = (2, 1, 2), (0, 1, 1, 12)
WAVE_WARMUP = 24
K, VMD_WARMUP = 4, 48
PUBLISHED = {"Wavelet-SARIMA-LSTM": 1.6227, "VMD-LSTM": 2.6349}


def seed_everything(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def make_windows(series):
    vals = series.values
    X, y, dates = [], [], []
    for i in range(LOOKBACK, len(vals)):
        X.append(vals[i - LOOKBACK:i].reshape(-1, 1))
        y.append(vals[i]); dates.append(series.index[i])
    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


def small_lstm(n_features=1):
    m = keras.Sequential([
        layers.Input(shape=(LOOKBACK, n_features)),
        layers.LSTM(32), layers.Dropout(0.2),
        layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def fit_keras(model, X, y):
    model.fit(X, y, validation_split=0.15, epochs=300, batch_size=16,
              callbacks=[keras.callbacks.EarlyStopping(
                  patience=25, restore_best_weights=True)], verbose=0)
    return model


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def diebold_mariano(e1, e2, h=1):
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    T = len(d); dbar = d.mean()
    lrv = np.sum((d - dbar) ** 2) / T
    if lrv <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(lrv / T) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dm, 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))


# ---------- decompositions (seed independent) --------------------------------
def wavelet_smooth(series, causal):
    vals = series.values
    smooth = np.full(len(vals), np.nan)

    def rec_last(w):
        c = pywt.wavedec(w, "db4", level=2)
        cs = [c[0]] + [np.zeros_like(x) for x in c[1:]]
        return pywt.waverec(cs, "db4")[:len(w)]

    if causal:
        for i in range(WAVE_WARMUP, len(vals)):
            smooth[i] = rec_last(vals[:i + 1])[-1]
    else:
        full = rec_last(vals)
        smooth[WAVE_WARMUP:] = full[WAVE_WARMUP:]
    return pd.Series(smooth, index=series.index)


def vmd_modes(series, causal):
    vals = series.values
    rows = np.full((len(vals), K), np.nan)
    if causal:
        for i in range(VMD_WARMUP, len(vals)):
            u, _, _ = VMD(vals[:i + 1], alpha=2000, tau=0., K=K,
                          DC=0, init=1, tol=1e-7)
            rows[i] = u[:, -1]
    else:
        u, _, _ = VMD(vals, alpha=2000, tau=0., K=K, DC=0, init=1, tol=1e-7)
        rows[len(vals) - u.shape[1]:] = u.T
        rows[:VMD_WARMUP] = np.nan
    return pd.DataFrame(rows, index=series.index,
                        columns=[f"IMF{k+1}" for k in range(K)]).dropna()


# ---------- seed independent preparation -------------------------------------
def prep_wavelet(y_full, causal):
    smooth = wavelet_smooth(y_full, causal).dropna()
    detail = (y_full - smooth).dropna()
    res = SARIMAX(smooth.loc[:TRAIN_END], order=SARIMA_ORDER,
                  seasonal_order=SARIMA_SORDER, enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    ext = res.append(smooth.loc[TEST_START:], refit=False)
    smooth_pred = ext.get_prediction(start=TEST_START).predicted_mean
    sc = StandardScaler().fit(detail.loc[:TRAIN_END].values.reshape(-1, 1))
    d_s = pd.Series(sc.transform(detail.values.reshape(-1, 1)).ravel(),
                    index=detail.index)
    X, y, dates = make_windows(d_s)
    tr = dates <= TRAIN_END
    return dict(smooth_pred=smooth_pred, sc=sc, X=X, y=y, dates=dates, tr=tr)


def prep_vmd(y_full, causal):
    imfs = vmd_modes(y_full, causal)
    target, base_s = y_full.diff(), y_full.shift(1)
    sc = StandardScaler().fit(imfs.loc[:TRAIN_END])
    imfs_s = pd.DataFrame(sc.transform(imfs), index=imfs.index, columns=imfs.columns)
    X, y, base, dates = [], [], [], []
    for i in range(LOOKBACK, len(imfs_s)):
        d = imfs_s.index[i]
        if pd.isna(target.loc[d]):
            continue
        X.append(imfs_s.iloc[i - LOOKBACK:i].values)
        y.append(target.loc[d]); base.append(base_s.loc[d]); dates.append(d)
    dates = pd.DatetimeIndex(dates)
    return dict(X=np.array(X), y=np.array(y), base=np.array(base),
                dates=dates, tr=dates <= TRAIN_END)


# ---------- one seed ---------------------------------------------------------
def run_wavelet(p, y_test, seed):
    keras.backend.clear_session(); seed_everything(seed)
    m = fit_keras(small_lstm(1), p["X"][p["tr"]], p["y"][p["tr"]])
    d_pred = p["sc"].inverse_transform(m.predict(p["X"][~p["tr"]], verbose=0)).ravel()
    te = p["dates"][~p["tr"]]
    common = p["smooth_pred"].index.intersection(te)
    pred = p["smooth_pred"].loc[common] + pd.Series(d_pred, index=te).loc[common]
    return y_test.loc[common].values, pred.values


def run_vmd(p, y_test, seed):
    keras.backend.clear_session(); seed_everything(seed)
    m = fit_keras(small_lstm(K), p["X"][p["tr"]], p["y"][p["tr"]])
    pred = p["base"][~p["tr"]] + m.predict(p["X"][~p["tr"]], verbose=0).ravel()
    return y_test.loc[p["dates"][~p["tr"]]].values, pred


# ================================ MAIN =======================================
def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    df = df[df[DATE_COL] <= TEST_END].reset_index(drop=True)
    y_full = pd.Series(df[CPI_COL].astype(float).values,
                       index=pd.DatetimeIndex(df[DATE_COL]))
    y_test = y_full.loc[TEST_START:]
    assert len(y_full) == 316 and len(y_test) == 64
    print(f"observations={len(y_full)} test={len(y_test)} seeds={len(SEEDS)}")

    print("precomputing decompositions (seed independent)...")
    preps = {
        ("Wavelet-SARIMA-LSTM", "causal"):      prep_wavelet(y_full, True),
        ("Wavelet-SARIMA-LSTM", "full-series"): prep_wavelet(y_full, False),
        ("VMD-LSTM", "causal"):                 prep_vmd(y_full, True),
        ("VMD-LSTM", "full-series"):            prep_vmd(y_full, False),
    }
    print(f"done in {time.time()-t0:.0f}s\n")

    runner = {"Wavelet-SARIMA-LSTM": run_wavelet, "VMD-LSTM": run_vmd}
    res, errs = {k: [] for k in preps}, {k: {} for k in preps}
    for s in SEEDS:
        line = f"  seed {s:>5}"
        for key, p in preps.items():
            yt, yp = runner[key[0]](p, y_test, s)
            res[key].append(rmse(yt, yp)); errs[key][s] = yt - yp
            line += f"   {key[1][:4]}-{key[0][:3]}={res[key][-1]:.4f}"
        print(line)

    print("\n" + "=" * 74)
    print(f"{'Model':<24}{'Decomp':<14}{'mean':>9}{'SD':>8}{'min':>9}{'max':>9}")
    print("=" * 74)
    for key, v in res.items():
        a = np.array(v)
        print(f"{key[0]:<24}{key[1]:<14}{a.mean():>9.4f}{a.std(ddof=1):>8.4f}"
              f"{a.min():>9.4f}{a.max():>9.4f}")

    print("\n--- IS THE PUBLISHED VALUE INSIDE THE CAUSAL SEED DISTRIBUTION? ---")
    for mdl in ("Wavelet-SARIMA-LSTM", "VMD-LSTM"):
        a = np.array(res[(mdl, "causal")]); pub = PUBLISHED[mdl]
        inside = a.min() <= pub <= a.max()
        z = (pub - a.mean()) / a.std(ddof=1)
        print(f"  {mdl:<22} published={pub:.4f}  range=[{a.min():.4f},{a.max():.4f}]"
              f"  z={z:+.2f}  [{'INSIDE' if inside else 'OUTSIDE'}]")

    print("\n--- LEAKAGE PREMIUM ACROSS SEEDS ---")
    for mdl in ("Wavelet-SARIMA-LSTM", "VMD-LSTM"):
        c = np.array(res[(mdl, "causal")]); f = np.array(res[(mdl, "full-series")])
        prem = (c - f) / c * 100
        sig = sum(1 for s in SEEDS
                  if diebold_mariano(errs[(mdl, "full-series")][s],
                                     errs[(mdl, "causal")][s])[1] < 0.05)
        print(f"  {mdl:<22} premium mean={prem.mean():.1f}%  SD={prem.std(ddof=1):.1f}pp"
              f"  range=[{prem.min():.1f}%,{prem.max():.1f}%]"
              f"  DM p<0.05 in {sig}/{len(SEEDS)} seeds")

    out = pd.DataFrame([{"model": k[0], "decomposition": k[1], "seed": s, "rmse": r}
                        for k, v in res.items() for s, r in zip(SEEDS, v)])
    out.to_csv(f"{FOLDER}/session5d_multiseed_ablation.csv", index=False)
    print(f"\nsaved: {FOLDER}/session5d_multiseed_ablation.csv")
    print(f"total runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
