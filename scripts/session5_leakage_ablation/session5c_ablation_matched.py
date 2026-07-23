"""
SESSION 5c - LEAKAGE ABLATION, REBUILT ON THE SESSION 3 PIPELINE

WHY THIS SUPERSEDES SESSION 5
-----------------------------
Session 5 wrote its own downstream pipeline and its causal VMD arm scored
3.4961 against the published 2.6349. Inspection of Session 3 showed the causal
decomposition there is strictly correct - VMD is called per month on vals[:i+1],
u[:, -1] is stored, and the warm-up is dropped, never backfilled. There is no
leak. The gap came from the downstream model, not the decomposition:

    Session 3 trains the VMD-LSTM on RAW delta-CPI.
    Session 5 standardised the target before training.

Same loss and optimiser, different loss surface. So Session 5's causal arm was
a different (and weaker) model, which made its 47.5% premium unreliable.

This script fixes that by lifting the Session 3 code verbatim - same windowing,
same architecture, same unscaled VMD target, same clear_session discipline -
and changing exactly one thing per pair: whether the decomposition is computed
causally or once over the full series. Warm-up rows are dropped identically in
both arms so the training samples match.

SUCCESS CRITERION
-----------------
The causal arms must reproduce the published values:
    Wavelet-SARIMA-LSTM  1.6227
    VMD-LSTM             2.6349
If they do, the full-series arms give a leakage premium measured against the
paper's own numbers, and Table XVI can cite Table VI directly.
"""

import os, random, warnings
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

# ============================== CONFIG =======================================
FOLDER    = "/content/drive/MyDrive/CPI_Research"
DATA_PATH = f"{FOLDER}/bangladesh_MASTER_dataset.csv"
DATE_COL, CPI_COL = "Date", "CPI"

TRAIN_END  = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
TEST_END   = pd.Timestamp("2026-04-30")

SEED = 42
LOOKBACK = 12
SARIMA_ORDER, SARIMA_SORDER = (2, 1, 2), (0, 1, 1, 12)
WAVE_WARMUP = 24
K, VMD_WARMUP = 4, 48

PUBLISHED = {"Wavelet-SARIMA-LSTM": 1.6227, "VMD-LSTM": 2.6349}


def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)


# ---------- Session 3 helpers, reproduced verbatim ---------------------------
def make_windows(series):
    vals = series.values
    X, y, dates = [], [], []
    for i in range(LOOKBACK, len(vals)):
        X.append(vals[i - LOOKBACK:i].reshape(-1, 1))
        y.append(vals[i])
        dates.append(series.index[i])
    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


def small_lstm(n_features=1):
    m = keras.Sequential([
        layers.Input(shape=(LOOKBACK, n_features)),
        layers.LSTM(32), layers.Dropout(0.2),
        layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def fit_stat_model(train, order, sorder=None):
    return SARIMAX(train, order=order, seasonal_order=sorder or (0, 0, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False
                   ).fit(disp=False)


def fit_keras(model, X, y):
    model.fit(X, y, validation_split=0.15, epochs=300, batch_size=16,
              callbacks=[keras.callbacks.EarlyStopping(
                  patience=25, restore_best_weights=True)], verbose=0)
    return model


def metrics(y_true, y_pred):
    e = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return {"R2": 1 - np.sum(e ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
            "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
            "MAPE": 100 * np.mean(np.abs(e / y_true))}


def diebold_mariano(e1, e2, h=1):
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    T = len(d); dbar = d.mean()
    lrv = np.sum((d - dbar) ** 2) / T
    if lrv <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(lrv / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dm, 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))


# ---------- decompositions: the ONLY thing that differs ----------------------
def wavelet_smooth(series, causal):
    """Session 3's rolling_wavelet_smooth, plus a full-series counterpart."""
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
        full = rec_last(vals)                       # one pass over everything
        smooth[WAVE_WARMUP:] = full[WAVE_WARMUP:]   # drop the same warm-up rows
    return pd.Series(smooth, index=series.index)


def vmd_modes(series, causal):
    """Session 3's causal VMD loop, plus a full-series counterpart."""
    vals = series.values
    rows = np.full((len(vals), K), np.nan)
    if causal:
        for i in range(VMD_WARMUP, len(vals)):
            u, _, _ = VMD(vals[:i + 1], alpha=2000, tau=0., K=K,
                          DC=0, init=1, tol=1e-7)
            rows[i] = u[:, -1]
    else:
        u, _, _ = VMD(vals, alpha=2000, tau=0., K=K, DC=0, init=1, tol=1e-7)
        L = u.shape[1]
        rows[len(vals) - L:] = u.T                  # right-align if truncated
        rows[:VMD_WARMUP] = np.nan                  # drop the same warm-up rows
    return pd.DataFrame(rows, index=series.index,
                        columns=[f"IMF{k+1}" for k in range(K)]).dropna()


# ---------- arms -------------------------------------------------------------
def wavelet_arm(y_full, y_test, causal):
    smooth = wavelet_smooth(y_full, causal).dropna()
    detail = (y_full - smooth).dropna()

    res_s = fit_stat_model(smooth.loc[:TRAIN_END], SARIMA_ORDER, SARIMA_SORDER)
    ext_s = res_s.append(smooth.loc[TEST_START:], refit=False)
    smooth_pred = ext_s.get_prediction(start=TEST_START).predicted_mean

    sc_d = StandardScaler().fit(detail.loc[:TRAIN_END].values.reshape(-1, 1))
    d_s = pd.Series(sc_d.transform(detail.values.reshape(-1, 1)).ravel(),
                    index=detail.index)
    Xd, yd, dd = make_windows(d_s)
    trd = dd <= TRAIN_END

    keras.backend.clear_session(); seed_everything()
    md = fit_keras(small_lstm(1), Xd[trd], yd[trd])
    d_pred = sc_d.inverse_transform(md.predict(Xd[~trd], verbose=0)).ravel()

    common = smooth_pred.index.intersection(dd[~trd])
    pred = smooth_pred.loc[common] + pd.Series(d_pred, index=dd[~trd]).loc[common]
    return y_test.loc[common].values, pred.values


def vmd_arm(y_full, y_test, causal):
    imfs = vmd_modes(y_full, causal)
    target = y_full.diff()
    base_series = y_full.shift(1)

    sc_i = StandardScaler().fit(imfs.loc[:TRAIN_END])
    imfs_s = pd.DataFrame(sc_i.transform(imfs), index=imfs.index,
                          columns=imfs.columns)

    X, y, base, dates = [], [], [], []
    for i in range(LOOKBACK, len(imfs_s)):
        d = imfs_s.index[i]
        if pd.isna(target.loc[d]):
            continue
        X.append(imfs_s.iloc[i - LOOKBACK:i].values)
        y.append(target.loc[d])            # RAW delta-CPI, as in Session 3
        base.append(base_series.loc[d]); dates.append(d)
    X, y, base = np.array(X), np.array(y), np.array(base)
    dates = pd.DatetimeIndex(dates)
    tr = dates <= TRAIN_END

    keras.backend.clear_session(); seed_everything()
    mv = fit_keras(small_lstm(K), X[tr], y[tr])
    pred = base[~tr] + mv.predict(X[~tr], verbose=0).ravel()
    return y_test.loc[dates[~tr]].values, pred


# ================================ MAIN =======================================
def main():
    seed_everything()
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    df = df[df[DATE_COL] <= TEST_END].reset_index(drop=True)
    y_full = pd.Series(df[CPI_COL].astype(float).values,
                       index=pd.DatetimeIndex(df[DATE_COL]))
    y_test = y_full.loc[TEST_START:]
    print(f"observations={len(y_full)}  test={len(y_test)}")
    assert len(y_full) == 316 and len(y_test) == 64

    rows, errs = [], {}
    for name, arm in [("Wavelet-SARIMA-LSTM", wavelet_arm), ("VMD-LSTM", vmd_arm)]:
        for causal in (True, False):
            tag = "causal" if causal else "full-series"
            yt, yp = arm(y_full, y_test, causal)
            m = metrics(yt, yp)
            errs[f"{name}|{tag}"] = yt - yp
            rows.append((name, tag, len(yt), m))
            print(f"  {name:<22}{tag:<13}n={len(yt):<4}RMSE={m['RMSE']:.4f}")

    print("\n" + "=" * 78)
    print(f"{'Model':<24}{'Decomposition':<15}{'R2':>8}{'RMSE':>9}{'MAE':>9}{'MAPE':>8}")
    print("=" * 78)
    for n_, t, _, m in rows:
        print(f"{n_:<24}{t:<15}{m['R2']:>8.4f}{m['RMSE']:>9.4f}"
              f"{m['MAE']:>9.4f}{m['MAPE']:>8.3f}")

    print("\n--- REPRODUCTION CHECK (causal arms vs published Table VI) ---")
    ok = True
    for n_, t, _, m in rows:
        if t == "causal":
            d = m["RMSE"] - PUBLISHED[n_]
            good = abs(d) < 0.01
            ok &= good
            print(f"  {n_:<22} run={m['RMSE']:.4f}  published={PUBLISHED[n_]:.4f}  "
                  f"diff={d:+.4f}  [{'MATCH' if good else 'DEVIATES'}]")

    print("\n--- LEAKAGE PREMIUM ---")
    for n_ in ("Wavelet-SARIMA-LSTM", "VMD-LSTM"):
        c = next(m["RMSE"] for a, t, _, m in rows if a == n_ and t == "causal")
        f = next(m["RMSE"] for a, t, _, m in rows if a == n_ and t == "full-series")
        dm, p = diebold_mariano(errs[f"{n_}|full-series"], errs[f"{n_}|causal"])
        print(f"  {n_:<22} causal={c:.4f}  full-series={f:.4f}  "
              f"premium={(c-f)/c*100:+.1f}%   DM={dm:+.3f}  p={p:.4f}")

    print("\n" + ("Causal arms reproduce Table VI. The premium is measured against\n"
                  "the paper's own numbers and Table XVI can cite Table VI directly."
                  if ok else
                  "Causal arms still deviate. Send this output before we write anything."))

    pd.DataFrame([{"model": n_, "decomposition": t, "n": k, **m}
                  for n_, t, k, m in rows]).to_csv(
        f"{FOLDER}/session5c_ablation_results.csv", index=False)
    np.savez(f"{FOLDER}/session5c_error_vectors.npz", **errs)
    print(f"\nsaved: {FOLDER}/session5c_ablation_results.csv")


if __name__ == "__main__":
    main()
