"""
SESSION 5 - LEAKAGE ABLATION
Bangladesh CPI forecasting: causal vs full-series decomposition.

PURPOSE
-------
Isolates the "leakage premium" in wavelet and variational-mode decomposition
hybrids. Only ONE thing differs between the two arms of each pair: whether the
decomposition is computed causally (month by month, past data only) or once over
the full series (contaminating the training period with test-era information).
Everything else - split, architecture, seeds, scaling, SARIMA orders - is shared
code, so the difference in RMSE is attributable to the decomposition alone.

Produces Table XVI for the manuscript.

USAGE (Google Colab) - run these in order
-----------------------------------------
    # Cell 1  mount Drive (required: DATA_PATH lives there)
    from google.colab import drive
    drive.mount('/content/drive')

    # Cell 2  install the one package Colab lacks
    !pip install vmdpy -q

    # Cell 3  locate the dataset and check column names
    !find /content/drive -name "bangladesh_MASTER_dataset.csv" 2>/dev/null
    import pandas as pd
    print(pd.read_csv("<path printed above>").columns.tolist())

    # Cell 4  set DATA_PATH / DATE_COL / CPI_COL below, then
    !python session5_leakage_ablation.py

Results are written back to the DATA_PATH folder on Drive, so a runtime
disconnect does not destroy them.

WHAT YOU MUST SET
-----------------
    DATA_PATH   path to bangladesh_MASTER_dataset.csv
    DATE_COL    name of the date column
    CPI_COL     name of the CPI column
Everything else is fixed to the values reported in the paper.
"""

import os, random, warnings
import numpy as np
import pandas as pd
import pywt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from vmdpy import VMD

warnings.filterwarnings("ignore")

# ============================== CONFIG =======================================
DATA_PATH = "/content/drive/MyDrive/CPI_Research/bangladesh_MASTER_dataset.csv"  # <-- SET FOLDER
DATE_COL  = "Date"                                                      # <-- SET
CPI_COL   = "CPI"                                                       # <-- SET

TRAIN_END = "2020-12-31"   # 252 training months, Jan 2000 - Dec 2020
TEST_END  = "2026-04-30"   # 64 test months,     Jan 2021 - Apr 2026

SEED        = 42
LOOKBACK    = 12
WAVELET     = "db4"
WAVE_LEVEL  = 2
WAVE_WARMUP = 24
VMD_K       = 4
VMD_ALPHA   = 2000
VMD_WARMUP  = 48
SARIMA_ORDER    = (2, 1, 2)
SARIMA_SEASONAL = (0, 1, 1, 12)

LSTM_UNITS   = 32
DENSE_UNITS  = 16
DROPOUT      = 0.2
BATCH_SIZE   = 16
LR           = 1e-3
VAL_SPLIT    = 0.15
ES_PATIENCE  = 25
MAX_EPOCHS   = 300

# Reference values from the published causal runs (Table VI)
REF = {"Wavelet-SARIMA-LSTM": 1.6227, "VMD-LSTM": 2.6349, "SARIMA": 1.3053}


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ============================== METRICS ======================================
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "R2":   1 - ss_res / ss_tot,
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAE":  np.mean(np.abs(err)),
        "MAPE": 100 * np.mean(np.abs(err / y_true)),
    }


def diebold_mariano(e1, e2, h=1):
    """DM on squared-error loss with Harvey-Leybourne-Newbold small-sample
    correction. Negative statistic favours model 1 (e1)."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    T = len(d)
    d_bar = d.mean()
    gamma0 = np.sum((d - d_bar) ** 2) / T
    lrv = gamma0
    for k in range(1, h):
        gk = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
        lrv += 2 * (1 - k / h) * gk
    if lrv <= 0:
        return np.nan, np.nan
    dm = d_bar / np.sqrt(lrv / T)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=T - 1))
    return dm_hln, p


# ========================= DECOMPOSITIONS ====================================
def _wave_components(series):
    """Return (approximation, detail) reconstructed to the length of `series`."""
    n = len(series)
    coeffs = pywt.wavedec(series, WAVELET, level=WAVE_LEVEL)
    a_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    approx = pywt.waverec(a_coeffs, WAVELET)[:n]
    detail = series - approx
    return approx, detail


def wavelet_full_series(series):
    """LEAKY: decompose the entire series once, then split downstream."""
    return _wave_components(np.asarray(series, float))


def wavelet_causal(series, warmup=WAVE_WARMUP):
    """CAUSAL: at each t, decompose y[:t+1] and keep only the last value."""
    s = np.asarray(series, float)
    n = len(s)
    approx = np.full(n, np.nan)
    detail = np.full(n, np.nan)
    for t in range(warmup, n):
        a, d = _wave_components(s[: t + 1])
        approx[t], detail[t] = a[-1], d[-1]
    return approx, detail


def _vmd_modes(series):
    s = np.asarray(series, float)
    if len(s) % 2:                      # vmdpy requires even length
        s = s[1:]
        pad = 1
    else:
        pad = 0
    u, _, _ = VMD(s, VMD_ALPHA, 0.0, VMD_K, 0, 1, 1e-7)
    if pad:
        u = np.hstack([np.full((VMD_K, 1), np.nan), u])
    return u                            # shape (K, len(series))


def vmd_full_series(series):
    """LEAKY: one VMD over the entire series."""
    return _vmd_modes(series)


def vmd_causal(series, warmup=VMD_WARMUP):
    """CAUSAL: at each t, run VMD on y[:t+1] and keep the last column."""
    s = np.asarray(series, float)
    n = len(s)
    modes = np.full((VMD_K, n), np.nan)
    for t in range(warmup, n):
        u = _vmd_modes(s[: t + 1])
        modes[:, t] = u[:, -1]
    return modes


# ============================ MODEL STAGES ===================================
def make_windows(X, y, lookback=LOOKBACK):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def build_lstm(n_features):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer=Adam(LR), loss="mse")
    return m


def fit_lstm(Xtr, ytr):
    from tensorflow.keras.callbacks import EarlyStopping
    set_seeds()
    m = build_lstm(Xtr.shape[2])
    es = EarlyStopping(patience=ES_PATIENCE, restore_best_weights=True,
                       monitor="val_loss")
    m.fit(Xtr, ytr, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
          validation_split=VAL_SPLIT, callbacks=[es], verbose=0)
    return m


def sarima_rolling(train, test):
    """One-step-ahead forecasts by state-space filtering on the extended
    history without re-estimating parameters (matches the paper's protocol)."""
    fit = SARIMAX(train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False)
    preds = []
    res = fit
    for obs in test:
        preds.append(float(res.forecast(1)[0]))
        res = res.append([obs], refit=False)
    return np.asarray(preds)


# ============================ EXPERIMENT ARMS ================================
def run_wavelet_arm(components, n_train, label, start=WAVE_WARMUP):
    """components: (approx, detail) from the chosen decomposition scheme.

    The leading `start` months are DROPPED from both arms, not zero-filled.
    The causal scheme cannot produce values there, and zero-filling would
    inject a false discontinuity that penalises the causal arm for a coding
    artefact instead of for leakage. Dropping the same rows from both arms
    keeps the comparison exact.
    """
    approx, detail = components
    approx, detail = approx[start:], detail[start:]
    n_tr = n_train - start
    assert not np.isnan(approx).any(), "NaNs remain after warm-up trim"

    # --- SARIMA on the smooth component ---
    a_pred = sarima_rolling(approx[:n_tr], approx[n_tr:])

    # --- LSTM on the detail component ---
    sc = StandardScaler().fit(detail[:n_tr].reshape(-1, 1))
    d_s = sc.transform(detail.reshape(-1, 1))
    X, y = make_windows(d_s, d_s.ravel())
    split = n_tr - LOOKBACK
    m = fit_lstm(X[:split], y[:split])
    d_pred = sc.inverse_transform(m.predict(X[split:], verbose=0)).ravel()

    assert len(a_pred) == len(d_pred), (len(a_pred), len(d_pred))
    return a_pred + d_pred, label


def run_vmd_arm(modes, cpi_levels, n_train, label, start=VMD_WARMUP):
    """Single LSTM over all K mode channels predicting the monthly change.

    As in the wavelet arm, the leading `start` months are dropped from both
    arms rather than zero-filled, so causal and full-series differ only in
    how the modes were computed.
    """
    M = modes[:, start:].T                      # (n, K)
    lv = cpi_levels[start:]
    n_tr = n_train - start
    assert not np.isnan(M).any(), "NaNs remain after warm-up trim"

    dcpi = np.diff(lv, prepend=lv[0])
    scX = StandardScaler().fit(M[:n_tr])
    scy = StandardScaler().fit(dcpi[:n_tr].reshape(-1, 1))
    Xs = scX.transform(M)
    ys = scy.transform(dcpi.reshape(-1, 1)).ravel()

    X, y = make_windows(Xs, ys)
    split = n_tr - LOOKBACK
    m = fit_lstm(X[:split], y[:split])
    dpred = scy.inverse_transform(m.predict(X[split:], verbose=0)).ravel()

    prev = lv[n_tr - 1:-1]
    assert len(prev) == len(dpred), (len(prev), len(dpred))
    return prev + dpred, label


# ================================ MAIN =======================================
def main():
    set_seeds()
    raw = pd.read_csv(DATA_PATH)
    missing = [c for c in (DATE_COL, CPI_COL) if c not in raw.columns]
    if missing:
        raise SystemExit(
            "Column(s) not found: %s\n"
            "Columns actually in %s:\n  %s\n"
            "Set DATE_COL / CPI_COL at the top of this file to match."
            % (missing, os.path.basename(DATA_PATH), "\n  ".join(raw.columns))
        )
    df = raw.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    df = df[df[DATE_COL] <= TEST_END].reset_index(drop=True)
    if df[CPI_COL].isna().any():
        raise SystemExit("CPI column contains %d missing values; the ablation "
                         "needs a complete target series."
                         % int(df[CPI_COL].isna().sum()))
    cpi = df[CPI_COL].astype(float).values
    n_train = int((df[DATE_COL] <= TRAIN_END).sum())
    y_test = cpi[n_train:]

    print(f"observations={len(cpi)}  train={n_train}  test={len(y_test)}")
    assert n_train == 252, f"expected 252 training months, got {n_train}"
    assert len(y_test) == 64, f"expected 64 test months, got {len(y_test)}"

    rows, errs = [], {}

    # ---- SARIMA reference ----
    s_pred = sarima_rolling(cpi[:n_train], y_test)
    errs["SARIMA"] = y_test - s_pred
    rows.append(("SARIMA (reference)", "-", metrics(y_test, s_pred)))

    # ---- Wavelet: causal vs leaky ----
    for scheme, fn in [("causal", wavelet_causal), ("full-series", wavelet_full_series)]:
        pred, _ = run_wavelet_arm(fn(cpi), n_train, scheme)
        errs[f"Wavelet-{scheme}"] = y_test - pred
        rows.append(("Wavelet-SARIMA-LSTM", scheme, metrics(y_test, pred)))

    # ---- VMD: causal vs leaky ----
    for scheme, fn in [("causal", vmd_causal), ("full-series", vmd_full_series)]:
        pred, _ = run_vmd_arm(fn(cpi), cpi, n_train, scheme)
        errs[f"VMD-{scheme}"] = y_test - pred
        rows.append(("VMD-LSTM", scheme, metrics(y_test, pred)))

    # ---- report ----
    print("\n" + "=" * 74)
    print(f"{'Model':<24}{'Decomposition':<16}{'R2':>8}{'RMSE':>9}{'MAE':>9}{'MAPE':>8}")
    print("=" * 74)
    for name, scheme, m in rows:
        print(f"{name:<24}{scheme:<16}{m['R2']:>8.4f}{m['RMSE']:>9.4f}"
              f"{m['MAE']:>9.4f}{m['MAPE']:>8.3f}")

    print("\n--- REPRODUCTION CHECKPOINT (causal arms vs published Table VI) ---")
    for name, scheme, m in rows:
        key = {"Wavelet-SARIMA-LSTM": "Wavelet-SARIMA-LSTM",
               "VMD-LSTM": "VMD-LSTM",
               "SARIMA (reference)": "SARIMA"}.get(name)
        if key in REF and scheme in ("causal", "-"):
            d = m["RMSE"] - REF[key]
            flag = "MATCH" if abs(d) < 0.01 else "DEVIATES"
            print(f"  {key:<22} run={m['RMSE']:.4f}  published={REF[key]:.4f}  "
                  f"diff={d:+.4f}  [{flag}]")

    print("\n--- LEAKAGE PREMIUM (causal RMSE minus full-series RMSE) ---")
    for fam in ["Wavelet", "VMD"]:
        c = next(m["RMSE"] for n_, s, m in rows
                 if n_.startswith(fam.replace("Wavelet", "Wavelet")) and s == "causal")
        f = next(m["RMSE"] for n_, s, m in rows
                 if n_.startswith(fam.replace("Wavelet", "Wavelet")) and s == "full-series")
        print(f"  {fam:<10} causal={c:.4f}  full-series={f:.4f}  "
              f"premium={c - f:+.4f} ({(c - f) / c * 100:+.1f}%)")
        dm, p = diebold_mariano(errs[f"{fam}-full-series"], errs[f"{fam}-causal"])
        print(f"             DM(full-series vs causal) = {dm:+.3f}, p = {p:.4f}")

    # Persist next to the dataset so results survive a runtime disconnect.
    outdir = os.path.dirname(DATA_PATH) or "."
    csv_path = os.path.join(outdir, "session5_leakage_ablation_results.csv")
    npz_path = os.path.join(outdir, "session5_error_vectors.npz")
    out = pd.DataFrame([{"model": n_, "decomposition": s, **m} for n_, s, m in rows])
    out.to_csv(csv_path, index=False)
    np.savez(npz_path, **errs)
    print("\nsaved to Drive:\n  %s\n  %s" % (csv_path, npz_path))


if __name__ == "__main__":
    main()
