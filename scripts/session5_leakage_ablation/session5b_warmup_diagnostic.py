"""
SESSION 5b - WARM-UP SENSITIVITY DIAGNOSTIC

WHY THIS EXISTS
---------------
The Session-5 causal VMD arm returned RMSE 3.4961 against the published
Table VI value of 2.6349 (+32.7%). The wavelet arm, on the identical code
path, reproduced to within 1.8%. The difference between the two arms is the
warm-up length: 24 months for wavelet, 48 for VMD.

If training length drives the deviation, causal VMD RMSE should move
systematically as the warm-up shrinks and the training window grows. If it
does not move, the cause is the causal edge handling instead, and Session 3
has to be inspected directly.

This matters because the headline leakage premium is 47.5% against the
Session-5 causal baseline but 30.3% against the published one, and we should
not publish either number until we know which baseline is right.

RUN IN THE SAME COLAB SESSION, AFTER session5_leakage_ablation.py.
Takes roughly as long as the VMD half of the main run, times four.
"""

import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("s5", "session5_leakage_ablation.py")
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)

PUBLISHED_VMD = 2.6349
SESSION5_VMD = 3.4961

# ---- load exactly as the main script does -----------------------------------
df = pd.read_csv(s5.DATA_PATH)
df[s5.DATE_COL] = pd.to_datetime(df[s5.DATE_COL])
df = df.sort_values(s5.DATE_COL)
df = df[df[s5.DATE_COL] <= s5.TEST_END].reset_index(drop=True)
cpi = df[s5.CPI_COL].astype(float).values
n_train = int((df[s5.DATE_COL] <= s5.TRAIN_END).sum())
y_test = cpi[n_train:]
assert n_train == 252 and len(y_test) == 64

print("warm-up sensitivity of the causal VMD arm")
print("(full-series arm re-run at each warm-up so the premium stays like-for-like)")
print()
print(f"{'warm-up':>8}{'train mo.':>11}{'causal':>10}{'full-series':>13}"
      f"{'premium':>10}{'vs Table VI':>13}")
print("-" * 65)

rows = []
for wu in (48, 36, 24, 12):
    s5.set_seeds()
    pred_c, _ = s5.run_vmd_arm(s5.vmd_causal(cpi, warmup=wu), cpi, n_train,
                               "causal", start=wu)
    rmse_c = s5.metrics(y_test, pred_c)["RMSE"]

    s5.set_seeds()
    pred_f, _ = s5.run_vmd_arm(s5.vmd_full_series(cpi), cpi, n_train,
                               "full-series", start=wu)
    rmse_f = s5.metrics(y_test, pred_f)["RMSE"]

    prem = (rmse_c - rmse_f) / rmse_c * 100
    delta = (rmse_c - PUBLISHED_VMD) / PUBLISHED_VMD * 100
    rows.append((wu, n_train - wu, rmse_c, rmse_f, prem, delta))
    print(f"{wu:>8}{n_train - wu:>11}{rmse_c:>10.4f}{rmse_f:>13.4f}"
          f"{prem:>9.1f}%{delta:>12.1f}%")

print()
spread = max(r[2] for r in rows) - min(r[2] for r in rows)
closest = min(rows, key=lambda r: abs(r[2] - PUBLISHED_VMD))
print(f"causal RMSE range across warm-ups : {spread:.4f} index points")
print(f"closest to published {PUBLISHED_VMD}      : warm-up {closest[0]} "
      f"({closest[2]:.4f}, {closest[5]:+.1f}%)")
print()
if spread > 0.40:
    print("VERDICT: causal VMD is strongly warm-up sensitive. Training length")
    print("         explains the deviation. Report the ablation at the warm-up")
    print("         whose causal arm is closest to Table VI, and state the")
    print("         effective training window in the caption.")
else:
    print("VERDICT: causal VMD is NOT warm-up sensitive, so training length does")
    print("         not explain the deviation. The cause is the causal edge")
    print("         handling. Open the Session 3 notebook and check three things:")
    print("           1. how many months the VMD-LSTM actually trained on;")
    print("           2. whether the 48-month warm-up was dropped, zero-filled,")
    print("              or BACKFILLED with full-series values;")
    print("           3. which index of the VMD output was taken at each step.")
    print()
    print("         If the answer to (2) is 'backfilled with full-series values',")
    print("         the published pipeline contains a warm-up leak of its own and")
    print("         Table VI's VMD row needs revisiting before submission.")

pd.DataFrame(rows, columns=["warmup", "train_months", "rmse_causal",
                            "rmse_full_series", "premium_pct", "vs_published_pct"]
             ).to_csv(f"{s5.os.path.dirname(s5.DATA_PATH)}/session5b_warmup_sensitivity.csv",
                      index=False)
print()
print("saved: session5b_warmup_sensitivity.csv")
