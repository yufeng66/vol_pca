"""Rolling VaR backtest: full reval vs greeks (k, +/- third-order equity).

Writes data/var_rolling.csv (gitignored, like the other pipeline outputs).
Heavy: ~2,900 as-of dates x 1.6M full revaluations, parallelized over cores.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca import load_surfaces, simulate_book, fit_pca
from vol_pca.factors import sticky_strike_dsigma
from vol_pca.key_rate import rolling_var_keyrate
from vol_pca.var import (rolling_var_adaptive, rolling_var_backtest,
                         rolling_var_bump, strike_histogram_weights)

ROOT = pathlib.Path(__file__).resolve().parents[1]

t0 = time.time()
sd = load_surfaces(ROOT / "SPX_volSurface.csv")
_, bucket_vega_hist, _ = simulate_book(sd)
normal = np.diff(sd.dates).astype(int) <= 7
vega_w = np.abs(bucket_vega_hist[normal]).mean(axis=0)
ds_roll = sticky_strike_dsigma(sd, include_roll=True)
model = fit_pca(ds_roll, fit_mask=normal, weights=vega_w)
(ROOT / "data").mkdir(exist_ok=True)

out = ROOT / "data" / "var_rolling.csv"
if not out.exists():
    roll = rolling_var_backtest(sd, model, ks=(3, 4, 5, 10), start=252,
                                mask=normal)
    roll.to_csv(out)
    print(f"{len(roll)} as-of dates -> {out}  ({time.time() - t0:,.0f}s)")
    print(roll[["full_h", "g3_h", "g3c_h", "g10_h", "g10c_h"]]
          .describe().round(0).to_string())
else:
    print(f"{out} exists, skipping the full-reval pass")

out_s = ROOT / "data" / "var_rolling_strike.csv"
if not out_s.exists():
    t0 = time.time()
    adapt = rolling_var_adaptive(sd, strike_histogram_weights, ds_roll,
                                 fit_mask=normal, ks=(4, 10), start=252,
                                 mask=normal)
    adapt.to_csv(out_s)
    print(f"{len(adapt)} as-of dates -> {out_s}  ({time.time() - t0:,.0f}s)")
else:
    print(f"{out_s} exists, skipping the adaptive pass")

out_b = ROOT / "data" / "var_rolling_bump.csv"
if not out_b.exists():
    t0 = time.time()
    bump = rolling_var_bump(sd, model, start=252, mask=normal)
    bump.to_csv(out_b)
    print(f"{len(bump)} as-of dates -> {out_b}  ({time.time() - t0:,.0f}s)")
else:
    print(f"{out_b} exists, skipping the bump-path pass")

out_k = ROOT / "data" / "var_rolling_keyrate.csv"
if not out_k.exists():
    t0 = time.time()
    kr = rolling_var_keyrate(sd, start=252, mask=normal)
    kr.to_csv(out_k)
    print(f"{len(kr)} as-of dates -> {out_k}  ({time.time() - t0:,.0f}s)")
else:
    print(f"{out_k} exists, skipping the key-rate pass")
