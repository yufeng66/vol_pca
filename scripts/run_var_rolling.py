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
from vol_pca.var import rolling_var_backtest

ROOT = pathlib.Path(__file__).resolve().parents[1]

t0 = time.time()
sd = load_surfaces(ROOT / "SPX_volSurface 2.csv")
_, bucket_vega_hist, _ = simulate_book(sd)
normal = np.diff(sd.dates).astype(int) <= 7
vega_w = np.abs(bucket_vega_hist[normal]).mean(axis=0)
model = fit_pca(sticky_strike_dsigma(sd, include_roll=True),
                fit_mask=normal, weights=vega_w)

roll = rolling_var_backtest(sd, model, ks=(3, 4, 5, 10), start=252,
                            mask=normal)
out = ROOT / "data" / "var_rolling.csv"
out.parent.mkdir(exist_ok=True)
roll.to_csv(out)
print(f"{len(roll)} as-of dates -> {out}  ({time.time() - t0:,.0f}s)")
print(roll[["full_h", "g3_h", "g3c_h", "g10_h", "g10c_h"]].describe().round(0).to_string())
