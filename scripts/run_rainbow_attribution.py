"""Full-history P&L attribution of the daily-sold rainbow book.

Three GPU passes with the CRN Sobol engine over every joint-quote date pair:
the sticky-moneyness driver (2,048 calibrated paths), the sticky-strike
re-cut (512-path book standard, incl. the bump delta/gamma/cross-gamma
attribution of the equity move), and the same sticky-strike driver on the
uncapped ATM-call book (k_hi = inf: one $1M 1Y 100-strike ranked-basket call
sold per date instead of the 100/112 spread), and the sticky-moneyness driver
on that ATM book at the same 512 paths/seed (its fixed-moneyness vol line and
autograd delta are the comparison set for rainbow_regression_delta.ipynb and
the vol-trend decomposition in the changelog). Each pass caches its per-date
book-level component table and is skipped if its CSV exists; remove a CSV to
force a recompute.

Run from the repo root: uv run python scripts/run_rainbow_attribution.py
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.rainbow_torch import rainbow_attribution, rainbow_attribution_ss

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
PASSES = [(DATA / "rainbow_attribution.csv", rainbow_attribution, {}),
          (DATA / "rainbow_attribution_ss.csv", rainbow_attribution_ss, {}),
          (DATA / "rainbow_attribution_ss_atm.csv", rainbow_attribution_ss,
           {"k_hi": np.inf}),
          (DATA / "rainbow_attribution_sm_atm.csv", rainbow_attribution,
           {"k_hi": np.inf, "n_paths": 512})]


def main():
    sds = None
    for out, driver, kw in PASSES:
        if out.exists():
            print(f"{out} exists - delete it to recompute")
            continue
        if sds is None:
            sds = {n: load_surfaces(f"{n}_volSurface.csv")
                   for n in ("SPX", "SX5E", "HSI")}
        t0 = time.time()
        res = driver(sds, progress=250, **kw)
        print(f"{out.name}: {len(res)} date pairs in "
              f"{time.time() - t0:.0f}s")
        out.parent.mkdir(exist_ok=True)
        res.to_csv(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
