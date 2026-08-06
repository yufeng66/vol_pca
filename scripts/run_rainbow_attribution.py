"""Full-history P&L attribution of the daily-sold rainbow book.

Two GPU passes with the CRN Sobol engine over every joint-quote date pair:
the sticky-moneyness driver (2,048 calibrated paths) and the sticky-strike
re-cut (512-path book standard, incl. the bump delta/gamma/cross-gamma
attribution of the equity move). Each pass caches its per-date book-level
component table and is skipped if its CSV exists; remove a CSV to force a
recompute.

Run from the repo root: uv run python scripts/run_rainbow_attribution.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.rainbow_torch import rainbow_attribution, rainbow_attribution_ss

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
PASSES = [(DATA / "rainbow_attribution.csv", rainbow_attribution),
          (DATA / "rainbow_attribution_ss.csv", rainbow_attribution_ss)]


def main():
    sds = None
    for out, driver in PASSES:
        if out.exists():
            print(f"{out} exists - delete it to recompute")
            continue
        if sds is None:
            sds = {n: load_surfaces(f"{n}_volSurface.csv")
                   for n in ("SPX", "SX5E", "HSI")}
        t0 = time.time()
        res = driver(sds, progress=250)
        print(f"{driver.__name__}: {len(res)} date pairs in "
              f"{time.time() - t0:.0f}s")
        out.parent.mkdir(exist_ok=True)
        res.to_csv(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
