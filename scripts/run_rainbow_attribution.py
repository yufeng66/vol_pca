"""Full-history P&L attribution of the daily-sold rainbow book.

One GPU pass with the CRN Sobol engine (2,048 calibrated paths per vintage
slot) over every joint-quote date pair; caches the per-date book-level
component table. Rerun deletes nothing: remove the CSV to force a recompute.

Run from the repo root: uv run python scripts/run_rainbow_attribution.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.rainbow_torch import rainbow_attribution

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "rainbow_attribution.csv"


def main():
    if OUT.exists():
        print(f"{OUT} exists - delete it to recompute")
        return
    sds = {n: load_surfaces(f"{n}_volSurface.csv") for n in ("SPX", "SX5E", "HSI")}
    t0 = time.time()
    res = rainbow_attribution(sds, progress=250)
    print(f"{len(res)} date pairs in {time.time() - t0:.0f}s")
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
