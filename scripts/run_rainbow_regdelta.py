"""Regression-delta study on the ATM rainbow book (user spec 2026-10-05).

One GPU pass over every joint-date pair of the daily-sold uncapped ATM-call
book (k_hi = inf) computes the sold-book delta vector under a family of
assumed surface responses to the spot bump, all through the same
MarginalFactory dgrid seam:

  sm        beta = 0            frozen tables (sticky moneyness)
  ss        strike relabeling   the attribution_ss / book_greeks delta
  h1        in-sample OLS slope of each pillar's fixed-moneyness change on
            the index's own daily return (the regression / min-variance delta)
  h5, h20, h60   the same from 5/20/60-step overlapping changes and returns
  big       daily betas from |ret| >= 1% days only

Betas are estimated per index on the joint-date axis (the attribution's day
pairs), in sample on purpose. The eq/vol/cross lines are re-cut for sm, ss
and h1 (vol = the surface move net of the response the delta assumes).
Caches (gitignored, skip-if-exists):

  data/rainbow_regdelta_betas.npz   beta / r2 per label and index, + ss beta
  data/rainbow_regdelta.csv         per-date deltas, delta P&L, pl, recuts

The notebook rainbow_regression_delta.ipynb reads these plus the two ATM
attribution caches. Run from the repo root (the allocator setting matters, see main()):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/run_rainbow_regdelta.py
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.rainbow import spot_panel, spot_vol_betas, sticky_strike_beta
from vol_pca.rainbow_torch import rainbow_regression_delta

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
HORIZONS = (1, 5, 20, 60)
BIG_MOVE = 0.01
CHUNK = 100


def main():
    out = DATA / "rainbow_regdelta.csv"
    out_b = DATA / "rainbow_regdelta_betas.npz"
    sds = {n: load_surfaces(f"{n}_volSurface.csv") for n in ("SPX", "SX5E", "HSI")}
    d64 = spot_panel(sds).index.values.astype("datetime64[D]")
    betas, arrays = {}, {}
    specs = [(f"h{h}", dict(horizon=h)) for h in HORIZONS]
    specs.append(("big", dict(horizon=1, min_abs_ret=BIG_MOVE)))
    for label, kw in specs:
        betas[label] = {}
        for n, sd in sds.items():
            b = spot_vol_betas(sd, d64, **kw)
            betas[label][n] = b.beta
            arrays[f"beta_{label}_{n}"] = b.beta
            arrays[f"r2_{label}_{n}"] = b.r2
            arrays[f"n_{label}_{n}"] = np.array(b.n)
    for n, sd in sds.items():
        arrays[f"beta_ss_{n}"] = sticky_strike_beta(sd, d64)
    DATA.mkdir(exist_ok=True)
    np.savez(out_b, **arrays)
    print(f"wrote {out_b}")
    if out.exists():
        print(f"{out} exists - delete it to recompute")
        return
    # chunked so a crash keeps its progress; the allocator is emptied between
    # chunks (mixed B / 2B / 6B batches fragment the caching allocator and
    # the run crawled at the 8GB ceiling without this + expandable segments)
    import torch
    t0 = time.time()
    parts, n = [], len(d64)
    for lo in range(1, n, CHUNK):
        res = rainbow_regression_delta(sds, betas, recut=("sm", "ss", "h1"),
                                       k_hi=np.inf, t_slice=(lo, min(lo + CHUNK, n)))
        parts.append(res)
        torch.cuda.empty_cache()
        print(f"  {min(lo + CHUNK, n) - 1}/{n - 1} dates, {time.time() - t0:.0f}s, "
              f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)
    res = pd.concat(parts)
    print(f"{len(res)} date pairs in {time.time() - t0:.0f}s")
    res.to_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
