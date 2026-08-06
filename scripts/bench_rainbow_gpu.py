"""Throughput of the batched torch quadrature vs the numpy engine.

Two future-shaped workloads:
- book: one as-of date's daily-sold rainbow book — 250 seasoned vintages,
  each its own marginal triple / discount / seasoning factors.
- sweep: a VaR-shaped pass — the same book under 8 spot scenarios as 2,000
  independent problems (tables honestly replicated; a real scenario set
  would rebuild marginals per scenario, same shapes and cost on device).

Run from the repo root: uv run python scripts/bench_rainbow_gpu.py
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.rainbow import (date_index, historical_corr, implied_marginal,
                             price_rainbow_quad, spot_panel)
from vol_pca.rainbow_torch import (QuadBatch, QuadProblem, price_quad_batch,
                                   price_sobol_batch, sobol_normals)

N_BOOK, N_SCEN = 250, 8


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def main():
    names = ("SPX", "SX5E", "HSI")
    sds = {n: load_surfaces(f"{n}_volSurface.csv") for n in names}
    spots = spot_panel(sds)
    val = spots.index[-1]
    idx = {n: date_index(sds[n], np.datetime64(val, "D")) for n in names}
    corr = historical_corr(spots, 5)

    t0 = time.time()
    probs = []
    for lag in range(N_BOOK):
        sale = spots.index[-1 - lag]
        tau = ((sale + pd.Timedelta(days=365)) - val).days / 365.0
        g = tuple((spots.loc[val] / spots.loc[sale]).to_numpy())
        mrg = tuple(implied_marginal(sds[n], idx[n], tau) for n in names)
        probs.append(QuadProblem(mrg, corr, float(sds["SPX"].discount(idx["SPX"], tau)), g))
    print(f"marginal tables for {N_BOOK} vintages: {time.time() - t0:.1f}s (CPU, once per as-of)")

    t0 = time.time()
    pv_np = np.array([price_rainbow_quad(list(p.marginals), p.corr, p.df,
                                         perf_to_date=p.perf_to_date)["pv"]
                      for p in probs])
    t_np = time.time() - t0
    print(f"numpy loop        : {t_np:6.1f}s  {1e3 * t_np / N_BOOK:6.1f} ms/price")

    for dev in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        t0 = time.time()
        batch = QuadBatch.from_problems(probs, device=dev)
        sync(dev)
        t_up = time.time() - t0
        price_quad_batch(batch)          # warmup (context, allocator)
        sync(dev)
        t0 = time.time()
        pv = price_quad_batch(batch)
        sync(dev)
        t_run = time.time() - t0
        d = np.abs(pv.cpu().numpy() - pv_np).max()
        print(f"torch {dev:4s} book  : {t_run:6.2f}s  {1e3 * t_run / N_BOOK:6.1f} ms/price"
              f"  ({t_np / t_run:5.1f}x numpy, stack+transfer {t_up:.2f}s,"
              f" max|diff| ${d:.4f})")

    if torch.cuda.is_available():
        rng = np.random.default_rng(0)
        sweep = [QuadProblem(p.marginals, p.corr, p.df,
                             tuple(np.asarray(p.perf_to_date) * (1.0 + 0.02 * z)))
                 for p in probs for z in rng.standard_normal(N_SCEN)]
        t0 = time.time()
        batch = QuadBatch.from_problems(sweep, device="cuda")
        sync("cuda")
        t_up = time.time() - t0
        t0 = time.time()
        pv = price_quad_batch(batch)
        sync("cuda")
        t_run = time.time() - t0
        n = len(sweep)
        print(f"torch cuda sweep : {t_run:6.2f}s  {1e3 * t_run / n:6.2f} ms/price"
              f"  ({n} prices, {n / t_run:,.0f}/s, stack+transfer {t_up:.2f}s)")
        print(f"  -> 3,138-scenario x {N_BOOK}-vintage full reval ~ "
              f"{3138 * N_BOOK * t_run / n / 60:.0f} min on device")

        batch = QuadBatch.from_problems(probs, device="cuda")
        batch.g.requires_grad_(True)
        sync("cuda")
        t0 = time.time()
        pv = price_quad_batch(batch)
        pv.sum().backward()
        sync("cuda")
        print(f"book + all spot deltas (autograd backward): {time.time() - t0:.2f}s"
              f"  ({3 * N_BOOK} deltas)")

        # Sobol sampler at the calibrated 2,048 paths, one scramble per slot
        zsets = sobol_normals(N_BOOK, 2048, seed=1, device="cuda")
        sbatch = QuadBatch.from_problems(probs, device="cuda", scores=False)
        price_sobol_batch(sbatch, zsets)
        sync("cuda")
        t0 = time.time()
        for _ in range(50):
            price_sobol_batch(sbatch, zsets)
        sync("cuda")
        ms = 1e3 * (time.time() - t0) / 50
        print(f"sobol cuda book @2048 paths: {ms:.2f} ms/book = "
              f"{1e3 * ms / N_BOOK:.0f} us/price (numpy sampler ~1.0 ms/price/core)")


if __name__ == "__main__":
    main()
