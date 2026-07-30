"""End-to-end: load surfaces, simulate the call-spread book, attribute P&L,
fit vol-surface PCA, and compare factor-based vega estimates against the full
per-option revaluation. Saves daily results to data/."""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca import (estimate_metrics, factor_vega_estimates, fit_pca,
                     load_surfaces, simulate_book)

MAX_GAP_DAYS = 7   # exclude month-long data-gap "days" from PCA fit and stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="SPX_volSurface 2.csv")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    sd = load_surfaces(args.csv)
    print(f"surfaces: {len(sd)} dates {sd.dates[0]} -> {sd.dates[-1]}  "
          f"(dropped {sd.n_dupes_dropped} dup rows, repaired {sd.n_wing_repaired} wing rows)")

    res, bucket_vega, dsigma = simulate_book(sd)
    normal = (res["gap_days"] <= MAX_GAP_DAYS).to_numpy()
    print(f"book: {res['n_spreads'].iloc[-1]} spreads at end, "
          f"steady-state median {int(res['n_spreads'].tail(2000).median())}; "
          f"{len(res)} P&L days ({(~normal).sum()} gap days excluded from stats)")

    print("\n=== P&L attribution ($, book of ~$250M notional) ===")
    cols = ["pl", "pl_theta", "pl_delta", "pl_gamma", "pl_vega_full", "pl_vanna", "pl_resid"]
    summ = res.loc[normal, cols].agg(["sum", "std"]).T
    summ.columns = ["total", "daily_std"]
    print(summ.round(0).to_string())

    lin_err = np.abs(res["pl_vega_lin"].to_numpy() - (bucket_vega * dsigma).sum(axis=1))
    print(f"\nconsistency: max |pl_vega_lin - bucket_vega . dsigma| = {lin_err.max():.2e} $")

    # headline model: pillars weighted by the book's average absolute vega,
    # so factors chase dollar-relevant surface movement, not wing noise
    vega_w = np.abs(bucket_vega[normal]).mean(axis=0)
    model = fit_pca(dsigma, fit_mask=normal, weights=vega_w)
    print("\n=== vega-weighted PCA of daily surface moves (fit on normal days) ===")
    print("explained variance ($-weighted): " + "  ".join(
        f"PC{i+1} {v:.1%}" for i, v in enumerate(model.evr[:6])) +
        f"  | top3 {model.evr[:3].sum():.1%} top5 {model.evr[:5].sum():.1%}")

    ks = [1, 2, 3, 5, 10, model.n_components]
    estimates, exposures, scores = factor_vega_estimates(bucket_vega, dsigma, model, ks)
    target = res["pl_vega_full"].to_numpy()
    est_named = {f"vega-PCA k={k}" if k < model.n_components else "all factors (=linear)": v
                 for k, v in estimates.items()}
    for name, wt in [("corr", "corr"), ("cov", "cov")]:
        alt, _, _ = factor_vega_estimates(
            bucket_vega, dsigma, fit_pca(dsigma, fit_mask=normal, weights=wt), [3])
        est_named[f"{name}-PCA k=3"] = alt[3]
    print("\n=== factor vega estimate vs full per-option revaluation ===")
    print(estimate_metrics(target, est_named, mask=normal).round(4).to_string())

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    res.to_csv(out / "daily_results.csv")
    np.savez_compressed(out / "factor_model.npz",
                        dates=res.index.to_numpy(), bucket_vega=bucket_vega,
                        dsigma=dsigma, mu=model.mu, weights=model.weights,
                        components=model.components, evr=model.evr,
                        exposures=exposures, scores=scores, normal=normal)
    print(f"\nsaved {out/'daily_results.csv'} and {out/'factor_model.npz'}")


if __name__ == "__main__":
    main()
