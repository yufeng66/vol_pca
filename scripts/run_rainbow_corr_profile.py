"""Smooth correlation profile calibrated to the 21-day history, priced
against the Gaussian baselines (user spec 2026-10-27).

Follow-up to run_rainbow_corr_regime.py. The realized 21-day correlation
profile (normal scores of overlapping 21-day log returns, pair correlations
inside quintile bins of the mean score = the market move) is Gaussian-like
through the middle three quintiles, well above the Gaussian benchmark in the
worst quintile on every pair, and below it in the best quintile on the HSI
pairs only. A single centred tanh cannot hold that shape, so the profile is
the two-sided sigmoid_profile: a plateau through the middle, a down-tail
lift and an up-tail cut, each per index, with common centres and width.

Calibration: the unconditional correlation is pinned to the 21-day
log-return matrix (so the Gaussian-21d row and the profile row differ by
shape only) and the shape is fit to the realized EXCESS over the Gaussian
benchmark in each bin (realized bin minus the Gaussian bin at the data's
rank-correlation level, added to the model's constant-profile bin at the
pinned level), residuals weighted by the block-bootstrap sd of each bin.
Least squares on CRN Sobol points so the objective is smooth.

Rows priced (ATM / 112% cap / spread, four dates, 2^20 CRN Sobol):
  gauss_5d     the existing baseline (5-day log-return correlation matrix)
  gauss_21d    Gaussian copula on the 21-day log-return matrix (Pearson)
  profile_21d  the calibrated smooth profile pinned to that same level
               (shape effect only vs gauss_21d)
  profile_rank the profile fit to the 21-day normal-score profile directly,
               level AND shape: the data's 21-day Pearson (0.82) is inflated
               by the COVID co-crash relative to its rank correlation (0.74),
               which is the tail dependence the profile models explicitly,
               so this is the copula-consistent reading of "21-day
               correlation"
  crash_lift   the one-parameter model recommended on 2026-10-27: the same
               sigmoid with the up-tail cut removed, centre -1 sigma and
               width 0.3 FIXED from the shape of the realized profile, one
               shared strength lambda fit to the bins, and the unconditional
               level re-matched to the existing 5-day matrix (so it differs
               from gauss_5d by the crash lift alone). Its lambda is also
               re-fit on sub-samples (ex-COVID, first / second half) as the
               parameter-stability check.

Writes data/rainbow_corr_profile.csv (prices) and
data/rainbow_corr_profile_fit.json (parameters, binned fit). Run from the
repo root: uv run python scripts/run_rainbow_corr_profile.py (~4 min CPU).
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm, qmc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.rainbow import (call_spread_payoff, date_index, historical_corr,
                             implied_marginal, one_factor_loadings,
                             price_rainbow_quad, profile_uniforms,
                             rainbow_basket, sigmoid_profile, spot_panel,
                             step_profile)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
DATES = ["2026-07-31", "2022-03-16", "2024-10-08", "2020-03-16"]
NAMES = ("SPX", "SX5E", "HSI")
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_LAB = ["SPX-SX5E", "SPX-HSI", "SX5E-HSI"]
HORIZON = 21
N_BINS = 5
N_FIT, N_PRICE = 2 ** 17, 2 ** 20
LEGS = {"atm": (1.00, np.inf), "cap": (1.12, np.inf), "spread": (1.00, 1.12)}
BANK = {"atm": 25.0, "cap": -25.0, "spread": 50.0}   # bank minus model, bps


def scores(r):
    return norm.ppf((r.argsort(0).argsort(0) + 0.5) / len(r))


def binned_corr(z, n_bins=N_BINS):
    """Pair correlations of normal scores inside quantile bins of the mean
    score (the market move); returns (n_bins, 3) and the bins' mean W."""
    w = z.mean(1)
    edges = np.quantile(w, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges[1:-1], w, side="right"), 0, n_bins - 1)
    out, wm = np.zeros((n_bins, 3)), np.zeros(n_bins)
    for k in range(n_bins):
        m = b == k
        wm[k] = w[m].mean()
        out[k] = [np.corrcoef(z[m, i], z[m, j])[0, 1] for i, j in PAIRS]
    return out, wm


def pair_corr(z):
    return np.corrcoef(z.T)[[0, 0, 1], [1, 2, 2]]


def unpack(p):
    lam_dn, lam_up, a = p[0:3], p[3:6], p[9:12]
    c_dn, c_up, width = p[6], p[7], np.exp(p[8])
    return lam_dn, lam_up, c_dn, c_up, width, a


def make_profile(p):
    lam_dn, lam_up, c_dn, c_up, width, a = unpack(p)
    return sigmoid_profile(a, lam_dn, lam_up, c_dn, c_up, width), (c_dn, c_up)


def main():
    t0 = time.time()
    sds = {n: load_surfaces(f"{n}_volSurface.csv") for n in NAMES}
    spots = spot_panel(sds)
    c5 = historical_corr(spots, 5).values
    c21 = historical_corr(spots, HORIZON).values
    print(f"loaded {time.time() - t0:.1f}s; 5d pairs {np.round(pair_corr_mat(c5), 3)}, "
          f"21d pairs {np.round(pair_corr_mat(c21), 3)}")

    # --- realized 21d profile and its bootstrap sd ---------------------------
    r = np.log(spots).diff(HORIZON).dropna().values
    z_hist = scores(r)
    real, w_real = binned_corr(z_hist)
    rng = np.random.default_rng(0)
    n, blk = len(r), 126
    boots = []
    for _ in range(400):
        st = rng.integers(0, n - blk, size=n // blk + 1)
        idx = np.concatenate([np.arange(s, s + blk) for s in st])[:n]
        boots.append(binned_corr(scores(r[idx]))[0])
    sd = np.maximum(np.array(boots).std(0), 0.04)
    c_rank = np.corrcoef(z_hist.T)
    gauss_rank = binned_corr(rng.standard_normal((2 ** 19, 3)) @ np.linalg.cholesky(c_rank).T)[0]
    excess = real - gauss_rank

    pts_fit = qmc.Sobol(d=4, scramble=True, seed=11).random_base2(int(np.log2(N_FIT)))
    c_rank = np.corrcoef(z_hist.T)
    gauss_rank = binned_corr(rng.standard_normal((2 ** 19, 3)) @ np.linalg.cholesky(c_rank).T)[0]
    excess = real - gauss_rank

    def fit_profile(label, level_target, bin_target, a_start):
        def residuals(p):
            prof, breaks = make_profile(p)
            z = norm.ppf(profile_uniforms(pts_fit, prof, breaks))
            b, _ = binned_corr(z)
            return np.concatenate([((b - bin_target) / sd).ravel(),
                                   (pair_corr(z) - level_target) / 0.005])
        p0 = np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.15, -0.8, 0.8, np.log(0.4), *a_start])
        lo = np.array([0, 0, 0, 0, 0, 0, -2.5, 0.0, np.log(0.15), 0.05, 0.05, 0.05])
        hi = np.array([1.5, 1.5, 1.5, 1, 1, 1, 0.0, 2.5, np.log(1.5), 0.999, 0.999, 0.999])
        fit = least_squares(residuals, p0, bounds=(lo, hi), diff_step=1e-3, xtol=1e-6,
                            ftol=1e-6, max_nfev=600)
        p = fit.x
        lam_dn, lam_up, c_dn, c_up, width, a = unpack(p)
        prof, breaks = make_profile(p)
        z0 = norm.ppf(profile_uniforms(pts_fit, step_profile(a_start, a_start), ()))
        bench, _ = binned_corr(z0)
        res_bins = fit.fun[:N_BINS * 3].reshape(N_BINS, 3)
        print(f"\n[{label}] fit: {fit.nfev} evals, chi2 bins {np.sum(res_bins ** 2):.1f} on 15 "
              f"(constant profile {np.sum(((bench - bin_target) / sd) ** 2):.1f}), "
              f"{time.time() - t0:.0f}s")
        print(f"  lam_dn {np.round(lam_dn, 3)}  lam_up {np.round(lam_up, 3)}  c_dn {c_dn:+.2f} "
              f"c_up {c_up:+.2f} width {width:.2f}  base loadings {np.round(a, 3)} "
              f"(constant-profile loadings {np.round(a_start, 3)})")
        z_fit = norm.ppf(profile_uniforms(pts_fit, prof, breaks))
        b_fit, _ = binned_corr(z_fit)
        print("  binned profile (quintiles of the market move): realized excess over the "
              "Gaussian at the data's rank level vs the model's excess over its constant profile")
        print("  bin  meanW |  realized excess (sd)              |  model excess")
        for k in range(N_BINS):
            re = " ".join(f"{excess[k, i]:+.2f}({sd[k, i]:.2f})" for i in range(3))
            me = " ".join(f"{b_fit[k, i] - bench[k, i]:+.2f}" for i in range(3))
            print(f"   {k}   {w_real[k]:+.2f} |  {re}  |  {me}")
        print(f"  unconditional pairs: model {np.round(pair_corr(z_fit), 3)} vs target "
              f"{np.round(level_target, 3)}")
        w_grid = np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
        pc = np.array([prof(w_grid, i) * prof(w_grid, j) for i, j in PAIRS])
        print("  correlation profile rho_ij(W):   W  " + "  ".join(f"{l:>8s}" for l in PAIR_LAB))
        for k, w in enumerate(w_grid):
            print(f"                                 {w:+.1f}  " + "  ".join(f"{pc[i, k]:8.3f}" for i in range(3)))
        wm = z_fit.mean(1)
        dn = [np.corrcoef(z_fit[wm < 0, i], z_fit[wm < 0, j])[0, 1] for i, j in PAIRS]
        up = [np.corrcoef(z_fit[wm >= 0, i], z_fit[wm >= 0, j])[0, 1] for i, j in PAIRS]
        print(f"  half-split down {np.round(dn, 2)} | up {np.round(up, 2)}")
        return dict(prof=prof, breaks=breaks, params=dict(
            lam_dn=lam_dn.tolist(), lam_up=lam_up.tolist(), c_dn=float(c_dn), c_up=float(c_up),
            width=float(width), a=a.tolist(), a_start=a_start.tolist(),
            level_target=np.asarray(level_target).tolist(), model_bins=b_fit.tolist(),
            model_bench=bench.tolist(), chi2_bins=float(np.sum(res_bins ** 2)),
            profile_w=w_grid.tolist(), profile_corr=pc.tolist(),
            half_split=dict(down=np.round(dn, 4).tolist(), up=np.round(up, 4).tolist())))

    r21_index = np.log(spots).diff(HORIZON).dropna().index
    wm = z_hist.mean(1)
    print("realized 21d half-split down "
          f"{np.round([np.corrcoef(z_hist[wm < 0, i], z_hist[wm < 0, j])[0, 1] for i, j in PAIRS], 2)}"
          f" | up {np.round([np.corrcoef(z_hist[wm >= 0, i], z_hist[wm >= 0, j])[0, 1] for i, j in PAIRS], 2)}"
          f"; rank pairs {np.round(pair_corr(z_hist), 3)} vs log-return Pearson {np.round(pair_corr_mat(c21), 3)}")
    a21 = one_factor_loadings(c21)
    z_bench21 = norm.ppf(profile_uniforms(pts_fit, step_profile(a21, a21), ()))
    fit_pin = fit_profile("profile_21d, level pinned to the 21d Pearson matrix", pair_corr_mat(c21),
                          binned_corr(z_bench21)[0] + excess, a21)
    a_rank = one_factor_loadings(c_rank)
    fit_rank = fit_profile("profile_rank, level + shape from the 21d normal scores", pair_corr(z_hist),
                           real, a_rank)

    # --- one-parameter crash lift: lift below C_DN only, level held at the 5-day matrix
    C_DN, WIDTH = -1.0, 0.3
    level5 = pair_corr_mat(c5)

    def crash_lift(lam):
        """base loadings re-matched by a fixed point so the unconditional
        normal-score correlation equals the 5-day matrix; returns profile, scores, loadings"""
        cb = np.array(c5)
        for _ in range(4):
            a = one_factor_loadings(cb)
            prof = sigmoid_profile(a, lam, 0.0, C_DN, 99.0, WIDTH)
            z = norm.ppf(profile_uniforms(pts_fit, prof, (C_DN,)))
            for (i, j), g in zip(PAIRS, level5 - pair_corr(z)):
                cb[i, j] = cb[j, i] = cb[i, j] + g
        return prof, z, a

    def fit_lambda(bin_target):
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda l: np.sum(((binned_corr(crash_lift(l)[1])[0] - bin_target) / sd) ** 2),
                              bounds=(0.0, 1.5), method="bounded", options={"xatol": 0.005})
        return float(res.x), float(res.fun)

    lam_c, chi_c = fit_lambda(real)
    prof_c, z_c, a_c = crash_lift(lam_c)
    b_c, _ = binned_corr(z_c)
    a5 = one_factor_loadings(c5)
    z_g5 = norm.ppf(profile_uniforms(pts_fit, step_profile(a5, a5), ()))
    bench5, _ = binned_corr(z_g5)
    print(f"\n[crash_lift] lambda {lam_c:.3f}, chi2 bins {chi_c:.1f} on 15 (Gaussian-5d "
          f"{np.sum(((bench5 - real) / sd) ** 2):.1f}); plateau loadings {np.round(a_c, 3)} vs {np.round(a5, 3)}; "
          f"unconditional {np.round(pair_corr(z_c), 3)} vs 5d {np.round(level5, 3)}  {time.time() - t0:.0f}s")
    print("  model bins minus realized, sd units:\n", np.round((b_c - real) / sd, 1))
    dn_c, up_c = half_split_pairs(z_c)
    print(f"  half-split down {np.round(dn_c, 2)} | up {np.round(up_c, 2)}")
    subs = {"ex_covid": ~((r21_index >= "2020-02-01") & (r21_index <= "2020-06-30")),
            "first_half": r21_index < "2022-10-01", "second_half": r21_index >= "2022-10-01"}
    lam_sub = {}
    for lab, m in subs.items():
        lam_sub[lab], _ = fit_lambda(binned_corr(scores(r[m]))[0])
        print(f"  lambda on {lab:12s} {lam_sub[lab]:.3f}")
    w_grid = np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
    pc_c = np.array([prof_c(w_grid, i) * prof_c(w_grid, j) for i, j in PAIRS])
    fit_crash = dict(prof=prof_c, breaks=(C_DN,), params=dict(
        lam=lam_c, c_dn=C_DN, width=WIDTH, a=a_c.tolist(), a_start=a5.tolist(), level_target=level5.tolist(),
        model_bins=b_c.tolist(), model_bench=bench5.tolist(), chi2_bins=chi_c, lam_subsamples=lam_sub,
        profile_w=w_grid.tolist(), profile_corr=pc_c.tolist(),
        half_split=dict(down=np.round(dn_c, 4).tolist(), up=np.round(up_c, 4).tolist())))

    # --- pricing: three rows, four dates, CRN ---------------------------------
    pts = qmc.Sobol(d=4, scramble=True, seed=7).random_base2(int(np.log2(N_PRICE)))
    a5 = one_factor_loadings(c5)
    rows_u = {"gauss_5d": profile_uniforms(pts, step_profile(a5, a5), ()),
              "gauss_21d": profile_uniforms(pts, step_profile(a21, a21), ()),
              "profile_21d": profile_uniforms(pts, fit_pin["prof"], fit_pin["breaks"]),
              "profile_rank": profile_uniforms(pts, fit_rank["prof"], fit_rank["breaks"]),
              "crash_lift": profile_uniforms(pts, fit_crash["prof"], fit_crash["breaks"])}
    out = []
    for date in DATES:
        idx = {n_: date_index(sd, date) for n_, sd in sds.items()}
        ms = [implied_marginal(sd, idx[n_], 1.0) for n_, sd in sds.items()]
        df = float(sds["SPX"].discount(idx["SPX"], 1.0))
        for row, u in rows_u.items():
            perf = np.column_stack([m.quantile(u[:, j]) for j, m in enumerate(ms)])
            b = rainbow_basket(perf)
            rec = dict(date=date, row=row, mean_basket=b.mean())
            for leg, (klo, khi) in LEGS.items():
                rec[leg] = df * call_spread_payoff(b, klo, khi).mean() * 1e4
            if row.startswith("gauss"):
                c = c5 if row == "gauss_5d" else c21
                rec["atm_quad"] = price_rainbow_quad(ms, c, df, k_lo=1.0, k_hi=np.inf)["pv"] / 100
                rec["cap_quad"] = price_rainbow_quad(ms, c, df, k_lo=1.12, k_hi=np.inf)["pv"] / 100
            out.append(rec)
    res = pd.DataFrame(out)
    base = res[res.row == "gauss_5d"].set_index("date")
    for leg in LEGS:
        res[f"d_{leg}"] = res[leg].values - base.loc[res.date, leg].values
        res[f"share_{leg}"] = res[f"d_{leg}"] / BANK[leg]
    DATA.mkdir(exist_ok=True)
    res.to_csv(DATA / "rainbow_corr_profile.csv", index=False)
    json.dump(dict(c5=c5.tolist(), c21=c21.tolist(), c_rank21=c_rank.tolist(),
                   bins_w=w_real.tolist(), realized=real.tolist(), realized_excess=excess.tolist(),
                   sd=sd.tolist(), profile_21d=fit_pin["params"], profile_rank=fit_rank["params"],
                   crash_lift=fit_crash["params"]),
              open(DATA / "rainbow_corr_profile_fit.json", "w"), indent=1)

    pd.set_option("display.width", 220)
    print("\nprices (bps of notional) and move vs the existing baseline; bank gap +25 / -25 / +50")
    for date in DATES:
        sub = res[res.date == date]
        print(f"\n{date}")
        print(sub[["row", "atm", "cap", "spread", "d_atm", "d_cap", "d_spread",
                   "share_atm", "share_cap", "share_spread", "atm_quad", "cap_quad"]]
              .round(2).to_string(index=False))
    print(f"\ndone {time.time() - t0:.0f}s -> {DATA / 'rainbow_corr_profile.csv'}")


def pair_corr_mat(c):
    return np.asarray(c)[[0, 0, 1], [1, 2, 2]]


def half_split_pairs(z):
    w = z.mean(1)
    dn = np.array([np.corrcoef(z[w < 0, i], z[w < 0, j])[0, 1] for i, j in PAIRS])
    up = np.array([np.corrcoef(z[w >= 0, i], z[w >= 0, j])[0, 1] for i, j in PAIRS])
    return dn, up


if __name__ == "__main__":
    main()
