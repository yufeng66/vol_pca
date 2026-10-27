"""Up/down correlation regimes on the rainbow book: how much of the bank-quote
gap does asymmetric dependence explain?

Context (user, 2026-10-22 / 2026-10-27): bank quotes sit ~+25 bps of notional
ABOVE the model on the ATM basket call and ~-25 bps BELOW it on the 112% cap
call. Every symmetric knob (correlation level, vol level, forwards) moves
both legs the same way, so only an asymmetric dependence structure can
reproduce the sign pattern. This script prices the two legs under a
two-regime copula with one correlation matrix when the common market factor
is down and another when it is up, marginals kept exact, and maps the
reachable (ATM, cap) moves against the bank's (+25, -25).

Model (Andersen-Sidenius random factor loadings, two states):
    Z_i = a_i(W) W + sqrt(1 - a_i(W)^2) e_i,   a_i(W) = a_i^dn if W < theta
                                                       else a_i^up
with W, e_i iid N(0,1). For three indices every positive pairwise target
(r12, r13, r23) is reached exactly by the loadings a_i = sqrt(r_ij r_ik /
r_jk), so "the correlation matrix on the down side" C_dn and "on the up
side" C_up are literal inputs. The raw Z_i are not normal any more (fatter
left tail when a^dn > a^up); their exact CDF is tabulated by quadrature over
W and applied before the implied-marginal quantile, so each index's
risk-neutral distribution is untouched - this is a copula-only change.
C_dn = C_up recovers the Gaussian copula to fp.

Sweep: C_dn / C_up = the 5-day historical matrix shifted additively on all
three pairs so the SPX-SX5E pair reads rho_dn / rho_up (grid rows/cols);
rho_dn is capped near 0.92 by the SX5E loading hitting 1. Side variants:
threshold theta = -1 (crash-only regime), theta = +0.5 / +1 (the low
correlation only in a strong rally), the SPX-SX5E-only shift, and the smooth
a_i(W) = a_i (1 - lam tanh W) profile of the 2026-10-22 scratch test.
Plausibility: exceedance correlations (normal scores, both below the q /
both above the 1-q quantile) of the simulated copula vs the realized 5/21/63
day returns.

Writes data/rainbow_corr_regime.csv (gitignored) and prints the tables. Run
from the repo root: uv run python scripts/run_rainbow_corr_regime.py
(~1.5 min CPU, 2^20 CRN Sobol paths per pricing).
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.rainbow import (call_spread_payoff, date_index, historical_corr,
                             implied_marginal, one_factor_loadings,
                             profile_uniforms, rainbow_basket, spot_panel,
                             step_profile, tanh_profile)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
DATES = ["2026-07-31", "2022-03-16", "2024-10-08", "2020-03-16"]
NAMES = ("SPX", "SX5E", "HSI")
N_PATHS = 2 ** 20
LEGS = {"atm": (1.00, np.inf), "cap": (1.12, np.inf), "spread": (1.00, 1.12)}
BANK = {"atm": 25.0, "cap": -25.0}          # bank minus model, bps of notional
RHO_DN = [0.751, 0.80, 0.85, 0.90, 0.92]
RHO_UP = [0.751, 0.70, 0.65, 0.60, 0.55, 0.45]


def shift_corr(c, delta, pairs="all"):
    out = np.array(c, dtype=float)
    if pairs == "all":
        out[0, 1] = out[1, 0] = c[0, 1] + delta
        out[0, 2] = out[2, 0] = c[0, 2] + delta
        out[1, 2] = out[2, 1] = c[1, 2] + delta
    elif pairs == "spx_sx5e":
        out[0, 1] = out[1, 0] = c[0, 1] + delta
    return out


def regime_uniforms(points, a_dn, a_up, theta=0.0):
    """Two-state model: C_dn when the market factor W < theta, C_up above."""
    return profile_uniforms(points, step_profile(a_dn, a_up, theta), (theta,))


def exceedance_corr(z, q):
    """corr(z1, z2 | both < q-quantile) and (| both > (1-q)-quantile), per
    pair, on normal scores; returns (down, up) arrays over the three pairs."""
    lo, hi = norm.ppf(q), norm.ppf(1 - q)
    dn, up = [], []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = (z[:, i] < lo) & (z[:, j] < lo)
        dn.append(np.corrcoef(z[m, i], z[m, j])[0, 1])
        m = (z[:, i] > hi) & (z[:, j] > hi)
        up.append(np.corrcoef(z[m, i], z[m, j])[0, 1])
    return np.array(dn), np.array(up)


def realized_exceedance(spots, horizon, q):
    r = np.log(spots).diff(horizon).dropna().values
    ranks = (r.argsort(0).argsort(0) + 0.5) / len(r)
    return exceedance_corr(norm.ppf(ranks), q)


def main():
    t0 = time.time()
    sds = {n: load_surfaces(f"{n}_volSurface.csv") for n in NAMES}
    spots = spot_panel(sds)
    c_hist = historical_corr(spots, 5).values
    a_hist = one_factor_loadings(c_hist)
    print(f"loaded {time.time() - t0:.1f}s; 5d corr pairs "
          f"{np.round(c_hist[[0, 0, 1], [1, 2, 2]], 3)}; loadings {np.round(a_hist, 3)}")

    points = qmc.Sobol(d=4, scramble=True, seed=7).random_base2(int(np.log2(N_PATHS)))

    # --- realized exceedance asymmetry (plausibility yardstick) -------------
    print("\nrealized exceedance correlation, normal scores, mean over pairs "
          "(SPX-SX5E pair in brackets): down - up")
    for q in (0.30, 0.15):
        for h in (5, 21, 63):
            dn, up = realized_exceedance(spots, h, q)
            print(f"  q={q:.2f} {h:3d}d: {dn.mean() - up.mean():+.3f} "
                  f"[{dn[0] - up[0]:+.3f}]  (down {dn.mean():.3f}, up {up.mean():.3f})")

    # --- the sweep ------------------------------------------------------------
    rows = []
    configs = []
    for rd in RHO_DN:
        for ru in RHO_UP:
            configs.append(dict(variant="all", theta=0.0, rho_dn=rd, rho_up=ru))
    for rd in (0.85, 0.92):                      # crash-only regime, up side at history
        configs.append(dict(variant="all", theta=-1.0, rho_dn=rd, rho_up=0.751))
    for rd, ru in ((0.85, 0.65), (0.90, 0.60)):  # crash-only, both sides moved
        configs.append(dict(variant="all", theta=-1.0, rho_dn=rd, rho_up=ru))
    for rd, ru in ((0.85, 0.65), (0.89, 0.60)):  # SPX-SX5E pair only
        configs.append(dict(variant="spx_sx5e", theta=0.0, rho_dn=rd, rho_up=ru))
    for th in (0.5, 1.0):                        # step inside the rally
        for rd, ru in ((0.751, 0.60), (0.751, 0.45), (0.80, 0.55), (0.85, 0.55),
                       (0.85, 0.45), (0.90, 0.45)):
            configs.append(dict(variant="all", theta=th, rho_dn=rd, rho_up=ru))
    for lam in (0.3, 0.6, 0.8):                  # smooth 2026-10-22 profile
        configs.append(dict(variant="tanh", theta=0.0, rho_dn=np.nan, rho_up=np.nan,
                            lam=lam))

    margs = {}
    for date in DATES:
        idx = {n: date_index(sd, date) for n, sd in sds.items()}
        margs[date] = ([implied_marginal(sd, idx[n], 1.0) for n, sd in sds.items()],
                       float(sds["SPX"].discount(idx["SPX"], 1.0)))

    for cfg in configs:
        cfg.setdefault("lam", 0.0)
        if cfg["variant"] == "tanh":
            prof = tanh_profile(a_hist, cfg["lam"])
            # label the smooth profile by its SPX-SX5E correlation at W = -/+1
            cfg["rho_dn"] = float(prof(np.array([-1.0]), 0)[0] * prof(np.array([-1.0]), 1)[0])
            cfg["rho_up"] = float(prof(np.array([1.0]), 0)[0] * prof(np.array([1.0]), 1)[0])
            u = profile_uniforms(points, prof, ())
        else:
            c_dn = shift_corr(c_hist, cfg["rho_dn"] - c_hist[0, 1], cfg["variant"])
            c_up = shift_corr(c_hist, cfg["rho_up"] - c_hist[0, 1], cfg["variant"])
            try:
                a_dn, a_up = one_factor_loadings(c_dn), one_factor_loadings(c_up)
            except ValueError as e:
                print("skip", cfg, e)
                continue
            u = regime_uniforms(points, a_dn, a_up, cfg["theta"])
        z = norm.ppf(u)
        pear = np.corrcoef(z.T)[[0, 0, 1], [1, 2, 2]]
        ex = {}
        for q in (0.30, 0.15):
            dn, up = exceedance_corr(z, q)
            ex[f"exc_asym_q{int(q * 100)}"] = dn.mean() - up.mean()
            ex[f"exc_asym12_q{int(q * 100)}"] = dn[0] - up[0]
        for date in DATES:
            ms, df = margs[date]
            perf = np.column_stack([m.quantile(u[:, j]) for j, m in enumerate(ms)])
            b = rainbow_basket(perf)
            row = dict(date=date, **cfg, pearson12=pear[0], pearson13=pear[1],
                       pearson23=pear[2], mean_basket=b.mean(), **ex)
            for leg, (klo, khi) in LEGS.items():
                row[leg] = df * call_spread_payoff(b, klo, khi).mean() * 1e4  # bps
            rows.append(row)
        print(f"  {cfg['variant']:8s} th={cfg['theta']:+.1f} lam={cfg['lam']:.1f} "
              f"rho_dn={cfg['rho_dn']:.3f} rho_up={cfg['rho_up']:.3f}  {time.time() - t0:6.1f}s")

    res = pd.DataFrame(rows)
    base = res[(res.variant == "all") & (res.theta == 0) & (res.rho_dn == 0.751)
               & (res.rho_up == 0.751)].set_index("date")
    for leg in LEGS:
        res[f"d_{leg}"] = res[leg].values - base.loc[res.date, leg].values
    res["explained_atm"] = res.d_atm / BANK["atm"]
    res["explained_cap"] = res.d_cap / BANK["cap"]
    DATA.mkdir(exist_ok=True)
    res.to_csv(DATA / "rainbow_corr_regime.csv", index=False)

    pd.set_option("display.width", 200)
    for date in DATES:
        sub = res[(res.date == date) & (res.variant == "all") & (res.theta == 0)]
        print(f"\n{date}: baseline atm {base.loc[date, 'atm']:.1f} / cap "
              f"{base.loc[date, 'cap']:.1f} / spread {base.loc[date, 'spread']:.1f} bps."
              " model minus baseline, bps: atm / cap  (rows rho_dn, cols rho_up)")
        tab = sub.pivot(index="rho_dn", columns="rho_up", values=["d_atm", "d_cap"])
        txt = sub.assign(cell=lambda d: d.d_atm.map("{:+.0f}".format) + "/"
                         + d.d_cap.map("{:+.0f}".format)).pivot(
            index="rho_dn", columns="rho_up", values="cell")
        print(txt.to_string())
    for date in DATES[:2]:
        print(f"\nside variants ({date}): rho_dn/rho_up of the tanh rows = SPX-SX5E corr at W=-1/+1")
        sub = res[(res.date == date) & ((res.theta != 0) | (res.variant != "all"))]
        print(sub[["variant", "theta", "lam", "rho_dn", "rho_up", "d_atm", "d_cap", "d_spread",
                   "pearson12", "exc_asym_q30", "exc_asym12_q30"]].round(3).to_string(index=False))
    print("\nexceedance asymmetry (mean over pairs, q=0.30) by cell, theta=0 all-pairs:")
    sub = res[(res.date == "2026-07-31") & (res.variant == "all") & (res.theta == 0)]
    print(sub.pivot(index="rho_dn", columns="rho_up", values="exc_asym_q30").round(3).to_string())
    print("pearson SPX-SX5E after the marginal fix:")
    print(sub.pivot(index="rho_dn", columns="rho_up", values="pearson12").round(3).to_string())
    print(f"\ndone {time.time() - t0:.0f}s -> {DATA / 'rainbow_corr_regime.csv'}")


if __name__ == "__main__":
    main()
