"""Rainbow VaR study: nested full reval vs bumped greeks on a regime panel.

Five hand-picked as-of dates spanning the book's gamma regimes (one as-of
date costs ~1 GPU minute, so a hand-picked panel replaces the ~33h full
rolling history):

  2020-03-16  COVID peak — displaced far-OTM book (long SPX gamma!), crash
              surfaces (the BL mean quirk's home turf)
  2021-09-20  mixed signs — SPX/SX5E short gamma, HSI long
  2022-03-16  the original study date — short gamma everywhere (1'G1 -$640M)
  2024-10-08  rally book — long gamma, all diagonals positive
  2026-07-31  the current book — deep ITM, longest gamma (1'G1 +$691M)

Per date, one GPU pass sweeps all 1,968 historical joint-date scenarios
through the factory's dgrid/two-tau seams (CRN Sobol, 512-path standard) and
one bumped-greeks pass measures the formula-free projection inputs (SS spot
block, per-index k=6 + joint k=8 sticky-strike factors — the factor models
are fit once on the full joint history and shared across dates, the vanilla
frozen-weights convention — own-spot PC1 and joint-PC1 crosses). Caches per
date, both skip-if-exists:

  data/rainbow_var_scen_<asof>.csv     per-scenario full-reval P&L + scores
  data/rainbow_var_greeks_<asof>.npz   every bump sensitivity + bookkeeping

The analysis notebook (rainbow_var.ipynb) reads the caches only. Run from
the repo root: uv run python scripts/run_rainbow_var.py
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.data import load_surfaces
from vol_pca.factors import factor_scores, fit_pca
from vol_pca.rainbow_torch import (MarginalFactory, rainbow_bump_greeks,
                                   rainbow_fit_dsigma,
                                   rainbow_scenario_dsigma, rainbow_scenarios,
                                   rainbow_var_full)

ASOF_DATES = ["2020-03-16", "2021-09-20", "2022-03-16", "2024-10-08",
              "2026-07-31"]
K_IDX, K_JOINT = 6, 8               # factors bumped per index / for the joint fit
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def main():
    todo = [a for a in ASOF_DATES
            if not ((DATA / f"rainbow_var_scen_{a}.csv").exists()
                    and (DATA / f"rainbow_var_greeks_{a}.npz").exists())]
    if not todo:
        print("all cached - delete data/rainbow_var_* to recompute")
        return
    sds = {n: load_surfaces(f"{n}_volSurface.csv")
           for n in ("SPX", "SX5E", "HSI")}
    names = list(sds)
    scen = rainbow_scenarios(sds)
    fac = MarginalFactory(sds)
    print(f"{len(scen)} scenarios; dates to run: {todo}")

    # --- factor models: fit ONCE on the full joint history, shared by every
    # as-of date (frozen-weights convention); scores re-anchor per date
    t0 = time.time()
    fit_ds = rainbow_fit_dsigma(sds)
    models = {n: fit_pca(fit_ds[n], weights="corr") for n in names}
    joint = fit_pca(np.hstack([fit_ds[n] for n in names]), weights="corr")
    print(f"factor fits {time.time() - t0:.1f}s  "
          f"evr3: " + "  ".join(f"{n} {models[n].evr[:3].sum():.3f}"
                                for n in names)
          + f"  joint {joint.evr[:3].sum():.3f}")

    # factor list: [spx 0..5 | sx5e 0..5 | hsi 0..5 | joint 0..7]
    factors, eps, labels = [], [], []
    for n in names:
        m = models[n]
        for k in range(K_IDX):
            factors.append({n: (m.components[k] / m.weights).reshape(8, 13)})
            eps.append(m.score_std[k])
            labels.append(f"{n.lower()}_{k}")
    npil = 104
    for k in range(K_JOINT):
        row = joint.components[k] / joint.weights
        factors.append({n: row[j * npil:(j + 1) * npil].reshape(8, 13)
                        for j, n in enumerate(names)})
        eps.append(joint.score_std[k])
        labels.append(f"joint_{k}")
    mu = {n: models[n].mu.reshape(8, 13) for n in names}
    cross = [(j, j * K_IDX) for j in range(3)]          # own spot x own PC1
    cross += [(j, 3 * K_IDX) for j in range(3)]         # each spot x joint PC1

    for asof in todo:
        print(f"\n=== {asof} ===")
        t0 = time.time()
        bg = rainbow_bump_greeks(sds, factors, eps, asof=asof, mu=mu,
                                 cross=cross, fac=fac)
        t_greeks = time.time() - t0
        print(f"bump greeks: {bg.n_revals} pricings in {t_greeks:.1f}s  "
              f"n_pos {bg.n_pos}  delta {np.round(bg.delta / 1e6, 1)}M  "
              f"diag gamma {np.round(np.diag(bg.gamma) / 1e6, 0)}M")

        t0 = time.time()
        ds_scen = rainbow_scenario_dsigma(sds, scen, asof=asof)
        scores = np.hstack(
            [factor_scores(ds_scen[n], models[n])[:, :K_IDX] for n in names]
            + [factor_scores(np.hstack([ds_scen[n] for n in names]),
                             joint)[:, :K_JOINT]])
        print(f"scenario scores {time.time() - t0:.1f}s")

        t0 = time.time()
        full = rainbow_var_full(sds, scen, asof=asof, fac=fac, progress=60)
        t_full = time.time() - t0
        print(f"full reval: {len(scen)} x {full['n_pos']} in {t_full:.1f}s  "
              f"(pv0 {full['pv0']:,.0f})")
        assert abs(full["pv0"] - bg.pv0) < 1e-6

        df = pd.DataFrame({"date": scen.dates, "dt": scen.dt,
                           **{f"ratio_{n.lower()}": scen.ratio[:, j]
                              for j, n in enumerate(names)},
                           "pnl_full": full["pnl"],
                           **{f"f_{lab}": scores[:, i]
                              for i, lab in enumerate(labels)}})
        DATA.mkdir(exist_ok=True)
        df.to_csv(DATA / f"rainbow_var_scen_{asof}.csv", index=False)
        np.savez(DATA / f"rainbow_var_greeks_{asof}.npz",
                 pv0=bg.pv0, delta=bg.delta, gamma=bg.gamma, speed=bg.speed,
                 base_mu=bg.base_mu, expos=bg.expos, curv=bg.curv,
                 cross=bg.cross, cross_spec=np.array(bg.cross_spec),
                 eps=bg.eps, eps_spot=bg.eps_spot, labels=np.array(labels),
                 names=names, n_pos=bg.n_pos, n_revals=bg.n_revals,
                 k_idx=K_IDX, k_joint=K_JOINT, asof=asof, t_full=t_full,
                 t_greeks=t_greeks, score_std=np.array(eps))
        print(f"wrote rainbow_var_scen_{asof}.csv + rainbow_var_greeks_{asof}.npz")


if __name__ == "__main__":
    main()
