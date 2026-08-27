"""RILA reset-package study: calibration, state grid and convergence sweeps.

Valuation-only thread (user spec 2026-08-27). Everything the notebook reads
is cached here; each stage is skip-if-exists, so deleting one file reruns
only that stage. Run from the repo root:

    uv run python scripts/run_rila.py

Stages and caches (all under data/, gitignored):

  1. calibration   rila_lv_<date>_<wing>.npz   sigma_loc grid + diagnostics
                   rila_calib.csv              per date x wing summary
                   rila_reprice.csv            vanilla repricing table
  2. wing study    rila_wing.csv               newly-issued value + legs,
                                               clamp vs taper, all five dates
  3. state grid    rila_states.csv             5 dates x (newly issued,
                                               mid-window r=4/2 x K/S,
                                               post-window ttm x K/S):
                                               PDE reference, GPU hybrid,
                                               CPU reference, reset premium
  4. convergence   rila_conv_cpu.csv           CPU plain MC + control variate
                   rila_conv_gpu.csv           device pseudo / Sobol+BB /
                                               hybrid rungs, error vs seconds
                   rila_bias.csv               value vs step size

Wall clock on the reference machine (RTX 4070 Laptop, WSL2): ~13 min cold.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.local_vol import (dupire_local_vol, european_package_surface,
                               load_term_surfaces, mc_step_data, mc_time_grid,
                               pde_package, pde_reprice_table,
                               pde_reset_price, per_100, price_rila_mc,
                               rila_state)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV = ROOT / "SPX_volSurface.csv"

DATES = ["2020-03-16", "2021-09-20", "2022-03-16", "2024-10-08", "2026-07-31"]
STUDY = ["2026-07-31", "2020-03-16"]          # deep convergence dates
WINGS = ["taper", "clamp"]
WING = "taper"                                # headline convention

# PDE reference config: converged to ~4e-4 per 100 of basis against the
# flat-vol oracle (see tests + CHANGELOG), ~3s per state
PDE_REF = dict(n_k=145, n_x=1201, per_year=320)
PDE_EUR = dict(n_x=1201, per_year=320)
MID_STATES = [(r, ks) for r in (4, 2) for ks in (0.95, 1.00, 1.08)]
POST_STATES = [(ttm, ks) for ttm in (5.5, 3.0, 1.0)
               for ks in (0.75, 0.90, 1.00, 1.10)]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------- 1. calibration

def stage_calibration(surfaces):
    lvs = {}
    todo = [(d, w) for d in DATES for w in WINGS
            if not (DATA / f"rila_lv_{d}_{w}.npz").exists()]
    have = (DATA / "rila_calib.csv").exists() and (DATA / "rila_reprice.csv").exists()
    if not todo and have:
        _log("calibration cached")
        for d in DATES:
            for w in WINGS:
                lvs[(d, w)] = dupire_local_vol(surfaces[d], wing=w)
        return lvs
    rows, tabs = [], []
    for d in DATES:
        ts = surfaces[d]
        for w in WINGS:
            t0 = time.time()
            lv = dupire_local_vol(ts, wing=w)
            lvs[(d, w)] = lv
            tab = pde_reprice_table(lv, ts, n_x=801, per_year=160, wing=w)
            tab.insert(0, "wing", w)
            tab.insert(0, "date", d)
            tabs.append(tab)
            far = tab[tab.tau >= 0.5]["d_vol_pts"].abs()
            allt = tab["d_vol_pts"].abs()
            rows.append({"date": d, "wing": w, "spot": ts.spot,
                         "n_terms": len(ts.taus), "tau_max": ts.taus[-1],
                         **{k: v for k, v in lv.diag.items() if k != "wing"},
                         "err_mean": allt.mean(), "err_p95": allt.quantile(.95),
                         "err_max": allt.max(), "err_mean_far": far.mean(),
                         "err_p95_far": far.quantile(.95), "err_max_far": far.max(),
                         "secs": time.time() - t0})
            np.savez(DATA / f"rila_lv_{d}_{w}.npz", taus=lv.taus, ys=lv.ys,
                     sig=lv.sig, w=lv.w, spot=lv.spot, date=d, wing=w,
                     **{k: v for k, v in lv.diag.items()
                        if not isinstance(v, str)})
            _log(f"  calib {d} {w:5s} floor {lv.diag['pct_floor']:5.2f}% "
                 f"cap {lv.diag['pct_cap']:5.1f}% "
                 f"err(tau>=0.5) mean {far.mean():.3f} max {far.max():.3f} vol pts")
    pd.DataFrame(rows).to_csv(DATA / "rila_calib.csv", index=False)
    pd.concat(tabs, ignore_index=True).to_csv(DATA / "rila_reprice.csv", index=False)
    return lvs


# --------------------------------------------------------- 2. wing study

def stage_wings(surfaces, lvs):
    out = DATA / "rila_wing.csv"
    if out.exists():
        _log("wing study cached")
        return
    rows = []
    for d in DATES:
        ts = surfaces[d]
        st = rila_state(ts, 6, 1.0)
        for w in WINGS:
            lv = lvs[(d, w)]
            eur = european_package_surface(ts, st.basis, st.ttm, wing=w)
            ref = pde_reset_price(lv, st, **PDE_REF)["pv"]
            frozen = pde_package(lv, st.basis, st.ttm, **PDE_EUR)
            rows.append({"date": d, "wing": w, "spot": ts.spot,
                         "reset": per_100(ref, st.basis),
                         "frozen_pde": per_100(frozen, st.basis),
                         "frozen_surface": per_100(eur["pv"], st.basis),
                         "put_80": per_100(eur["put_80"], st.basis),
                         "call_100": per_100(eur["call_100"], st.basis),
                         "call_200": per_100(eur["call_200"], st.basis),
                         "vol_200": eur["vols"][2],
                         "reset_premium": per_100(ref - frozen, st.basis)})
        _log(f"  wing {d}: taper {rows[-2]['reset']:.4f} clamp {rows[-1]['reset']:.4f}")
    pd.DataFrame(rows).to_csv(out, index=False)


# ---------------------------------------------------------- 3. state grid

def stage_states(surfaces, lvs):
    out = DATA / "rila_states.csv"
    if out.exists():
        _log("state grid cached")
        return
    import torch

    from vol_pca.rila_torch import make_tail_table, prepare, rila_replicates
    rows = []
    for d in DATES:
        ts, lv = surfaces[d], lvs[(d, WING)]
        for r, ks in [(6, 1.00)] + MID_STATES:
            st = rila_state(ts, r, ks)
            t0 = time.time()
            ref = pde_reset_price(lv, st, **PDE_REF)["pv"]
            t_pde = time.time() - t0
            frozen = pde_package(lv, st.basis, st.ttm, **PDE_EUR)
            t0 = time.time()
            tab = make_tail_table(lv, st, **PDE_REF)
            t_tab = time.time() - t0
            prep = prepare(lv, st, per_year=252, tail=tab)
            hyb = rila_replicates(lv, st, 1 << 18, replicates=24, prep=prep,
                                  seed0=7000)
            frz = rila_replicates(lv, st, 1 << 18, replicates=24, prep=prep,
                                  seed0=7000, reset=False)     # same scrambles
            dv = hyb["vals"] - frz["vals"]                      # CRN premium
            t0 = time.time()
            cpu = price_rila_mc(lv, st, n_paths=1 << 16, per_year=252, seed=17)
            t_cpu = time.time() - t0
            rows.append({
                "date": d, "resets_left": r, "k_over_s": ks, "ttm": st.ttm,
                "window_end": st.window_end, "basis": st.basis, "spot": ts.spot,
                "pde": per_100(ref, st.basis), "pde_secs": t_pde,
                "frozen_pde": per_100(frozen, st.basis),
                "reset_prem_pde": per_100(ref - frozen, st.basis),
                "gpu": per_100(hyb["pv"], st.basis),
                "gpu_se": per_100(hyb["se"], st.basis),
                "gpu_secs": hyb["secs"], "tail_secs": t_tab,
                "reset_prem_mc": per_100(dv.mean(), st.basis),
                "reset_prem_se": per_100(dv.std(ddof=1) / np.sqrt(len(dv)), st.basis),
                "cpu": per_100(cpu["pv"], st.basis),
                "cpu_se": per_100(cpu["se"], st.basis), "cpu_secs": t_cpu,
                "engine": "hybrid"})
            _log(f"  state {d} r={r} K/S={ks}: pde {rows[-1]['pde']:8.4f} "
                 f"gpu {rows[-1]['gpu']:8.4f}+-{rows[-1]['gpu_se']:.4f} "
                 f"prem {rows[-1]['reset_prem_pde']:.4f}")
        for ttm, ks in POST_STATES:
            st = rila_state(ts, 0, ks, ttm=ttm)
            eur = european_package_surface(ts, st.basis, st.ttm, wing=WING)
            pde = pde_package(lv, st.basis, st.ttm, **PDE_EUR)
            rows.append({
                "date": d, "resets_left": 0, "k_over_s": ks, "ttm": ttm,
                "window_end": 0.0, "basis": st.basis, "spot": ts.spot,
                "surface": per_100(eur["pv"], st.basis),
                "put_80": per_100(eur["put_80"], st.basis),
                "call_100": per_100(eur["call_100"], st.basis),
                "call_200": per_100(eur["call_200"], st.basis),
                "pde": per_100(pde, st.basis), "engine": "surface"})
    pd.DataFrame(rows).to_csv(out, index=False)
    _log("wrote rila_states.csv")


# ------------------------------------------------------- 4a. CPU sweep

CPU_LADDER = {12: range(12, 21), 52: range(12, 21), 252: range(12, 19)}


def stage_cpu(surfaces, lvs):
    out = DATA / "rila_conv_cpu.csv"
    if out.exists():
        _log("cpu convergence cached")
        return
    rows = []
    for d in STUDY:
        ts, lv = surfaces[d], lvs[(d, WING)]
        st = rila_state(ts, 6, 1.0)
        ref = pde_reset_price(lv, st, **PDE_REF)["pv"]
        ec = pde_package(lv, st.basis, st.ttm, **PDE_EUR)
        for per_year, ks in CPU_LADDER.items():
            t_grid, fix_idx = mc_time_grid(st, per_year)
            step = mc_step_data(lv, t_grid)
            for k in ks:
                n = 1 << k
                for cv in ([None, "frozen"] if per_year == 252 else [None]):
                    vals, ses, secs = [], [], []
                    for s in range(3):
                        t0 = time.time()
                        m = price_rila_mc(lv, st, n_paths=n, seed=1000 + s,
                                          t_grid=t_grid, fix_idx=fix_idx,
                                          step=step, cv=cv, cv_exact=ec)
                        secs.append(time.time() - t0)
                        vals.append(m["pv"])
                        ses.append(m["se"])
                    v = np.asarray(vals)
                    rows.append({
                        "date": d, "rung": "cpu_cv" if cv else "cpu_plain",
                        "per_year": per_year, "n_steps": len(t_grid) - 1,
                        "n_paths": n, "seeds": len(v),
                        "pv": per_100(v.mean(), st.basis),
                        "se_internal": per_100(np.mean(ses), st.basis),
                        "rmse_seeds": per_100(v.std(ddof=1), st.basis),
                        "err_vs_pde": per_100(v.mean() - ref, st.basis),
                        "secs": float(np.median(secs)),
                        "ref": per_100(ref, st.basis)})
            _log(f"  cpu {d} py={per_year}: "
                 f"N=2^{max(ks)} se {rows[-1]['se_internal']:.4f} "
                 f"in {rows[-1]['secs']:.2f}s")
    pd.DataFrame(rows).to_csv(out, index=False)


# ------------------------------------------------------- 4b. GPU sweep

GPU_RUNGS = [
    # name, method, bridge, hybrid, per_year, k range, replicates
    ("gpu_pseudo",     "pseudo", False, False, 252, range(12, 19), 24),
    ("gpu_sobol",      "sobol",  False, False, 252, range(12, 18), 16),
    ("gpu_sobol_bb",   "sobol",  True,  False, 252, range(12, 18), 16),
    ("hybrid_pseudo",  "pseudo", False, True,  252, range(12, 21), 24),
    ("hybrid_sobol_bb", "sobol", True,  True,  252, range(11, 21), 24),
]
BIAS_FULL = (12, 24, 52, 126, 252)
BIAS_WIN = (21, 63, 126, 252, 504, 1008)


def stage_gpu(surfaces, lvs):
    out, out_b = DATA / "rila_conv_gpu.csv", DATA / "rila_bias.csv"
    if out.exists() and out_b.exists():
        _log("gpu convergence cached")
        return
    from vol_pca.rila_torch import make_tail_table, prepare, rila_replicates
    rows, brows = [], []
    for d in STUDY:
        ts, lv = surfaces[d], lvs[(d, WING)]
        st = rila_state(ts, 6, 1.0)
        ref = pde_reset_price(lv, st, **PDE_REF)["pv"]
        t0 = time.time()
        tab = make_tail_table(lv, st, **PDE_REF)
        t_tab = time.time() - t0
        for name, method, bridge, hyb, per_year, ks, reps in GPU_RUNGS:
            prep = prepare(lv, st, per_year=per_year, tail=tab if hyb else None)
            for k in ks:
                r = rila_replicates(lv, st, 1 << k, replicates=reps, prep=prep,
                                    method=method, bridge=bridge, seed0=3000)
                rows.append({
                    "date": d, "rung": name, "method": method, "bridge": bridge,
                    "hybrid": hyb, "per_year": per_year, "n_dim": r["n_dim"],
                    "n_paths": 1 << k, "replicates": reps,
                    "pv": per_100(r["pv"], st.basis),
                    "se": per_100(r["se"], st.basis),
                    "rmse": per_100(r["rmse"], st.basis),
                    "err_vs_pde": per_100(r["pv"] - ref, st.basis),
                    "secs": r["secs"], "setup_secs": t_tab if hyb else 0.0,
                    "ref": per_100(ref, st.basis)})
            _log(f"  gpu {d} {name:16s} N=2^{max(ks)} rmse "
                 f"{rows[-1]['rmse']:.5f} in {rows[-1]['secs'] * 1e3:.0f}ms")
        # bias ladders: full path (whole 6y discretised) and hybrid window
        for py in BIAS_FULL:
            prep = prepare(lv, st, per_year=py)
            r = rila_replicates(lv, st, 1 << 15, replicates=12, prep=prep,
                                seed0=4000)
            brows.append({"date": d, "kind": "full", "per_year": py,
                          "n_dim": r["n_dim"], "pv": per_100(r["pv"], st.basis),
                          "se": per_100(r["se"], st.basis),
                          "ref": per_100(ref, st.basis), "secs": r["secs"]})
        for py in BIAS_WIN:
            prep = prepare(lv, st, per_year=py, tail=tab)
            r = rila_replicates(lv, st, 1 << 17, replicates=12, prep=prep,
                                seed0=4000)
            brows.append({"date": d, "kind": "hybrid", "per_year": py,
                          "n_dim": r["n_dim"], "pv": per_100(r["pv"], st.basis),
                          "se": per_100(r["se"], st.basis),
                          "ref": per_100(ref, st.basis), "secs": r["secs"]})
        _log(f"  gpu {d} bias ladders done")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(brows).to_csv(out_b, index=False)


def main():
    DATA.mkdir(exist_ok=True)
    t_all = time.time()
    _log(f"loading {CSV.name}")
    surfaces = load_term_surfaces(CSV, DATES)
    missing = [d for d in DATES if d not in surfaces]
    if missing:
        raise SystemExit(f"dates missing from the SPX file: {missing}")
    top = max(s.taus[-1] for s in surfaces.values())
    _log(f"{len(surfaces)} dates, richest term grid out to {top:.2f}y "
         f"(the 8-pillar surface tops out at 1.0y — hence the raw-row route)")
    lvs = stage_calibration(surfaces)
    stage_wings(surfaces, lvs)
    stage_states(surfaces, lvs)
    stage_cpu(surfaces, lvs)
    stage_gpu(surfaces, lvs)
    _log(f"done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
