"""PDE-vs-MC demonstration pass: does full-path Monte Carlo converge to the
deterministic PDE reference for the RILA reset package?

Companion to run_rila.py — same five dates, same calibration defaults, same
house quoting (per 100 of basis). The claim under test: `pde_reset_price` is
the value of record, and plain full-path Monte Carlo converges to it in the
JOINT limit (paths -> inf kills sampling error, steps/year -> inf kills the
log-Euler discretisation bias; the PDE integrates the same local-vol model
with neither). Newly issued AND seasoned states, because the ratchet gather
and the K-axis interpolation only bite when fixings remain / the basis is
off-spot.

    uv run python scripts/run_rila_pde_mc.py

Stages and caches (data/, gitignored, skip-if-exists per file):

  1. references  rila_pdemc_ref.csv     PDE value per (date, state) at the
                                        production config + a refined-grid
                                        solve (reference self-convergence)
                                        + surface reads for post-window states
  2. funnel      rila_pdemc_funnel.csv  full-path MC in N at daily steps:
                                        GPU pseudo (every state), CPU plain
                                        (the canonical numpy engine, anchor
                                        points), Sobol+BB (newly issued —
                                        same limit, faster approach)
  3. bias        rila_pdemc_bias.csv    full-path MC vs steps/year at fixed
                                        N — the Euler-bias axis; first-order,
                                        and the COVID surface is the stress
                                        case (sigma_loc reaches ~324%)

Wall clock on the reference machine (RTX 4070 Laptop, WSL2): ~15-20 min cold.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vol_pca.local_vol import (dupire_local_vol, european_package_surface,
                               load_term_surfaces, mc_step_data, mc_time_grid,
                               pde_package, pde_reset_price, per_100,
                               price_rila_mc, rila_state)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV = ROOT / "SPX_volSurface.csv"

DATES = ["2020-03-16", "2021-09-20", "2022-03-16", "2024-10-08", "2026-07-31"]
STUDY = ["2026-07-31", "2020-03-16"]          # deep-dive dates (calm, COVID)
WING = "taper"

# production PDE config (run_rila.py's PDE_REF) and a refined-grid solve for
# the reference's own convergence check
PDE_REF = dict(n_k=145, n_x=1201, per_year=320)
PDE_FINE = dict(n_k=217, n_x=1601, per_year=480)
PDE_EUR = dict(n_x=1201, per_year=320)
PDE_EUR_FINE = dict(n_x=1601, per_year=480)

# (state id, rila_state kwargs): newly issued, two mid-window seasoned states
# (basis below spot -> next fixing almost surely ratchets; basis above spot ->
# ratchet nearly dead, put leg closer), one post-window seasoned state (pure
# European under LV, priced by full-path MC all the same)
STATES = [
    ("new", dict(resets_left=6, k_over_s=1.00)),
    ("mid_r4_k095", dict(resets_left=4, k_over_s=0.95)),
    ("mid_r2_k108", dict(resets_left=2, k_over_s=1.08)),
    ("post_3y_k090", dict(resets_left=0, k_over_s=0.90, ttm=3.0)),
]

FUNNEL_KS = range(12, 22)                     # N = 2^12 .. 2^21
SOBOL_KS = range(12, 18)
CPU_NS = (1 << 14, 1 << 16)
BIAS_PY = (12, 52, 126, 252, 504)
BIAS_N, BIAS_REPS = 1 << 17, 8


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _reps(k):
    return 16 if k <= 15 else (12 if k <= 18 else 8)


def build_lvs(surfaces):
    return {d: dupire_local_vol(surfaces[d], wing=WING) for d in DATES}


# ------------------------------------------------------- 1. PDE references

def stage_refs(surfaces, lvs):
    out = DATA / "rila_pdemc_ref.csv"
    if out.exists():
        _log("references cached")
        return pd.read_csv(out)
    rows = []
    for d in DATES:
        ts, lv = surfaces[d], lvs[d]
        for sid, kw in STATES:
            st = rila_state(ts, **kw)
            t0 = time.time()
            if st.fixings:
                ref = pde_reset_price(lv, st, **PDE_REF)["pv"]
            else:
                ref = pde_package(lv, st.basis, st.ttm, **PDE_EUR)
            t_ref = time.time() - t0
            fine = np.nan
            t_fine = np.nan
            if d in STUDY or sid == "new":
                t0 = time.time()
                fine = (pde_reset_price(lv, st, **PDE_FINE)["pv"] if st.fixings
                        else pde_package(lv, st.basis, st.ttm, **PDE_EUR_FINE))
                t_fine = time.time() - t0
            surf = (european_package_surface(ts, st.basis, st.ttm, wing=WING)["pv"]
                    if not st.fixings else np.nan)
            rows.append({
                "date": d, "state": sid, "resets_left": st.resets_left,
                "k_over_s": st.k_over_s, "ttm": st.ttm, "basis": st.basis,
                "spot": ts.spot, "pde": per_100(ref, st.basis),
                "pde_fine": per_100(fine, st.basis) if np.isfinite(fine) else np.nan,
                "surface": per_100(surf, st.basis) if np.isfinite(surf) else np.nan,
                "pde_secs": t_ref, "fine_secs": t_fine})
            _log(f"  ref {d} {sid:12s} pde {rows[-1]['pde']:9.4f}"
                 + (f"  fine-ref {rows[-1]['pde_fine'] - rows[-1]['pde']:+.5f}"
                    if np.isfinite(fine) else ""))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


# ------------------------------------------------------------- 2. funnel

def stage_funnel(surfaces, lvs, refs):
    out = DATA / "rila_pdemc_funnel.csv"
    if out.exists():
        _log("funnel cached")
        return
    from vol_pca.rila_torch import prepare, rila_replicates
    rows = []
    idx = 0
    for d in DATES:
        ts, lv = surfaces[d], lvs[d]
        for sid, kw in STATES:
            st = rila_state(ts, **kw)
            ref = float(refs.set_index(["date", "state"]).loc[(d, sid), "pde"])
            idx += 1
            prep = prepare(lv, st, per_year=252)
            n_steps = len(prep["t_grid"]) - 1
            for k in FUNNEL_KS:
                r = rila_replicates(lv, st, 1 << k, replicates=_reps(k),
                                    prep=prep, method="pseudo", bridge=False,
                                    seed0=3000 + 137 * idx)
                rows.append({
                    "date": d, "state": sid, "rung": "gpu_pseudo",
                    "n_paths": 1 << k, "replicates": _reps(k),
                    "n_steps": n_steps,
                    "pv": per_100(r["pv"], st.basis),
                    "se": per_100(r["se"], st.basis),
                    "rmse": per_100(r["rmse"], st.basis),
                    "err_vs_pde": per_100(r["pv"], st.basis) - ref,
                    "ref": ref, "secs": r["secs"]})
            _log(f"  funnel {d} {sid:12s} gpu_pseudo N=2^{max(FUNNEL_KS)} "
                 f"gap {rows[-1]['err_vs_pde']:+.4f} +- {2 * rows[-1]['se']:.4f}")
            if sid == "new":
                for k in SOBOL_KS:
                    r = rila_replicates(lv, st, 1 << k, replicates=12,
                                        prep=prep, method="sobol", bridge=True,
                                        seed0=5000 + 137 * idx)
                    rows.append({
                        "date": d, "state": sid, "rung": "gpu_sobol_bb",
                        "n_paths": 1 << k, "replicates": 12,
                        "n_steps": n_steps,
                        "pv": per_100(r["pv"], st.basis),
                        "se": per_100(r["se"], st.basis),
                        "rmse": per_100(r["rmse"], st.basis),
                        "err_vs_pde": per_100(r["pv"], st.basis) - ref,
                        "ref": ref, "secs": r["secs"]})
                _log(f"  funnel {d} {sid:12s} sobol_bb   N=2^{max(SOBOL_KS)} "
                     f"gap {rows[-1]['err_vs_pde']:+.4f} +- {2 * rows[-1]['se']:.4f}")
            # CPU anchor: the canonical numpy engine, same estimator
            t_grid, fix_idx = mc_time_grid(st, 252)
            step = mc_step_data(lv, t_grid)
            for n in CPU_NS:
                vals, ses, secs = [], [], []
                for s in range(3):
                    t0 = time.time()
                    m = price_rila_mc(lv, st, n_paths=n, seed=1000 + s,
                                      t_grid=t_grid, fix_idx=fix_idx, step=step)
                    secs.append(time.time() - t0)
                    vals.append(m["pv"])
                    ses.append(m["se"])
                v = np.asarray(vals)
                rows.append({
                    "date": d, "state": sid, "rung": "cpu_plain",
                    "n_paths": n, "replicates": 3,
                    "n_steps": len(t_grid) - 1,
                    "pv": per_100(v.mean(), st.basis),
                    "se": per_100(np.mean(ses) / np.sqrt(3), st.basis),
                    "rmse": per_100(np.mean(ses), st.basis),
                    "err_vs_pde": per_100(v.mean(), st.basis) - ref,
                    "ref": ref, "secs": float(np.median(secs))})
    pd.DataFrame(rows).to_csv(out, index=False)
    _log("wrote rila_pdemc_funnel.csv")


# --------------------------------------------------------------- 3. bias

def stage_bias(surfaces, lvs, refs):
    out = DATA / "rila_pdemc_bias.csv"
    if out.exists():
        _log("bias ladder cached")
        return
    from vol_pca.rila_torch import prepare, rila_replicates
    rows = []
    configs = [(d, "new") for d in DATES] + [(d, "mid_r2_k108") for d in STUDY]
    kwmap = dict(STATES)
    for i, (d, sid) in enumerate(configs):
        ts, lv = surfaces[d], lvs[d]
        st = rila_state(ts, **kwmap[sid])
        ref = float(refs.set_index(["date", "state"]).loc[(d, sid), "pde"])
        for py in BIAS_PY:
            prep = prepare(lv, st, per_year=py)
            r = rila_replicates(lv, st, BIAS_N, replicates=BIAS_REPS,
                                prep=prep, method="sobol", bridge=True,
                                seed0=8000 + 211 * i)
            rows.append({
                "date": d, "state": sid, "per_year": py,
                "n_steps": len(prep["t_grid"]) - 1,
                "pv": per_100(r["pv"], st.basis),
                "se": per_100(r["se"], st.basis),
                "err_vs_pde": per_100(r["pv"], st.basis) - ref,
                "ref": ref, "secs": r["secs"]})
        _log(f"  bias {d} {sid:12s} err(252) "
             f"{rows[-2]['err_vs_pde']:+.4f}  err(504) {rows[-1]['err_vs_pde']:+.4f}")
    pd.DataFrame(rows).to_csv(out, index=False)
    _log("wrote rila_pdemc_bias.csv")


def main():
    DATA.mkdir(exist_ok=True)
    t_all = time.time()
    import torch
    _log(f"cuda available: {torch.cuda.is_available()}")
    _log(f"loading {CSV.name}")
    surfaces = load_term_surfaces(CSV, DATES)
    lvs = build_lvs(surfaces)
    _log("calibrations built (taper)")
    refs = stage_refs(surfaces, lvs)
    stage_funnel(surfaces, lvs, refs)
    stage_bias(surfaces, lvs, refs)
    _log(f"done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
