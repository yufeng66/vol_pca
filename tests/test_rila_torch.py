import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vol_pca.local_vol import (_flat_package_bs, dupire_local_vol,
                               flat_local_vol, flat_reset_oracle,
                               flat_term_surface, load_term_surfaces,
                               mc_time_grid, pde_package, pde_reset_price,
                               pde_tail_table, per_100, price_rila_mc,
                               rila_state)
from vol_pca.rila_torch import (DeviceLV, TailTable, bb_increments, bb_plan,
                                make_tail_table, prepare, rila_price_torch,
                                rila_replicates)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "SPX_volSurface.csv"
_CACHE = {}


def _spx(date="2026-07-31"):
    if date not in _CACHE:
        _CACHE[date] = load_term_surfaces(CSV, [date])[date]
    return _CACHE[date]


def _flat():
    vol, rate, div = 0.22, 0.03, 0.015
    ts = flat_term_surface(vol=vol, spot=100.0, rate=rate, div=div)
    lv = flat_local_vol(vol=vol, spot=100.0, rate=rate, div=div)
    return ts, lv, (vol, rate, div)


# --------------------------------------------------- Brownian bridge plan

def test_bb_reproduces_brownian_covariance():
    t = np.linspace(0.0, 2.0, 17)[1:]
    plan = bb_plan(t)
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    z = torch.randn(200_000, len(t), generator=g, dtype=torch.float64)
    dw = bb_increments(z, plan)
    w = torch.cumsum(dw, dim=1).numpy()
    cov = np.cov(w.T)
    ref = np.minimum(t[:, None], t[None, :])
    assert np.abs(cov - ref).max() < 0.02
    # coordinate 0 must carry the terminal value: it explains ~all of W(T)
    assert abs(np.corrcoef(z[:, 0].numpy(), w[:, -1])[0, 1] - 1.0) < 1e-12


def test_bb_is_a_pure_reordering_of_the_same_law():
    ts, lv, _ = _flat()
    st = rila_state(ts, 6, 1.0)
    prep = prepare(lv, st, per_year=12, dtype=torch.float64)
    a = rila_replicates(lv, st, 1 << 13, replicates=8, method="pseudo",
                        bridge=False, prep=prep, seed0=10)
    b = rila_replicates(lv, st, 1 << 13, replicates=8, method="pseudo",
                        bridge=True, prep=prep, seed0=90)
    assert abs(a["pv"] - b["pv"]) < 3 * np.hypot(a["se"], b["se"])


# --------------------------------------------------------- 6. CRN parity

@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_cpu_gpu_crn_parity():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    t_grid, fix_idx = mc_time_grid(st, 12)
    nrm = np.random.default_rng(7).standard_normal((4096, len(t_grid) - 1))
    cpu = price_rila_mc(lv, st, n_paths=4096, t_grid=t_grid, fix_idx=fix_idx,
                        normals=nrm)
    for dtype, tol in ((torch.float64, 1e-10), (torch.float32, 1e-5)):
        gpu = rila_price_torch(lv, st, n_paths=4096, normals=nrm, dtype=dtype,
                               t_grid=t_grid, fix_idx=fix_idx)
        assert abs(gpu["pv"] / cpu["pv"] - 1.0) < tol


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_fp32_matches_fp64_under_scrambled_sobol():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    out = {}
    for dtype in (torch.float32, torch.float64):
        prep = prepare(lv, st, per_year=52, dtype=dtype)
        out[dtype] = rila_replicates(lv, st, 1 << 13, replicates=8, prep=prep,
                                     seed0=5)
    a, b = out[torch.float32], out[torch.float64]
    # same scrambles both times, so the two runs share their sampling error:
    # the residual is float precision only, orders below the QMC error bar
    assert abs(a["pv"] - b["pv"]) < 0.05 * max(a["rmse"], 1e-12)


# ------------------------------------------------- flat-world oracle rung

def test_hybrid_matches_the_flat_reset_oracle():
    ts, lv, (vol, rate, div) = _flat()
    st = rila_state(ts, 6, 1.0)
    oracle = st.basis * flat_reset_oracle(vol, rate, div, st.ttm, st.fixings)
    tab = make_tail_table(lv, st, n_k=49, n_x=601, per_year=160,
                          dtype=torch.float64)
    r = rila_replicates(lv, st, 1 << 14, replicates=12, per_year=252, tail=tab,
                        dtype=torch.float64, seed0=21)
    assert abs(r["pv"] - oracle) < max(3 * r["se"], 0.01)     # 1bp of spot


def test_tail_table_lookup_reproduces_black_scholes():
    ts, lv, (vol, rate, div) = _flat()
    st = rila_state(ts, 6, 1.0)
    tab = pde_tail_table(lv, st, n_k=25, n_x=601, per_year=120)
    dev = TailTable(tab, dtype=torch.float64)
    s = np.linspace(60.0, 190.0, 41)
    k = np.full_like(s, float(tab["k"][12]))
    got = dev.lookup(torch.as_tensor(np.log(s), dtype=torch.float64,
                                     device=dev.x.device),
                     torch.as_tensor(k, dtype=torch.float64,
                                     device=dev.x.device)).cpu().numpy()
    ref = k * _flat_package_bs(s / k, st.ttm - tab["t_window"], vol, rate, div)
    assert np.abs(got - ref).max() < 0.05


# --------------------------------------------- hybrid vs full-path parity

@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_hybrid_matches_full_path_within_ci():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    tab = make_tail_table(lv, st, n_k=49, n_x=801, per_year=200)
    hyb = rila_replicates(lv, st, 1 << 14, replicates=12, per_year=252,
                          tail=tab, seed0=31)
    full = rila_replicates(lv, st, 1 << 14, replicates=12, per_year=252,
                           seed0=41)
    assert hyb["n_dim"] == 126 and full["n_dim"] == 1512
    assert abs(hyb["pv"] - full["pv"]) < 3 * np.hypot(hyb["se"], full["se"])
    # the whole point of the rung: far smaller error at far lower cost
    assert hyb["rmse"] < 0.25 * full["rmse"]


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_hybrid_seasoned_state_matches_pde_reference():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 4, 1.08)
    assert st.window_end == pytest.approx(4 / 12)
    ref = pde_reset_price(lv, st, n_k=97, n_x=801, per_year=200)["pv"]
    tab = make_tail_table(lv, st, n_k=97, n_x=801, per_year=200)
    r = rila_replicates(lv, st, 1 << 14, replicates=12, per_year=252, tail=tab,
                        seed0=51)
    assert r["n_dim"] == 84
    assert abs(per_100(r["pv"] - ref, st.basis)) < 0.05        # 5bp of spot


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_frozen_basis_hybrid_is_the_pde_european():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    tab = make_tail_table(lv, st, n_k=49, n_x=801, per_year=200)
    eur = pde_package(lv, st.basis, st.ttm, n_x=801, per_year=200)
    r = rila_replicates(lv, st, 1 << 14, replicates=12, per_year=252, tail=tab,
                        reset=False, seed0=61)
    # with the ratchet off the tail table is integrated against the marginal
    # of S_w alone: a pure tower-property identity, no path dependence left
    assert abs(per_100(r["pv"] - eur, st.basis)) < 0.05


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_device_martingale():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    out = rila_price_torch(lv, st, n_paths=1 << 15, per_year=52, seed=3)
    assert abs(out["mean_s"] / out["fwd"] - 1.0) < 0.01


def test_device_lv_mirrors_the_numpy_lookup():
    ts, lv, _ = _flat()
    st = rila_state(ts, 6, 1.0)
    t_grid, _ = mc_time_grid(st, 12)
    d = DeviceLV(lv, t_grid, dtype=torch.float64)
    q = torch.linspace(-1.5, 1.5, 33, dtype=torch.float64, device=d.rows.device)
    for k in (0, 5, 40):
        got = d.sigma(k, q).cpu().numpy()
        ref = lv.sigma_at(t_grid[k], lv.spot * np.exp(q.cpu().numpy()))
        assert np.abs(got - ref).max() < 1e-12
