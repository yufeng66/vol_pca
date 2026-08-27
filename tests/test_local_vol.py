import pathlib
import subprocess
import sys

import numpy as np
import pytest

from vol_pca.local_vol import (K_CALL_HI, K_CALL_LO, K_PUT, RilaState,
                               _flat_package_bs, cn_rollback,
                               dupire_local_vol, european_package_surface,
                               flat_local_vol, flat_reset_oracle,
                               flat_term_surface, load_term_surfaces,
                               mc_time_grid, package_payoff, pde_package,
                               pde_reprice_table, pde_reset_price,
                               pde_tail_table, pde_vanilla_calls, per_100,
                               price_rila_mc, rila_state, rila_value)
from vol_pca.pricing import black76

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "SPX_volSurface.csv"
_CACHE = {}


def _spx(date="2026-07-31"):
    if date not in _CACHE:
        _CACHE[date] = load_term_surfaces(CSV, [date])[date]
    return _CACHE[date]


# ---------------------------------------------------------------- product

def test_package_payoff_legs():
    k = 100.0
    s = np.array([50.0, 80.0, 100.0, 150.0, 200.0, 300.0])
    p = package_payoff(s, k)
    assert np.allclose(p, [30.0, 0.0, 0.0, -50.0, -100.0, -100.0])
    # above 2K the two calls net to the constant -K: the linearity boundary
    # condition the PDE leans on
    assert np.allclose(package_payoff(np.array([500.0, 1000.0]), k), -k)


def test_rila_state_schedule():
    ts = flat_term_surface()
    st = rila_state(ts, 6, 1.0)
    assert st.ttm == 6.0 and st.resets_left == 6
    assert np.allclose(st.fixings, np.arange(1, 7) / 12.0)
    st4 = rila_state(ts, 4, 1.08)
    assert abs(st4.ttm - (6.0 - 2 / 12)) < 1e-12
    assert np.allclose(st4.fixings, np.arange(1, 5) / 12.0)
    assert abs(st4.k_over_s - 1.08) < 1e-12
    assert rila_state(ts, 0, 1.0, ttm=3.0).fixings == ()


# ------------------------------------------------------ 2. Dupire on flat

def test_flat_surface_gives_constant_local_vol():
    ts = flat_term_surface(vol=0.23, spot=100.0, rate=0.03, div=0.015)
    lv = dupire_local_vol(ts)
    assert np.abs(lv.sig - 0.23).max() < 1e-9
    assert lv.diag["n_floor"] == 0 and lv.diag["n_cap"] == 0
    assert lv.diag["n_calendar_arb"] == 0 and lv.diag["n_bfly_arb"] == 0
    # total variance is exactly sigma^2 tau on the whole grid
    assert np.abs(lv.w - 0.23**2 * lv.taus[:, None]).max() < 1e-12


def test_flat_surface_local_vol_under_both_wing_conventions():
    ts = flat_term_surface(vol=0.19)
    for wing in ("clamp", "taper"):
        lv = dupire_local_vol(ts, wing=wing)
        assert np.abs(lv.sig - 0.19).max() < 1e-9


# ---------------------------------------- 3. PDE vs Black-76 / the surface

def test_pde_reprices_black76_on_flat_surface():
    ts = flat_term_surface(vol=0.22, spot=100.0, rate=0.03, div=0.015)
    lv = flat_local_vol(vol=0.22, spot=100.0, rate=0.03, div=0.015)
    strikes = np.array([70.0, 85.0, 100.0, 120.0, 150.0])
    for tau in (1.0, 6.0):
        ref = black76(ts.forward(tau), strikes, tau, 0.22, ts.discount(tau))
        pde = pde_vanilla_calls(lv, tau, strikes, n_x=601, per_year=100)
        assert np.abs(pde - ref).max() < 2e-3          # 0.2bp of spot


def test_pde_package_matches_direct_surface_read_on_flat_surface():
    ts = flat_term_surface(vol=0.22, spot=100.0, rate=0.03, div=0.015)
    lv = flat_local_vol(vol=0.22, spot=100.0, rate=0.03, div=0.015)
    eur = european_package_surface(ts, 100.0, 6.0)
    assert abs(pde_package(lv, 100.0, 6.0, n_x=801, per_year=160)
               - eur["pv"]) < 5e-3
    # legs: only the 100-strike call is short
    assert eur["call_100"] < 0 and eur["put_80"] > 0 and eur["call_200"] > 0


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_pde_reprices_market_surface_within_tolerance():
    ts = _spx("2022-03-16")                                   # calm date
    lv = dupire_local_vol(ts)
    taus = ts.taus[(ts.taus >= 0.5) & (ts.taus <= 6.0)]
    tab = pde_reprice_table(lv, ts, taus=taus, n_x=601, per_year=100)
    err = tab["d_vol_pts"].abs()
    assert err.mean() < 0.2 and err.max() < 0.6              # vol points


# ---------------------------------------------------- 1. flat reset oracle

def test_flat_reset_oracle_matches_mc():
    vol, rate, div, spot = 0.22, 0.03, 0.015, 100.0
    ts = flat_term_surface(vol=vol, spot=spot, rate=rate, div=div)
    lv = flat_local_vol(vol=vol, spot=spot, rate=rate, div=div)
    st = rila_state(ts, 6, 1.0)
    oracle = st.basis * flat_reset_oracle(vol, rate, div, st.ttm, st.fixings)
    # with sigma constant the log-Euler scheme is EXACT (the log increment is
    # exactly normal), so any step count is unbiased and the whole gap is MC
    # noise; the frozen-basis control variate (exact Black value, no PDE)
    # tightens it to a real test
    exact_frozen = st.basis * _flat_package_bs(1.0, st.ttm, vol, rate, div)
    mc = price_rila_mc(lv, st, n_paths=1 << 16, per_year=12, seed=3,
                       cv="frozen", cv_exact=exact_frozen)
    assert abs(mc["pv"] - oracle) < 3 * mc["se"]
    assert mc["se"] < 0.05 * abs(oracle)


def test_flat_reset_oracle_matches_pde_reference():
    vol, rate, div = 0.22, 0.03, 0.015
    ts = flat_term_surface(vol=vol, spot=100.0, rate=rate, div=div)
    lv = flat_local_vol(vol=vol, spot=100.0, rate=rate, div=div)
    st = rila_state(ts, 6, 1.0)
    oracle = 100.0 * flat_reset_oracle(vol, rate, div, st.ttm, st.fixings)
    pde = pde_reset_price(lv, st, n_k=97, n_x=801, per_year=200)["pv"]
    assert abs(per_100(pde, st.basis) - oracle) < 0.02       # 2bp of spot


def test_flat_reset_oracle_seasoned_and_monotone():
    vol, rate, div = 0.20, 0.02, 0.01
    fix = tuple(np.arange(1, 5) / 12.0)
    v = [flat_reset_oracle(vol, rate, div, 6.0 - 2 / 12, fix, x0=x)
         for x in (0.9, 1.0, 1.1)]
    # the package is short the 100 call: value falls as spot rises vs basis
    assert v[0] > v[1] > v[2]
    # more resets ahead is worth more (the ratchet only ever helps)
    a = flat_reset_oracle(vol, rate, div, 6.0, tuple(np.arange(1, 7) / 12.0))
    b = flat_reset_oracle(vol, rate, div, 6.0, (1 / 12.0,))
    assert a > b


# ------------------------------------- 4. no-reset MC vs the PDE European

@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_frozen_basis_mc_matches_pde_european():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    pde = pde_package(lv, st.basis, st.ttm, n_x=801, per_year=160)
    mc = price_rila_mc(lv, st, n_paths=1 << 16, per_year=52, seed=5, reset=False)
    assert abs(mc["pv"] - pde) < 3 * mc["se"]


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_martingale_forward():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    mc = price_rila_mc(lv, st, n_paths=1 << 16, per_year=52, seed=11)
    # log-Euler keeps S/F an exact martingale (sigma_k is F_k-measurable),
    # so this is pure MC error at ~1/sqrt(N)
    assert abs(mc["mean_s"] / mc["fwd"] - 1.0) < 0.01


# --------------------------------------- 5. post-window = surface identity

@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_post_window_state_is_a_direct_surface_read():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 0, 0.9, ttm=3.0)
    out = rila_value(ts, lv, st)
    assert out["engine"] == "surface" and st.fixings == ()
    ref = european_package_surface(ts, st.basis, st.ttm)
    assert out["pv"] == ref["pv"]                     # identity, not a fit
    # and the LV engine lands on it (calibration anchor)
    assert abs(pde_package(lv, st.basis, st.ttm, n_x=801, per_year=160)
               - ref["pv"]) < 0.002 * st.basis


# ------------------------------------------------------------- mechanics

def test_time_grid_contains_every_fixing_exactly():
    ts = flat_term_surface()
    for r in (6, 4, 2):
        st = rila_state(ts, r, 1.0)
        for per_year in (12, 52, 252):
            t, idx = mc_time_grid(st, per_year)
            assert np.all(np.diff(t) > 0) and t[0] == 0.0 and t[-1] == st.ttm
            assert np.allclose(t[idx], st.fixings)


def test_cn_rollback_is_exact_on_an_affine_payoff():
    # V_T = a S_T + b has the closed-form solution
    # a S D(t,T) F(T)/F(t) + b D(t,T) for ANY local vol — and the linearity
    # boundary condition is exact for it, so this pins drift, discounting and
    # both boundaries at once
    lv = flat_local_vol(vol=0.30, spot=100.0, rate=0.04, div=0.017)
    x = np.linspace(np.log(20.0), np.log(500.0), 401)
    s = np.exp(x)
    a, b = 1.5, -37.0
    out = cn_rollback(lv, (a * s + b)[:, None], x, np.linspace(2.0, 0.5, 61))
    d = np.exp(lv.ln_df(2.0) - lv.ln_df(0.5))
    fr = np.exp(lv.ln_fwd(2.0) - lv.ln_fwd(0.5))
    ref = d * (a * s * fr + b)
    # exact up to the interior stencil's O(dx^2) treatment of e^x (the BC
    # itself is exact here), so ~7 significant digits at dx = 0.008
    assert np.abs(out[:, 0] - ref).max() < 1e-6 * np.abs(ref).max()


def test_tail_table_matches_black_scholes_in_the_flat_world():
    vol, rate, div = 0.22, 0.03, 0.015
    ts = flat_term_surface(vol=vol, spot=100.0, rate=rate, div=div)
    lv = flat_local_vol(vol=vol, spot=100.0, rate=rate, div=div)
    st = rila_state(ts, 6, 1.0)
    tab = pde_tail_table(lv, st, n_k=25, n_x=601, per_year=120)
    assert tab["t_window"] == st.fixings[-1]
    tail_ttm = st.ttm - tab["t_window"]
    x, k = tab["x"], tab["k"]
    j = np.arange(len(x))[(x > np.log(40.0)) & (x < np.log(260.0))]
    for m in (0, 12, 24):
        ref = k[m] * _flat_package_bs(np.exp(x[j]) / k[m], tail_ttm, vol, rate, div)
        assert np.abs(tab["v"][j, m] - ref).max() < 0.02       # 2bp of spot


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_tail_table_hits_its_asymptotes():
    ts = _spx()
    lv = dupire_local_vol(ts)
    st = rila_state(ts, 6, 1.0)
    tab = pde_tail_table(lv, st, n_k=49, n_x=601, per_year=120)
    d = ts.discount(st.ttm) / ts.discount(tab["t_window"])
    fr = ts.fwd_ratio(st.ttm) / ts.fwd_ratio(tab["t_window"])
    k, s_hi, s_lo = tab["k"], np.exp(tab["x"][-1]), np.exp(tab["x"][0])
    assert np.abs(tab["v"][-1] / (-k * d) - 1.0).max() < 1e-3   # -> -K D
    lo_ref = d * (K_PUT * k - s_lo * fr)                        # -> 0.8K D - S
    assert np.abs(tab["v"][0] / lo_ref - 1.0).max() < 1e-3


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_wing_convention_moves_the_200_call():
    ts = _spx()
    lo = european_package_surface(ts, ts.spot, 6.0, wing="clamp")
    hi = european_package_surface(ts, ts.spot, 6.0, wing="taper")
    # the 80 put and 100 call are inside the quoted range: identical
    assert abs(lo["put_80"] - hi["put_80"]) < 1e-9
    assert abs(lo["call_100"] - hi["call_100"]) < 1e-9
    # the 200 call is outside it: the falling call wing makes the taper cheaper
    assert hi["call_200"] < lo["call_200"]


def test_local_vol_imports_without_torch():
    code = ("import sys, vol_pca.local_vol as m; "
            "assert 'torch' not in sys.modules, sorted(sys.modules)[:0]; "
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr
