"""Rainbow VaR machinery: scenario semantics and engine identities.

The load-bearing checks are exact identities under CRN: an unchanged
world reprices to the same value (so scenario P&L is exactly zero), and a
pure-spot scenario goes through precisely the sticky-strike bump tables
the greeks pass uses. Everything runs on CPU torch with tiny books.
"""

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_book import _flat_sd
from vol_pca.data import load_surfaces
from vol_pca.rainbow_torch import (MarginalFactory, RainbowScenarios,
                                   book_greeks, prepare_book_greeks,
                                   price_sobol_batch, rainbow_bump_greeks,
                                   rainbow_bump_pnl, rainbow_scenario_dsigma,
                                   rainbow_scenarios, rainbow_var_full,
                                   sobol_normals)
from vol_pca.surface import MONEYNESS, TTM_PILLARS

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV3 = [ROOT / f"{n}_volSurface.csv" for n in ("SPX", "SX5E", "HSI")]
CORR = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
needs_data = pytest.mark.skipif(not all(p.exists() for p in CSV3),
                                reason="vol surface data not present")


def _flat3(n_days=6):
    return {n: _flat_sd(n_days=n_days, vol=v, spot=s)
            for n, v, s in zip(("SPX", "SX5E", "HSI"), (0.2, 0.22, 0.25),
                               (100.0, 90.0, 80.0))}


def test_scenarios_flat_world_are_identity():
    scen = rainbow_scenarios(_flat3())
    assert len(scen) == 5
    assert np.allclose(scen.ratio, 1.0)
    for n in scen.names:
        assert np.abs(scen.dgrid[n]).max() == 0.0
    assert (scen.dt > 0).all()


def test_var_full_flat_world_zero_pnl():
    # constant flat surfaces: the dt-decayed lookup reads the same vol,
    # the pricing tau is frozen, curves are frozen -> identical tables,
    # and CRN makes the scenario-vs-base difference *exactly* zero
    res = rainbow_var_full(_flat3(), n_paths=256, n_slots=8, device="cpu",
                           corr=CORR)
    assert res["n_pos"] == 5          # sold days 0..4, unexpired at t=5
    assert np.abs(res["pnl"]).max() < 1e-9


@needs_data
def test_var_full_pure_spot_scenario_is_the_ss_bump():
    # a scenario that only moves one spot must go through exactly the
    # sticky-strike bumped tables prepare_book_greeks builds: same dgrid
    # composition, same seasoning scale, same CRN points
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    t, eps, n_paths, n_slots = 40, 0.01, 256, 8
    names = list(sds)
    one = np.ones((1, 3))
    zero = {n: np.zeros((1, 8, 13)) for n in names}
    st = prepare_book_greeks(sds, asof=t, eps=eps, n_slots=n_slots,
                             device="cpu", corr=CORR)
    zs = sobol_normals(n_slots, n_paths, seed=1, device="cpu")
    pricer = lambda b, sl: price_sobol_batch(b, zs, slots=sl)
    gk = book_greeks(st, pricer)
    for j in range(3):
        ratio = one.copy()
        ratio[0, j] = 1.0 + eps
        scen = RainbowScenarios(dates=np.array(["2020-01-01"],
                                               dtype="datetime64[D]"),
                                ratio=ratio,
                                dgrid={n: zero[n] * 0 for n in names},
                                dt=np.zeros(1), names=names)
        # dgrid = the sticky-strike shift itself: the scenario surface IS
        # the strike-shifted smile, so full reval == the greek bump combo
        fac = st["fac"]
        di = {n: int(np.searchsorted(sds[n].dates,
                                     np.datetime64(st["date"], "D")))
              for n in names}
        scen.dgrid[names[j]][0] = fac.strike_shift_dgrid(
            names[j], di[names[j]], eps).cpu().numpy()
        res = rainbow_var_full(sds, scen, asof=t, n_paths=n_paths,
                               n_slots=n_slots, seed=1, device="cpu",
                               corr=CORR, fac=fac)
        want = -(gk["pv_combos"][("d", j, 1)] - gk["pv_combos"]["base"])
        assert abs(res["pnl"][0] - want) < 1e-9


@needs_data
def test_scenario_dsigma_pillar_identity():
    # u=1, dt=0: the re-anchored lookup lands exactly on the pillar nodes,
    # so dsigma reproduces the raw dgrid there (cardinal weights at the
    # knots are a kronecker delta)
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    names = list(sds)
    rng = np.random.default_rng(0)
    dg = {n: rng.normal(scale=0.01, size=(2, 8, 13)) for n in names}
    scen = RainbowScenarios(dates=np.array(["2020-01-01"] * 2,
                                           dtype="datetime64[D]"),
                            ratio=np.ones((2, 3)), dgrid=dg,
                            dt=np.zeros(2), names=names)
    ds = rainbow_scenario_dsigma(sds, scen, asof=40)
    for n in names:
        assert np.abs(ds[n] - dg[n].reshape(2, -1)).max() < 1e-9


@needs_data
def test_bump_greeks_spot_block_matches_book_greeks():
    # the VaR bump driver's +-1eps spot stencil is the same construction
    # as prepare_book_greeks/book_greeks -> identical delta and gamma
    # (same tables, same CRN points, same sold sign)
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    t, n_paths, n_slots = 40, 256, 8
    st = prepare_book_greeks(sds, asof=t, n_slots=n_slots, device="cpu",
                             corr=CORR)
    zs = sobol_normals(n_slots, n_paths, seed=1, device="cpu")
    gk = book_greeks(st, lambda b, sl: price_sobol_batch(b, zs, slots=sl))
    bg = rainbow_bump_greeks(sds, factors=[], eps=[], asof=t,
                             n_paths=n_paths, n_slots=n_slots, seed=1,
                             device="cpu", corr=CORR, fac=st["fac"])
    assert np.abs(bg.delta - gk["delta"]).max() < 1e-9
    assert np.abs(bg.gamma - gk["gamma"]).max() < 1e-9
    assert abs(bg.pv0 - gk["pv"]) < 1e-9
    assert bg.n_revals == 25            # base + 12 singles + 12 corners


@needs_data
def test_bump_pnl_reproduces_factor_move():
    # a scenario that moves the surfaces along a bumped factor direction
    # by +1 score unit is reproduced by exposure*score (+ curvature term)
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    t, n_paths, n_slots = 40, 512, 8
    names = list(sds)
    rng = np.random.default_rng(1)
    direc = {n: rng.normal(scale=0.004, size=(8, 13)) for n in names}
    eps = 1.0
    bg = rainbow_bump_greeks(sds, factors=[direc], eps=[eps], asof=t,
                             n_paths=n_paths, n_slots=n_slots, seed=1,
                             device="cpu", corr=CORR)
    scen = RainbowScenarios(dates=np.array(["2020-01-01"],
                                           dtype="datetime64[D]"),
                            ratio=np.ones((1, 3)),
                            dgrid={n: direc[n][None] * eps for n in names},
                            dt=np.zeros(1), names=names)
    res = rainbow_var_full(sds, scen, asof=t, n_paths=n_paths,
                           n_slots=n_slots, seed=1, device="cpu", corr=CORR)
    est = rainbow_bump_pnl(bg, scen.ratio, scores=np.array([[eps]]),
                           cube=False, quad="all")
    # exposure*f + curvature*f^2/2 at f = eps telescopes to exactly the
    # +eps revaluation (central-difference algebra), and the full-reval
    # engine builds those very tables -> identity to float noise
    assert abs(est[0] - res["pnl"][0]) < 1e-6
