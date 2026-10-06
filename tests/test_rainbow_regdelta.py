"""Regression (minimum-variance) delta machinery: the three deltas as three
beta surfaces through one bump seam.

Exact identities under CRN: beta = 0 is the frozen-table (sticky-moneyness)
bump; the "ss" label is the attribution_ss / book_greeks construct and must
match that driver to fp; the recuts reproduce both attribution drivers'
eq/vol/cross lines; a flat smile makes every framework coincide. CPU torch,
tiny books."""

import dataclasses
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_book import _flat_sd
from vol_pca.data import load_surfaces
from vol_pca.rainbow import spot_vol_betas, spot_panel, sticky_strike_beta
from vol_pca.rainbow_torch import (rainbow_attribution, rainbow_attribution_ss,
                                   rainbow_regression_delta)
from vol_pca.surface import MONEYNESS, TTM_PILLARS

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV3 = [ROOT / f"{n}_volSurface.csv" for n in ("SPX", "SX5E", "HSI")]
needs_data = pytest.mark.skipif(not all(p.exists() for p in CSV3),
                                reason="vol surface data not present")
SHAPE = (len(TTM_PILLARS), len(MONEYNESS))


def _planted_sd(beta, rets, vol=0.2):
    """grid_t = grid_{t-1} + beta * ret_t exactly, spot compounding rets."""
    n = len(rets) + 1
    sd = _flat_sd(n_days=n, vol=vol)
    spot = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(rets)])
    grids = np.empty((n,) + SHAPE)
    grids[0] = vol
    for t in range(1, n):
        grids[t] = grids[t - 1] + beta * rets[t - 1]
    return dataclasses.replace(sd, spot=spot, grids=grids)


def test_spot_vol_betas_recover_planted_slope():
    rng = np.random.default_rng(0)
    beta = rng.normal(-0.5, 0.2, SHAPE)
    rets = rng.normal(0, 0.01, 40)
    sd = _planted_sd(beta, rets)
    b = spot_vol_betas(sd)
    assert b.n == 40 and b.horizon == 1
    assert np.allclose(b.beta, beta, atol=1e-10)
    assert np.allclose(b.alpha, 0.0, atol=1e-12)
    assert np.allclose(b.r2, 1.0, atol=1e-10)
    # big-move filter keeps the right count and the same slope
    thr = np.quantile(np.abs(rets), 0.5)
    bb = spot_vol_betas(sd, min_abs_ret=thr)
    assert bb.n == (np.abs(rets) >= thr).sum()
    assert np.allclose(bb.beta, beta, atol=1e-10)
    # a date subset regresses over consecutive subset entries
    sub = sd.dates[::2]
    bs = spot_vol_betas(sd, dates=sub)
    assert bs.n == len(sub) - 1
    with pytest.raises(KeyError):
        spot_vol_betas(sd, dates=[np.datetime64("1999-01-01")])
    with pytest.raises(ValueError):
        spot_vol_betas(sd, horizon=0)


def test_sticky_strike_beta_flat_is_zero():
    assert np.abs(sticky_strike_beta(_flat_sd(n_days=3))).max() < 1e-10


def test_regression_delta_flat_world_identities():
    # flat smiles + moving spots: strike shift and any beta=0 bump are the
    # frozen tables, so sm == ss == zero-beta to fp; the recut vol lines
    # vanish and eq + vol + cross is label-independent
    paths = {"A": [1.00, 1.01, 0.99, 1.02, 1.005],
             "B": [1.00, 0.99, 1.01, 0.985, 1.02],
             "C": [1.00, 1.005, 0.995, 1.01, 0.99]}
    flat = {}
    for (n, p), v, s0 in zip(paths.items(), (0.2, 0.25, 0.3), (100.0, 90.0, 80.0)):
        sd = _flat_sd(n_days=5, vol=v, spot=s0)
        flat[n] = dataclasses.replace(sd, spot=s0 * np.asarray(p))
    corr = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]])
    zero = {n: np.zeros(SHAPE) for n in flat}
    tilt = {n: np.full(SHAPE, -0.5) for n in flat}        # vol falls 0.5pt per 1%
    res = rainbow_regression_delta(flat, {"z": zero, "tilt": tilt},
                                   recut=("sm", "ss", "z", "tilt"), n_paths=256,
                                   corr=corr, device="cpu", t_slice=(2, 5))
    for n in ("a", "b", "c"):
        assert np.allclose(res[f"delta_ss_{n}"], res[f"delta_sm_{n}"], atol=1e-9)
        assert np.allclose(res[f"delta_z_{n}"], res[f"delta_sm_{n}"], atol=1e-9)
        # a downward vol response makes the sold call's delta less negative
        assert (res[f"delta_tilt_{n}"] > res[f"delta_sm_{n}"]).all()
    for L in ("sm", "ss", "z"):
        assert res[f"vol_{L}"].abs().max() < 1e-6
        assert np.allclose(res[f"eq_{L}"], res["eq_sm"], atol=1e-9)
    tot = {L: res[[f"eq_{L}", f"vol_{L}", f"cross_{L}"]].sum(axis=1)
           for L in ("sm", "ss", "z", "tilt")}
    for L in ("ss", "z", "tilt"):
        assert np.allclose(tot[L], tot["sm"], atol=1e-9)
    assert (res["vol_tilt"] != 0).any()


@needs_data
def test_regression_delta_matches_attribution_drivers():
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    kw = dict(n_paths=256, seed=1, t_slice=(30, 32), device="cpu")
    ss = rainbow_attribution_ss(sds, **kw)
    sm = rainbow_attribution(sds, **kw)
    zero = {n: np.zeros(SHAPE) for n in sds}
    res = rainbow_regression_delta(sds, {"z": zero}, recut=("sm", "ss", "z"),
                                   **kw)
    tol = 1e-6
    assert np.abs(res["pl"] - ss["pl"]).max() < tol
    assert np.abs(res["pl"] - sm["pl"]).max() < tol
    # the "ss" label IS the attribution_ss delta construct
    assert np.abs(res["eq_delta_ss"] - ss["eq_delta"]).max() < tol
    for c_new, c_old in [("eq_ss", "eq"), ("vol_ss", "vol"),
                         ("cross_ss", "cross_ev"), ("vol_ss_spx", "vol_spx"),
                         ("vol_ss_hsi", "vol_hsi")]:
        assert np.abs(res[c_new] - ss[c_old]).max() < tol, c_new
    for c_new, c_old in [("eq_sm", "eq"), ("vol_sm", "vol"),
                         ("cross_sm", "cross_sv"), ("vol_sm_spx", "vol_spx"),
                         ("vol_sm_sx5e", "vol_sx5e")]:
        assert np.abs(res[c_new] - sm[c_old]).max() < tol, c_new
    # beta = 0 is the frozen-table bump, exactly
    for n in ("spx", "sx5e", "hsi"):
        assert np.allclose(res[f"delta_z_{n}"], res[f"delta_sm_{n}"], atol=1e-9)
    for c in ("eq", "vol", "cross"):
        assert np.allclose(res[f"{c}_z"], res[f"{c}_sm"], atol=1e-9)
    # the central 1% bump of the frozen tables sits close to the exact
    # pathwise (autograd) sticky-moneyness delta P&L
    gap = (res["eq_delta_sm"] - sm["eq_delta"]).abs().max()
    assert gap < 0.02 * sm["eq_delta"].abs().max() + 100.0


@needs_data
def test_sticky_strike_beta_reproduces_the_strike_shift_delta():
    # feeding the sticky-strike-implied beta (m * dsigma/dm of THAT date's
    # smile) through the regression seam must land on the strike-shift
    # delta up to the O(eps^2) curvature the linearisation drops
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    d64 = spot_panel(sds).index.values.astype("datetime64[D]")
    t = 30
    ssb = {n: sticky_strike_beta(sd, dates=[d64[t - 1]]) for n, sd in sds.items()}
    assert ssb["SPX"][-1, 6] < 0                       # negative skew at 1Y ATM
    res = rainbow_regression_delta(sds, {"ssb": ssb}, recut=(), n_paths=256,
                                   seed=1, t_slice=(t, t + 1), device="cpu")
    for n in ("spx", "sx5e", "hsi"):
        d_ss, d_b, d_sm = (float(res[f"delta_{k}_{n}"].iloc[0])
                           for k in ("ss", "ssb", "sm"))
        # the beta route recovers the SS-vs-SM gap to a few percent of it
        assert abs(d_b - d_ss) < 0.05 * abs(d_ss - d_sm) + 1e-6, (n, d_ss, d_b, d_sm)
