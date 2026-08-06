import pathlib

import numpy as np
import pytest
from scipy.stats import norm

from tests.test_book import _flat_sd
from vol_pca.data import load_surfaces
from vol_pca.pricing import black76
from vol_pca.rainbow import (bs_implied_vol, call_spread_payoff, date_index,
                             historical_corr, implied_marginal,
                             marginal_call_spread, price_rainbow,
                             price_rainbow_quad, rainbow_basket,
                             sample_performances, spot_panel)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "SPX_volSurface.csv"
CSV3 = [ROOT / f"{n}_volSurface.csv" for n in ("SPX", "SX5E", "HSI")]


def test_flat_vol_marginal_is_lognormal():
    # flat 20% surface, zero rates: the implied marginal must be the exact
    # lognormal with mean 1, recovered through the numerical BL derivatives
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    exact = norm.cdf((np.log(marg.x) + 0.5 * 0.2**2) / 0.2)
    assert np.abs(marg.cdf - exact).max() < 1e-5
    assert abs(marg.mean - 1.0) < 1e-5
    assert abs(marg.fwd_ratio - 1.0) < 1e-12
    z = np.linspace(-3, 3, 25)
    assert np.allclose(marg.quantile(norm.cdf(z)),
                       np.exp(-0.5 * 0.2**2 + 0.2 * z), rtol=1e-4)


def test_marginal_density_integrates_to_one():
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    assert abs(np.trapezoid(marg.pdf, marg.x) - 1.0) < 1e-5
    assert marg.cdf[0] < 1e-10 and marg.cdf[-1] > 1.0 - 1e-10


def test_rainbow_basket_rank_identity():
    # 0.5 max + 0.3 min == 0.4 (p1 + p2) + 0.1 |p1 - p2|
    rng = np.random.default_rng(0)
    perf = rng.lognormal(0.0, 0.2, (500, 3))
    b = rainbow_basket(perf)
    plain = 0.4 * (perf[:, 0] + perf[:, 1]) + 0.2 * perf[:, 2]
    assert np.allclose(b, plain + 0.1 * np.abs(perf[:, 0] - perf[:, 1]))
    assert np.allclose(call_spread_payoff(np.array([0.9, 1.05, 1.5])),
                       [0.0, 0.05, 0.12])


def test_copula_hits_target_dependence():
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    tight = np.array([[1.0, 1.0 - 1e-6, 0.0], [1.0 - 1e-6, 1.0, 0.0], [0.0, 0.0, 1.0]])
    perf = sample_performances([marg] * 3, tight, 100_000, seed=1)
    r = np.log(perf)
    c = np.corrcoef(r.T)
    assert c[0, 1] > 0.999
    assert abs(c[0, 2]) < 0.02 and abs(c[1, 2]) < 0.02
    # marginals survive the copula: sample mean/std match the lognormal
    assert np.allclose(perf.mean(axis=0), 1.0, atol=3e-3)
    assert np.allclose(perf.std(axis=0), np.sqrt(np.exp(0.2**2) - 1.0), atol=4e-3)


def test_mc_single_index_matches_quadrature():
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    perf = sample_performances([marg] * 3, np.eye(3), 200_000, seed=2)
    pay = call_spread_payoff(perf[:, 0])
    quad = marginal_call_spread(marg, df=1.0)
    se = pay.std() / np.sqrt(len(pay))
    assert abs(pay.mean() - quad) < 4 * se


def test_historical_corr_recovers_generator():
    rng = np.random.default_rng(3)
    n, rho = 4000, 0.6
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    r = 0.01 * rng.standard_normal((n, 2)) @ chol.T
    import pandas as pd
    spots = pd.DataFrame(100.0 * np.exp(np.cumsum(r, axis=0)), columns=["a", "b"],
                         index=pd.bdate_range("2020-01-01", periods=n))
    for h in (1, 5):
        assert abs(historical_corr(spots, horizon=h).iloc[0, 1] - rho) < 0.06


def test_quad_matches_mc_flat():
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    corr = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    q = price_rainbow_quad([marg] * 3, corr, df=0.96, n_outer=64, n_inner=64)
    p = price_rainbow([marg] * 3, corr, df=0.96, n_paths=2**19, seed=7,
                      method="sobol")
    assert abs(q["pv"] - p["pv"]) < 20.0            # dollars on $1M notional
    q2 = price_rainbow_quad([marg] * 3, corr, df=0.96, n_outer=48, n_inner=48)
    assert abs(q["pv"] - q2["pv"]) < 5.0            # rule-doubling stability


def test_seasoned_quad_matches_mc():
    # performance-to-date factors displace the basket well off the strikes;
    # the quadrature must track the MC engine there too, and neutral factors
    # must reproduce the fresh price exactly
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 0.25)
    corr = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    g = (1.15, 0.92, 1.05)
    q = price_rainbow_quad([marg] * 3, corr, df=0.99, n_outer=64, n_inner=64,
                           perf_to_date=g)
    p = price_rainbow([marg] * 3, corr, df=0.99, n_paths=2**19, seed=3,
                      method="sobol", perf_to_date=g)
    assert abs(q["pv"] - p["pv"]) < 25.0
    fresh = price_rainbow_quad([marg] * 3, corr, df=0.99)
    neutral = price_rainbow_quad([marg] * 3, corr, df=0.99,
                                 perf_to_date=(1.0, 1.0, 1.0))
    assert neutral["pv"] == fresh["pv"]


def test_quad_analytic_anchors():
    # strikes far below the support: the capped spread pays its full width
    # almost surely (exercises the below-grid survival branch); far above: 0
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    deep = price_rainbow_quad([marg] * 3, np.eye(3), df=1.0, k_lo=0.01, k_hi=0.02)
    assert abs(deep["pv"] - 0.01 * 1e6) < 1e-3
    far = price_rainbow_quad([marg] * 3, np.eye(3), df=1.0, k_lo=50.0, k_hi=50.01)
    assert abs(far["pv"]) < 1e-6
    assert deep["n_nodes"] == 48 * 2 * 48


@pytest.mark.skipif(not all(p.exists() for p in CSV3),
                    reason="vol surface data not present")
def test_quad_matches_mc_real():
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    spots = spot_panel(sds)
    margs = [implied_marginal(sd, date_index(sd, np.datetime64(spots.index[-1], "D")), 1.0)
             for sd in sds.values()]
    corr = historical_corr(spots, 5)
    q = price_rainbow_quad(margs, corr, df=0.96, n_outer=64, n_inner=64)
    p = price_rainbow(margs, corr, df=0.96, n_paths=2**18, seed=11, method="sobol")
    assert abs(q["pv"] / p["pv"] - 1.0) < 5e-4


def test_implied_vol_round_trip():
    f, ttm, df = 1.03, 1.0, 0.97
    k = np.array([0.7, 0.9, 1.0, 1.2, 1.5])
    sig = np.array([0.32, 0.22, 0.20, 0.17, 0.15])
    iv = bs_implied_vol(black76(f, k, ttm, sig, df), f, k, ttm, df=df)
    assert np.allclose(iv, sig, atol=1e-9)
    # unattainable prices (at/below intrinsic, at/above forward) give NaN
    assert np.isnan(bs_implied_vol([0.0, df * f], f, [1.0, 1.0], ttm, df=df)).all()


def test_sobol_agrees_with_pseudo():
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    corr = np.full((3, 3), 0.5)
    np.fill_diagonal(corr, 1.0)
    perf = sample_performances([marg] * 3, corr, 2**14, seed=1, method="sobol")
    assert np.allclose(perf.mean(axis=0), 1.0, atol=2e-3)
    ps = price_rainbow([marg] * 3, corr, df=1.0, n_paths=2**16, seed=5, method="sobol")
    pp = price_rainbow([marg] * 3, corr, df=1.0, n_paths=400_000, seed=5)
    assert abs(ps["pv"] - pp["pv"]) < 6 * pp["se"]
    with pytest.raises(ValueError, match="unknown method"):
        sample_performances([marg] * 3, corr, 100, method="halton")


def test_price_rainbow_smoke():
    sd = _flat_sd()
    marg = implied_marginal(sd, 0, 1.0)
    res = price_rainbow([marg] * 3, np.eye(3), df=0.96, n_paths=100_000, seed=4)
    assert 0.0 < res["pv"] < 0.96 * 0.12 * 1e6
    assert 0.0 < res["se"] < res["pv"]
    assert 0.0 <= res["p_capped"] <= res["p_itm"] <= 1.0


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_real_surface_marginal_consistency():
    sd = load_surfaces(CSV)
    i = len(sd) - 1
    assert date_index(sd, sd.dates[i]) == i
    marg = implied_marginal(sd, i, 1.0)
    # the recovered mean must reproduce the model-free forward ratio
    assert abs(marg.mean / marg.fwd_ratio - 1.0) < 1e-3
    assert marg.cdf[0] < 1e-6 and marg.cdf[-1] > 1.0 - 1e-6
    assert abs(np.trapezoid(marg.pdf, marg.x) - 1.0) < 1e-3
    panel = spot_panel({"SPX": sd})
    assert len(panel) == len(sd)
