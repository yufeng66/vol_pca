import pathlib

import numpy as np
import pytest

from vol_pca.book import simulate_book
from vol_pca.data import SurfaceData, load_surfaces
from vol_pca.factors import factor_vega_estimates, fit_pca
from vol_pca.surface import MONEYNESS, TTM_PILLARS

CSV = pathlib.Path(__file__).resolve().parents[1] / "SPX_volSurface 2.csv"


def _flat_sd(n_days=4, vol=0.20, spot=100.0, bump_day=None, bump=0.01):
    shape = (len(TTM_PILLARS), len(MONEYNESS))
    dates = np.datetime64("2024-01-01") + np.arange(n_days)
    grids = np.full((n_days,) + shape, vol)
    if bump_day is not None:
        grids[bump_day:] += bump
    return SurfaceData(
        dates=dates.astype("datetime64[D]"),
        spot=np.full(n_days, spot),
        grids=grids,
        curve_ttms=[np.array([0.0, 2.0])] * n_days,
        curve_lndf=[np.zeros(2)] * n_days,
        curve_lnfr=[np.zeros(2)] * n_days,
    )


def test_static_world_pl_is_theta_only():
    res, _, _ = simulate_book(_flat_sd())
    assert np.allclose(res["pl_delta"], 0) and np.allclose(res["pl_gamma"], 0)
    assert np.allclose(res["pl_vega_full"], 0)
    assert np.allclose(res["pl"], res["pl_theta"] + res["pl_resid"])
    assert (res["pl_resid"].abs() < 1e-6 * res["pl_theta"].abs().max()).all()


def test_vol_bump_captured_by_vega():
    res, bucket_vega, dsigma = simulate_book(_flat_sd(bump_day=2, bump=0.01))
    day = res.iloc[1]  # move from day1 -> day2 carries the bump
    assert day["pl_vega_full"] != 0
    # parallel bump: linear estimate tracks full reval up to spread-vega
    # convexity (net vega is a small difference of two large leg vegas, so
    # relative volga error is material even for a 1-pt bump)
    assert abs(day["pl_vega_lin"] / day["pl_vega_full"] - 1) < 0.15
    assert np.isclose((bucket_vega[1] * dsigma[1]).sum(), day["pl_vega_lin"], rtol=1e-10)


def test_all_factor_estimate_reproduces_linear_vega():
    res, bucket_vega, dsigma = simulate_book(_flat_sd(n_days=6, bump_day=3, bump=0.005))
    for standardize in (False, True):
        model = fit_pca(dsigma, standardize=standardize)
        est, _, _ = factor_vega_estimates(bucket_vega, dsigma, model, [model.n_components])
        assert np.allclose(est[model.n_components], res["pl_vega_lin"].to_numpy(), atol=1e-8)


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_real_data_bucket_vega_consistency():
    sd = load_surfaces(CSV)
    m = 60
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    res, bucket_vega, dsigma = simulate_book(small)
    lin = (bucket_vega * dsigma).sum(axis=1)
    assert np.abs(res["pl_vega_lin"].to_numpy() - lin).max() < 1e-6
    assert res["n_spreads"].iloc[-1] == m - 1  # one new spread per day, none expired yet
