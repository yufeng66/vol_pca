import pathlib

import numpy as np
import pytest

from tests.test_book import _flat_sd
from vol_pca.data import SurfaceData, load_surfaces
from vol_pca.factors import (bumped_factor_exposures, factor_exposures,
                             factor_scores, fit_pca)
from vol_pca.surface import MONEYNESS, N_PILLARS, TTM_PILLARS
from vol_pca.var import (book_price_fn, book_speed, build_book,
                         build_scenarios, bump_greeks, bump_pnl,
                         full_reval_pnl, greeks_pnl, hist_var,
                         rolling_var_adaptive, rolling_var_backtest,
                         rolling_var_bump, scenario_dsigma,
                         strike_histogram_weights)

CSV = pathlib.Path(__file__).resolve().parents[1] / "SPX_volSurface 2.csv"


def test_static_world_scenarios_are_zero_pnl():
    # flat surface, flat curves, constant spot: every historical "move" is a
    # no-op, so full reval and the scenario pillar changes are exactly zero
    sd = _flat_sd(n_days=6)
    book = build_book(sd)
    scen = build_scenarios(sd, book)
    assert book.n_legs == 12                      # 6 days x 2 legs, none expired
    assert np.allclose(full_reval_pnl(book, scen), 0.0, atol=1e-9)
    assert np.allclose(scenario_dsigma(book, scen), 0.0, atol=1e-12)


def test_parallel_vol_bump_scenario():
    # one scenario carries a +1pt parallel surface bump: full reval matches
    # dollar vega x 0.01 up to spread-vega convexity, and the scenario pillar
    # move is exactly +0.01 everywhere
    sd = _flat_sd(n_days=6, bump_day=3, bump=0.01)
    book = build_book(sd)
    scen = build_scenarios(sd, book)
    ds = scenario_dsigma(book, scen)
    j = 2                                          # move day2 -> day3
    assert np.allclose(ds[j], 0.01)
    assert np.allclose(np.delete(ds, j, axis=0), 0.0, atol=1e-12)
    pnl = full_reval_pnl(book, scen)
    lin = (book.units * book.vega0).sum() * 0.01
    assert abs(pnl[j] / lin - 1) < 0.15
    assert np.allclose(np.delete(pnl, j), 0.0, atol=1e-9)


def test_greeks_all_factor_vol_term_is_exact():
    # scored on the fitting rows, the all-factor vol term reproduces
    # bucket_vega . dsigma_scen exactly, for any positive weighting
    sd = _flat_sd(n_days=6, bump_day=3, bump=0.005)
    book = build_book(sd)
    scen = build_scenarios(sd, book)
    ds = scenario_dsigma(book, scen)
    rng = np.random.default_rng(0)
    ds = ds + 1e-3 * rng.standard_normal(ds.shape)  # make the fit non-degenerate
    for w in ("cov", rng.uniform(0.5, 2.0, ds.shape[1])):
        model = fit_pca(ds, weights=w)
        est, _ = greeks_pnl(book, scen, model, [model.n_components],
                            dsigma_scen=ds)
        assert np.allclose(est[model.n_components],
                           book.bucket_vega @ ds.T, atol=1e-8)


def test_pure_spot_scenario_hedged_pnl_is_gamma():
    # flat surface, moving spot: the delta-hedged full reval is the gamma term
    # up to higher order, and the greeks projection agrees (vol terms vanish)
    sd = _flat_sd(n_days=6)
    sd.spot[:] = [100.0, 101.0, 99.9, 100.4, 100.9, 100.0]
    book = build_book(sd)
    scen = build_scenarios(sd, book)
    assert np.allclose(scenario_dsigma(book, scen), 0.0, atol=1e-12)
    pnl = full_reval_pnl(book, scen)
    dS = book.spot * (scen.ratio - 1.0)
    hedged = pnl - book.net_delta * dS
    gamma = 0.5 * (book.units * book.gamma0).sum() * dS**2
    assert np.allclose(hedged, gamma, rtol=0.2)
    model = fit_pca(scenario_dsigma(book, scen), weights="cov")
    est, parts = greeks_pnl(book, scen, model, [3])
    assert np.allclose(parts["vol"][3], 0.0, atol=1e-6)
    assert np.allclose(est[3], book.net_delta * dS + gamma, atol=1e-6)


def test_hist_var_orders_levels():
    pnl = np.linspace(-100.0, 100.0, 201)
    assert hist_var(pnl, 0.99) > hist_var(pnl, 0.95) > 0
    assert np.isclose(hist_var(pnl, 0.99), 98.0)


def test_book_speed_matches_stencil_delta_gamma_scale():
    # sanity of the same stencil at lower orders: on a real curved book the
    # finite-difference speed is finite and the analytic delta/gamma are
    # reproduced by central differences at matching accuracy
    sd = _flat_sd(n_days=6)
    book = build_book(sd)
    h = 0.01 * book.spot
    from vol_pca.attribution import _bs
    v = {m: float((book.units * _bs(book.spot + m * h, book.strike, book.ttm,
                                    book.sig0, book.r0, book.q0)).sum())
         for m in (-1, 0, 1)}
    assert np.isclose((v[1] - v[-1]) / (2 * h), book.net_delta, rtol=1e-3)
    assert np.isclose((v[1] - 2 * v[0] + v[-1]) / h**2,
                      (book.units * book.gamma0).sum(), rtol=1e-2)
    assert np.isfinite(book_speed(book))


def test_strike_histogram_weights():
    # flat book: 6 daily spreads at 100/110% of a constant spot, all ~1Y TTM
    sd = _flat_sd(n_days=6)
    book = build_book(sd)
    w = strike_histogram_weights(book, floor_frac=0.05)
    assert w.shape == (N_PILLARS,)
    assert np.isclose(w.max(), 1.0) and w.min() >= 0.05
    g = w.reshape(len(TTM_PILLARS), len(MONEYNESS))
    # mass sits on the 12M row at the 100/110 strike columns...
    i100, i110 = list(MONEYNESS).index(100), list(MONEYNESS).index(110)
    assert g[-1, i100] > 0.5 and g[-1, i110] > 0.5
    # ...and the short-TTM rows and far wings are pure floor
    assert np.allclose(g[:5], 0.05)
    assert np.allclose(g[-1, [0, -1]], 0.05)


def _noisy_model_and_book(spot_moves=False):
    sd = _flat_sd(n_days=6, bump_day=3, bump=0.005)
    if spot_moves:
        sd.spot[:] = [100.0, 101.0, 99.9, 100.4, 100.9, 100.0]
    book = build_book(sd)
    ds = scenario_dsigma(book, build_scenarios(sd, book))
    rng = np.random.default_rng(1)
    ds = ds + 1e-3 * rng.standard_normal(ds.shape)   # non-degenerate fit
    model = fit_pca(ds, weights=rng.uniform(0.5, 2.0, ds.shape[1]))
    return sd, book, model, ds


def test_bumped_exposures_match_analytic():
    # small-eps central differences along L_j/w reproduce the analytic
    # (bucket_vega / w) . L_j exposures — the bump route generalizes the
    # vega-scatter route, it does not change it
    _, book, model, ds = _noisy_model_and_book()
    pf = book_price_fn(book)
    assert np.isclose(pf(np.zeros(ds.shape[1])),
                      float((book.units * book.v0).sum()))
    s1 = factor_scores(ds, model)[:, 0].std()
    expos, curv = bumped_factor_exposures(pf, model, k=4, eps=0.01 * s1)
    analytic = factor_exposures(book.bucket_vega, model)[:4]
    assert np.allclose(expos, analytic, rtol=1e-3,
                       atol=1e-4 * np.abs(analytic).max())
    assert curv is not None and np.isfinite(curv)


def test_pc1_squared_improves_large_move_estimate():
    # calibrate slope + curvature from the +/-eps pair, then predict the book
    # value at 1.5 eps along PC1 (outside the calibration points): the
    # 0.5 * curv * f^2 term must move the estimate toward the truth
    _, book, model, ds = _noisy_model_and_book()
    pf = book_price_fn(book)
    eps = 2.0 * factor_scores(ds, model)[:, 0].std()
    expos, curv = bumped_factor_exposures(pf, model, k=1, eps=eps)
    x = 1.5 * eps
    truth = pf(x * model.components[0] / model.weights) - pf(
        np.zeros(model.components.shape[1]))
    lin = expos[0] * x
    quad = lin + 0.5 * curv * x**2
    assert abs(quad - truth) < abs(lin - truth)


def test_bump_greeks_match_analytic():
    # every sensitivity in the formula-free pass reproduces its closed-form
    # counterpart (the formulas' only remaining job: validating the bumps)
    _, book, model, ds = _noisy_model_and_book()
    bg = bump_greeks(book, model, k=3, k_cross=3,
                     eps=1e-3 * model.score_std[:3])
    assert bg.n_revals == 10 + 6 + 12
    assert np.isclose(bg.v0, float((book.units * book.v0).sum()))
    assert np.isclose(bg.delta, book.net_delta, rtol=1e-3)
    assert np.isclose(bg.gamma, float((book.units * book.gamma0).sum()),
                      rtol=1e-2)
    assert np.isclose(bg.speed, book_speed(book), rtol=1e-6)
    assert np.isclose(bg.rho, float(book.rho0.sum()), rtol=1e-2)
    assert np.isclose(bg.dvq, float(book.dvq0.sum()), rtol=1e-2)
    ana_e = factor_exposures(book.bucket_vega, model)[:3]
    assert np.allclose(bg.expos, ana_e, rtol=1e-3,
                       atol=1e-4 * np.abs(ana_e).max())
    ana_x = (model.components[:3] / model.weights) @ book.bucket_vanna
    assert np.allclose(bg.cross, ana_x, rtol=2e-2,
                       atol=1e-3 * np.abs(ana_x).max())


def test_bump_pnl_matches_formula_projection():
    # with tiny bumps, no quad term and crosses for every factor, the bump
    # projection agrees with the analytic greeks_pnl estimate (mu zeroed so
    # both vanna conventions coincide; rate/div are zero in the flat world)
    sd, book, model, ds = _noisy_model_and_book(spot_moves=True)
    model.mu[:] = 0.0
    scen = build_scenarios(sd, book)
    est_a = greeks_pnl(book, scen, model, ks=[3], dsigma_scen=ds)[0][3]
    bg = bump_greeks(book, model, k=3, k_cross=3,
                     eps=1e-3 * model.score_std[:3])
    est_b = bump_pnl(book, bg, scen, model, dsigma_scen=ds,
                     quad=None, cube=False)
    assert np.allclose(est_b, est_a,
                       atol=max(1e-6, 5e-3 * np.abs(est_a).max()))
    with_cube = bump_pnl(book, bg, scen, model, dsigma_scen=ds, quad=None)
    dS = book.spot * (scen.ratio - 1.0)
    assert np.allclose(with_cube - est_b, bg.speed / 6.0 * dS**3)


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_rolling_adaptive_smoke():
    sd = load_surfaces(CSV)
    m = 90
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    book = build_book(small)
    ds = scenario_dsigma(book, build_scenarios(small, book))
    adapt = rolling_var_adaptive(small, strike_histogram_weights, ds,
                                 ks=(3,), start=85, n_jobs=2)
    assert len(adapt) == 5
    for col in ("s3_h", "s3c_h", "s3_raw", "s3c_raw"):
        assert (adapt[col] > 0).all()


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_rolling_bump_smoke():
    sd = load_surfaces(CSV)
    m = 90
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    book = build_book(small)
    ds = scenario_dsigma(book, build_scenarios(small, book))
    model = fit_pca(ds, weights=np.abs(book.bucket_vega))
    roll = rolling_var_bump(small, model, start=85, n_jobs=2, k=3, k_cross=1)
    assert len(roll) == 5
    assert (roll.n_revals == 10 + 6 + 4).all()
    for col in ("b3c_h", "b3_h", "b3c_raw", "b3_raw"):
        assert (roll[col] > 0).all()


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_rolling_backtest_smoke():
    sd = load_surfaces(CSV)
    m = 90
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    ds = scenario_dsigma(build_book(small), build_scenarios(small, build_book(small)))
    model = fit_pca(ds, weights="cov")
    roll = rolling_var_backtest(small, model, ks=(3,), start=85, n_jobs=2)
    assert len(roll) == 5
    for col in ("full_h", "g3_h", "g3c_h", "g3_raw", "g3c_raw"):
        assert (roll[col] > 0).all()


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_real_data_greeks_track_full_reval():
    sd = load_surfaces(CSV)
    m = 120
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    book = build_book(small)
    scen = build_scenarios(small, book)
    pnl = full_reval_pnl(book, scen)
    ds = scenario_dsigma(book, scen)
    model = fit_pca(ds, weights=np.abs(book.bucket_vega))
    est, _ = greeks_pnl(book, scen, model, [model.n_components], dsigma_scen=ds)
    assert np.corrcoef(pnl, est[model.n_components])[0, 1] > 0.99
