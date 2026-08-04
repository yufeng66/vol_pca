import numpy as np

from tests.test_book import _flat_sd
from vol_pca.book import simulate_book
from vol_pca.key_rate import (KNOTS, bumped_key_rate_sens, daily_key_rate_sens,
                              key_rate_pnl, key_rate_scores, key_rate_sens,
                              score_matrix, tent_weights, tilt)
from vol_pca.surface import MONEYNESS, N_PILLARS, TTM_PILLARS
from vol_pca.var import build_book, build_scenarios


def test_tent_weights_partition_and_linear_reproduction():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.05, 1.2, 50)
    w = tent_weights(t)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all()
    # exact at the knots, clamped outside
    assert np.allclose(tent_weights(KNOTS), np.eye(len(KNOTS)))
    assert np.allclose(tent_weights([0.05, 2.0]),
                       [[1, 0, 0, 0], [0, 0, 0, 1]])
    # inside the knot range the weights reproduce the TTM itself
    inside = np.clip(t, KNOTS[0], KNOTS[-1])
    assert np.allclose(tent_weights(inside) @ KNOTS, inside)


def test_score_matrix_reads_are_exact_at_pillar_knots():
    R = score_matrix()
    assert R.shape == (8, N_PILLARS)
    nm = len(MONEYNESS)
    atm = list(MONEYNESS).index(100)
    # 3M/6M/12M knots sit on TTM pillars and 100 is a quoted column, so the
    # level reads there are one-hot; only the 9M knot interpolates
    for row, pillar in ((0, 2), (1, 4), (3, 7)):
        expect = np.zeros(N_PILLARS)
        expect[pillar * nm + atm] = 1.0
        assert np.allclose(R[row], expect, atol=1e-12)
    assert (np.abs(R[2]) > 1e-12).sum() > 1
    # every skew row is a difference of two reads: weights sum to zero
    assert np.allclose(R[4:].sum(axis=1), 0.0, atol=1e-12)


def test_scores_parallel_tilt_and_linear_ttm_fields():
    nt, nm = len(TTM_PILLARS), len(MONEYNESS)
    # parallel field: all levels equal, no skew
    s = key_rate_scores(np.full(N_PILLARS, 0.02))
    assert np.allclose(s[0, :4], 0.02) and np.allclose(s[0, 4:], 0.0)
    # pure tilt field (constant in TTM): unit skew everywhere, no level
    s = key_rate_scores(np.tile(0.003 * tilt(MONEYNESS), nt))
    assert np.allclose(s[0, :4], 0.0, atol=1e-15)
    assert np.allclose(s[0, 4:], 0.003)
    # linear-in-TTM level field: the cubic read reproduces the line exactly,
    # including at the interpolated 9M knot
    a, b = 0.004, -0.006
    s = key_rate_scores(np.repeat(a + b * TTM_PILLARS, nm))
    assert np.allclose(s[0, :4], a + b * KNOTS, atol=1e-12)
    assert np.allclose(s[0, 4:], 0.0, atol=1e-12)


def test_estimate_reproduces_linear_pnl_for_in_span_field():
    # for any book and any surface move inside the level+skew span (linear in
    # TTM on [3M, 1Y], tilt-linear in moneyness), sens . scores equals the
    # exact per-leg linear P&L sum(d * dsigma(leg)) -- the framework's
    # internal consistency identity
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 1e5, 40)
    mon = rng.uniform(70.0, 130.0, 40)
    ttm = rng.uniform(KNOTS[0], KNOTS[-1], 40)
    a, b, c = 0.002, -0.004, 0.0015
    truth = float((d * (a + b * ttm + c * tilt(mon))).sum())
    field = a + b * TTM_PILLARS[:, None] + c * tilt(MONEYNESS)[None, :]
    est = float(key_rate_sens(d, mon, ttm) @ key_rate_scores(field.ravel())[0])
    assert np.isclose(est, truth, rtol=1e-10)


def test_daily_sens_match_simulate_book_net_vega():
    sd = _flat_sd(n_days=6)
    _, bucket_vega, _ = simulate_book(sd)
    dates, sens = daily_key_rate_sens(sd)
    assert (dates == sd.dates[1:]).all() and sens.shape == (5, 8)
    # same legs, same vegas: tent-grouped levels sum to the scattered net
    assert np.allclose(sens[:, :4].sum(axis=1), bucket_vega.sum(axis=1),
                       rtol=1e-10)
    # all legs are ~1Y here, so nothing lands on the 3M/6M knots
    assert np.allclose(sens[:, [0, 1, 4, 5]], 0.0)


def test_bumped_sens_match_analytic():
    book = build_book(_flat_sd(n_days=6))
    mon0 = book.strike / book.spot * 100.0
    analytic = key_rate_sens(book.units * book.vega0, mon0, book.ttm)
    bumped = bumped_key_rate_sens(book, eps=1e-4)
    assert np.allclose(bumped, analytic, rtol=1e-3,
                       atol=1e-6 * np.abs(analytic).max())


def test_key_rate_pnl_parallel_scenario_and_parts_sum():
    sd = _flat_sd(n_days=6, bump_day=3, bump=0.01)
    book = build_book(sd)
    scen = build_scenarios(sd, book)
    est, parts = key_rate_pnl(book, scen)
    total = sum(parts[p] for p in ("delta", "gamma", "vol", "vanna",
                                   "rate", "div"))
    assert np.allclose(est, total)
    # the one live scenario is a +1pt parallel move at constant spot: the
    # estimate is exactly net vega x 0.01, everything else zero
    j = 2
    lin = float((book.units * book.vega0).sum()) * 0.01
    assert np.isclose(est[j], lin, rtol=1e-9)
    assert np.allclose(np.delete(est, j), 0.0, atol=1e-9)
    assert np.allclose(parts["sens_vega"][:4].sum(),
                       float((book.units * book.vega0).sum()))
