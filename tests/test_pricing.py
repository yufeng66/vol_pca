import numpy as np

from vol_pca.pricing import black76


def test_atm_call_known_value():
    # Black-76: F=K=100, T=1, sigma=0.2, DF=1 -> 7.9656
    p = black76(100.0, 100.0, 1.0, 0.2, 1.0)
    assert abs(p[0] - 7.9656) < 1e-3


def test_deep_itm_approaches_discounted_forward_minus_strike():
    p = black76(100.0, 1e-6, 1.0, 0.2, 0.97)
    assert abs(p[0] - 0.97 * (100.0 - 1e-6)) < 1e-6


def test_expired_is_intrinsic():
    p = black76(np.array([120.0, 80.0]), 100.0, 0.0, 0.2, 1.0)
    assert np.allclose(p, [20.0, 0.0])


def test_greeks_signs_and_monotonicity():
    p1, delta, gamma, vega, _ = black76(100.0, 100.0, 0.5, 0.2, 0.99, spot=98.0, greeks=True)
    p2 = black76(100.0, 110.0, 0.5, 0.2, 0.99)
    assert p2[0] < p1[0]
    assert 0 < delta[0] < 1.1 and gamma[0] > 0 and vega[0] > 0


def test_vega_matches_finite_difference():
    eps = 1e-5
    _, _, _, vega, vanna = black76(100.0, 105.0, 0.7, 0.22, 0.98, spot=100.0, greeks=True)
    up = black76(100.0, 105.0, 0.7, 0.22 + eps, 0.98)
    dn = black76(100.0, 105.0, 0.7, 0.22 - eps, 0.98)
    assert abs(vega[0] - (up[0] - dn[0]) / (2 * eps)) < 1e-4
    # vanna vs cross finite difference
    h = 1e-3
    vu = black76(100.0 * (1 + h), 105.0, 0.7, 0.22 + eps, 0.98)
    vd = black76(100.0 * (1 + h), 105.0, 0.7, 0.22 - eps, 0.98)
    vega_up = (vu[0] - vd[0]) / (2 * eps)
    assert abs(vanna[0] - (vega_up - vega[0]) / (100.0 * h)) < 1e-2
