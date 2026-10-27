"""Correlation-profile copula: exact marginals, Gaussian limit, loadings."""
import numpy as np
import pytest
from scipy.stats import kstest, norm, qmc

from vol_pca.rainbow import (crash_lift_profile, one_factor_loadings,
                             profile_cdf_table, profile_tables,
                             profile_uniforms, sigmoid_profile, step_profile,
                             tanh_profile)

C = np.array([[1.0, 0.751, 0.429], [0.751, 1.0, 0.478], [0.429, 0.478, 1.0]])


def test_one_factor_loadings_reproduce_pairs():
    a = one_factor_loadings(C)
    assert np.allclose(np.outer(a, a)[np.triu_indices(3, 1)], C[np.triu_indices(3, 1)])
    with pytest.raises(ValueError):
        one_factor_loadings(np.array([[1, 0.95, 0.3], [0.95, 1, 0.9], [0.3, 0.9, 1]]))


def test_constant_profile_is_standard_normal_cdf():
    a = one_factor_loadings(C)
    z, f = profile_cdf_table(step_profile(a, a), 1, (0.0,))
    assert np.abs(f - norm.cdf(z)).max() < 1e-9


def test_constant_profile_recovers_gaussian_copula():
    a = one_factor_loadings(C)
    pts = qmc.Sobol(d=4, scramble=True, seed=3).random_base2(16)
    u = profile_uniforms(pts, step_profile(a, a), (0.0,))
    z = norm.ppf(u)
    assert np.abs(np.corrcoef(z.T) - C).max() < 0.01


@pytest.mark.parametrize("prof,breaks", [
    (step_profile(one_factor_loadings(C) * 1.08, one_factor_loadings(C) * 0.85), (0.0,)),
    (tanh_profile(one_factor_loadings(C), 0.4), ()),
    (sigmoid_profile(one_factor_loadings(C), 0.2, 0.15, -0.8, 0.8, 0.4), (-0.8, 0.8)),
])
def test_marginals_stay_uniform_and_dependence_is_asymmetric(prof, breaks):
    pts = qmc.Sobol(d=4, scramble=True, seed=5).random_base2(16)
    u = profile_uniforms(pts, prof, breaks)
    for j in range(3):
        assert kstest(u[:, j], "uniform").statistic < 0.01
    z = norm.ppf(u)
    w = z.mean(1)
    dn = np.corrcoef(z[w < 0, 0], z[w < 0, 1])[0, 1]
    up = np.corrcoef(z[w >= 0, 0], z[w >= 0, 1])[0, 1]
    assert dn > up + 0.05


def test_sigmoid_profile_per_index_strengths():
    a = one_factor_loadings(C)
    w = np.linspace(-3, 3, 7)
    prof = sigmoid_profile(a, [0.3, 0.0, 0.5], [0.0, 0.0, 0.2], -1.0, 0.5, 0.2)
    for j, (ld, lu) in enumerate([(0.3, 0.0), (0.0, 0.0), (0.5, 0.2)]):
        g = (1 + ld / (1 + np.exp(-(-1.0 - w) / 0.2)) - lu / (1 + np.exp(-(w - 0.5) / 0.2)))
        assert np.allclose(prof(w, j), np.clip(a[j] * g, 0, 0.999))
    flat = sigmoid_profile(a, 0.0, 0.0)
    assert all(np.allclose(flat(w, j), a[j]) for j in range(3))


def test_profile_tables_reuse_is_exact():
    a = one_factor_loadings(C)
    prof = sigmoid_profile(a, 0.3, 0.0, -1.0, 99.0, 0.3)
    pts = qmc.Sobol(d=4, scramble=True, seed=5).random_base2(12)
    u0 = profile_uniforms(pts, prof, (-1.0,))
    u1 = profile_uniforms(pts, prof, (-1.0,), tables=profile_tables(prof, 3, (-1.0,)))
    assert np.array_equal(u0, u1)


def test_crash_lift_profile_rematches_level_and_nests_gaussian():
    a_g = one_factor_loadings(C)
    prof0, brk, a0 = crash_lift_profile(C, 0.0, n_points=2 ** 14)
    assert np.allclose(a0, a_g, atol=1e-3)           # lam = 0: the Gaussian loadings (to the fixed point's sampling precision)
    prof, brk, a, z = crash_lift_profile(C, 0.25, n_points=2 ** 15, return_scores=True)
    assert (a < a_g).all()                           # the lift is paid for by the plateau
    pc = np.corrcoef(z.T)[[0, 0, 1], [1, 2, 2]]
    assert np.abs(pc - C[[0, 0, 1], [1, 2, 2]]).max() < 0.01   # unconditional level held
    w = np.array([-3.0, -1.0, 1.0, 2.0])
    r = prof(w, 0) * prof(w, 1)                      # SPX-SX5E correlation by state
    assert r[0] > r[1] > r[2] and abs(r[2] - r[3]) < 1e-3   # lifted crash, flat plateau
