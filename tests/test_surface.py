import numpy as np

from vol_pca.surface import MONEYNESS, TTM_PILLARS, build_pillar_grid, grid_lookup


def _grid():
    rng = np.random.default_rng(7)
    return 0.15 + 0.1 * rng.random((len(TTM_PILLARS), len(MONEYNESS)))


def test_lookup_reproduces_nodes():
    g = _grid()
    tt, mm = np.meshgrid(TTM_PILLARS, MONEYNESS, indexing="ij")
    v = grid_lookup(g, tt.ravel(), mm.ravel())
    assert np.allclose(v, g.ravel())


def test_weights_sum_to_one_and_reproduce_value():
    g = _grid()
    v, idx, w = grid_lookup(g, [0.3, 0.9, 0.05], [97.0, 132.0, 55.0], want_weights=True)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose((g.ravel()[idx] * w).sum(axis=1), v)


def test_clamping_outside_grid():
    g = _grid()
    lo = grid_lookup(g, 0.001, 40.0)
    hi = grid_lookup(g, 5.0, 200.0)
    assert np.isclose(lo[0], g[0, 0]) and np.isclose(hi[0], g[-1, -1])


def test_build_pillar_grid_flat_extrapolation():
    raw_ttms = np.array([0.2, 0.5, 1.5])
    raw = np.tile(np.array([[0.3], [0.2], [0.25]]), (1, len(MONEYNESS)))
    g = build_pillar_grid(raw_ttms, raw)
    assert np.allclose(g[0], 0.3)          # 0.08 pillar clamps to shortest term
    assert np.allclose(g[TTM_PILLARS == 0.5], 0.2)
