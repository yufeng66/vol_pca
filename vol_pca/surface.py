"""Pillar vol grid: fixed (TTM x moneyness) surface per date.

All pricing looks vols up on this grid with bilinear interpolation (linear in
vol across TTM and across moneyness, flat extrapolation outside). Linear
interpolation means every queried vol is an exact linear combination of pillar
vols, so bucketed vegas reproduce the full linear vol P&L exactly.
"""

import numpy as np

# Moneyness columns as quoted in the data: strike as % of spot.
MONEYNESS = np.array([50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140, 150], dtype=float)

# TTM pillars (years). The call-spread book only holds options with TTM in
# (0, 1]; queries outside [min, max] pillar are clamped (flat extrapolation).
TTM_PILLARS = np.array([0.08, 0.17, 0.25, 1 / 3, 0.5, 2 / 3, 0.83, 1.0])

N_PILLARS = len(TTM_PILLARS) * len(MONEYNESS)


def build_pillar_grid(raw_ttms, raw_vols, ttm_pillars=TTM_PILLARS):
    """Interpolate raw term rows (n_terms, n_mon) onto fixed TTM pillars."""
    grid = np.empty((len(ttm_pillars), raw_vols.shape[1]))
    for j in range(raw_vols.shape[1]):
        grid[:, j] = np.interp(ttm_pillars, raw_ttms, raw_vols[:, j])
    return grid


def _axis_weights(pillars, q):
    q = np.clip(q, pillars[0], pillars[-1])
    i0 = np.clip(np.searchsorted(pillars, q, side="right") - 1, 0, len(pillars) - 2)
    w1 = (q - pillars[i0]) / (pillars[i0 + 1] - pillars[i0])
    return i0, w1


def grid_lookup(grid, ttm_q, mon_q, want_weights=False,
                ttm_pillars=TTM_PILLARS, moneyness=MONEYNESS):
    """Bilinear vol lookup on a pillar grid.

    Returns vols, and optionally the sparse interpolation weights as
    (flat_idx, w) arrays of shape (n_query, 4) with flat_idx into
    grid.ravel(). Weights sum to 1 per query row.
    """
    ttm_q = np.atleast_1d(np.asarray(ttm_q, dtype=float))
    mon_q = np.atleast_1d(np.asarray(mon_q, dtype=float))
    it, wt = _axis_weights(ttm_pillars, ttm_q)
    im, wm = _axis_weights(moneyness, mon_q)

    nm = len(moneyness)
    idx = np.stack([it * nm + im, it * nm + im + 1,
                    (it + 1) * nm + im, (it + 1) * nm + im + 1], axis=1)
    w = np.stack([(1 - wt) * (1 - wm), (1 - wt) * wm,
                  wt * (1 - wm), wt * wm], axis=1)
    vols = (grid.ravel()[idx] * w).sum(axis=1)
    if want_weights:
        return vols, idx, w
    return vols
