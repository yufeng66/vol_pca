"""PCA factor model of vol surface moves and factor-based vega estimates.

Fit: PCA on daily pillar-vol changes X, transformed per pillar by a weight
vector w: SVD of (X - mu) * w. Supported weightings:

  "cov"   w = 1          plain covariance PCA; high-variance short-TTM pillars
                         dominate the top factors
  "corr"  w = 1/std      correlation PCA; every pillar matters equally
  array   custom         e.g. the book's average absolute bucketed vega, so
                         factor capacity is allocated by dollar P&L impact
                         instead of raw surface variance (far-OTM short-term
                         smile noise the book barely prices off gets ignored)

Book exposure to factor k on a given day is E_k = (bucket_vega / w) . L_k,
the day's factor movement is the score f_k = L_k . ((dsigma - mu) * w), and
the K-factor vega P&L estimate is bucket_vega . mu + sum_{k<=K} E_k f_k.
With all factors this reproduces bucket_vega . dsigma (= pl_vega_lin)
exactly, for any positive weighting.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vol_pca.surface import MONEYNESS, TTM_PILLARS, grid_lookup


@dataclass
class PCAModel:
    mu: np.ndarray            # (n_pillars,) mean daily change
    weights: np.ndarray       # (n_pillars,) per-pillar transform weights
    components: np.ndarray    # (n_comp, n_pillars) rows orthonormal
    evr: np.ndarray           # explained variance ratio (in weighted space)

    @property
    def n_components(self):
        return self.components.shape[0]


def fit_pca(dsigma, fit_mask=None, weights="corr"):
    X = dsigma if fit_mask is None else dsigma[fit_mask]
    mu = X.mean(axis=0)
    if isinstance(weights, str):
        if weights == "cov":
            w = np.ones(X.shape[1])
        elif weights == "corr":
            std = X.std(axis=0)
            w = 1.0 / np.where(std > 0, std, np.inf)
        else:
            raise ValueError(f"unknown weighting {weights!r}")
    else:
        w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-9 * w.max())   # keep the transform invertible
    _, svals, components = np.linalg.svd((X - mu) * w, full_matrices=False)
    return PCAModel(mu=mu, weights=w, components=components,
                    evr=svals**2 / (svals**2).sum())


def sticky_strike_dsigma(sd):
    """Daily pillar vol changes in sticky-STRIKE coordinates.

    The default `dsigma` from simulate_book is sticky-moneyness: the change
    of the surface at a fixed (TTM, % of spot) point. Here each pillar
    instead tracks a fixed STRIKE, re-anchored to the previous close every
    day: pillar (tau_i, m_j) on day t-1 is the strike K = m_j% x S_{t-1},
    whose moneyness on day t is m_j x S_{t-1}/S_t. So

        dsigma_ss[t, (i,j)] = sigma_t(tau_i, m_j S_{t-1}/S_t) - sigma_{t-1}[i, j]

    i.e. surface change plus smile slide, at fixed TTM (term roll excluded).
    Two caveats vs the fixed-moneyness version: the rescaled query can cross
    interpolation brackets, so the exact bucketed-vega identity is lost
    (second-order error), and the outermost columns clamp against the 50/150
    moneyness edge of the quoted grid.
    """
    nt, nm = len(TTM_PILLARS), len(MONEYNESS)
    ttm_q = np.repeat(TTM_PILLARS, nm)
    out = np.empty((len(sd) - 1, nt * nm))
    for t in range(1, len(sd)):
        mon_q = np.tile(MONEYNESS * (sd.spot[t - 1] / sd.spot[t]), nt)
        out[t - 1] = grid_lookup(sd.grids[t], ttm_q, mon_q) - sd.grids[t - 1].ravel()
    return out


def factor_scores(dsigma, model):
    return ((dsigma - model.mu) * model.weights) @ model.components.T


def factor_exposures(bucket_vega, model):
    return (bucket_vega / model.weights) @ model.components.T


def factor_vega_estimates(bucket_vega, dsigma, model, ks):
    """Vega P&L estimates per day for each factor count in `ks`."""
    scores = factor_scores(dsigma, model)
    exposures = factor_exposures(bucket_vega, model)
    base = bucket_vega @ model.mu
    out = {}
    for k in ks:
        out[k] = base + (exposures[:, :k] * scores[:, :k]).sum(axis=1)
    return out, exposures, scores


def estimate_metrics(target, estimates, mask=None):
    """R^2, correlation and RMSE of each estimate vs the target P&L series."""
    rows = []
    t = target if mask is None else target[mask]
    for name, est in estimates.items():
        e = est if mask is None else est[mask]
        err = t - e
        rows.append({
            "estimate": name,
            "r2": 1 - err.var() / t.var(),
            "corr": np.corrcoef(t, e)[0, 1],
            "rmse": np.sqrt((err**2).mean()),
        })
    return pd.DataFrame(rows).set_index("estimate")
