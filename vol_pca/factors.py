"""PCA factor model of vol surface moves and factor-based vega estimates.

Fit: PCA on daily pillar-vol changes, by default on the correlation matrix
(per-pillar standardized). Standardizing stops the high-variance short-TTM
pillars from dominating the top factors; empirically it roughly halves the
low-order truncation error on this book's vega P&L versus covariance PCA.

Book exposure to factor k on a given day is the bucketed pillar vega dotted
with the factor loading mapped back to vol units: E_k = (bucket_vega * scale)
. L_k. The day's factor movement is the score f_k = L_k . ((dsigma - mu) /
scale), and the K-factor vega P&L estimate is bucket_vega . mu + sum_{k<=K}
E_k f_k. With all factors this reproduces bucket_vega . dsigma
(= pl_vega_lin) exactly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAModel:
    mu: np.ndarray            # (n_pillars,) mean daily change
    scale: np.ndarray         # (n_pillars,) 1.0s if not standardized
    components: np.ndarray    # (n_comp, n_pillars) rows orthonormal
    evr: np.ndarray           # explained variance ratio per component

    @property
    def n_components(self):
        return self.components.shape[0]


def fit_pca(dsigma, fit_mask=None, standardize=True):
    X = dsigma if fit_mask is None else dsigma[fit_mask]
    mu = X.mean(axis=0)
    scale = X.std(axis=0) if standardize else np.ones(X.shape[1])
    _, svals, components = np.linalg.svd((X - mu) / scale, full_matrices=False)
    return PCAModel(mu=mu, scale=scale, components=components,
                    evr=svals**2 / (svals**2).sum())


def factor_scores(dsigma, model):
    return ((dsigma - model.mu) / model.scale) @ model.components.T


def factor_exposures(bucket_vega, model):
    return (bucket_vega * model.scale) @ model.components.T


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
