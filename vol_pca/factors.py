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
from scipy.interpolate import PchipInterpolator

from vol_pca.surface import MONEYNESS, TTM_PILLARS, grid_lookup


@dataclass
class PCAModel:
    mu: np.ndarray            # (n_pillars,) mean daily change
    weights: np.ndarray       # (n_pillars,) per-pillar transform weights
    components: np.ndarray    # (n_comp, n_pillars) rows orthonormal
    evr: np.ndarray           # explained variance ratio (in weighted space)
    score_std: np.ndarray = None   # (n_comp,) fit-sample std of each score —
                                   # the natural "1 sigma" bump size per factor

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
                    evr=svals**2 / (svals**2).sum(),
                    score_std=svals / np.sqrt(X.shape[0]))


def sticky_strike_dsigma(sd, interp="cubic", include_roll=False):
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

    include_roll=True folds the term-structure roll in as well: the pillar
    then tracks a fixed OPTION (fixed strike AND fixed expiry), so day t
    samples at the decayed TTM too:

        dsigma[t, (i,j)] = sigma_t(tau_i - dt_t, m_j S_{t-1}/S_t) - sigma_{t-1}[i, j]

    with dt_t the calendar-day gap / 365 — surface move + smile slide + term
    roll, i.e. the option's own implied-vol change, so bucket_vega . dsigma
    estimates the attribution's full `vol` component and the roll drift moves
    into the factor mean. The shortest TTM pillar clamps at the 0.08 grid
    edge, losing a sliver of roll there.

    interp selects how day t's surface is sampled at the rescaled point:
    "cubic" (the grid_lookup interpolant pricing itself uses), "linear"
    (bilinear), or "pchip" (shape-preserving cubic along the moneyness axis:
    C1-smooth but monotone between quotes, so it cannot overshoot in the
    steep put wing the way an unconstrained spline can; moneyness-only, so
    not available with include_roll).
    """
    if include_roll and interp == "pchip":
        raise ValueError("include_roll needs 2-D interpolation (linear/cubic)")
    nt, nm = len(TTM_PILLARS), len(MONEYNESS)
    out = np.empty((len(sd) - 1, nt * nm))
    for t in range(1, len(sd)):
        mon_q = np.clip(MONEYNESS * (sd.spot[t - 1] / sd.spot[t]),
                        MONEYNESS[0], MONEYNESS[-1])
        ttm = TTM_PILLARS
        if include_roll:
            dt = (sd.dates[t] - sd.dates[t - 1]).astype(float) / 365.0
            ttm = TTM_PILLARS - dt          # grid_lookup clamps at the edge
        if interp in ("linear", "cubic"):
            new = grid_lookup(sd.grids[t], np.repeat(ttm, nm),
                              np.tile(mon_q, nt), interp=interp)
        elif interp == "pchip":
            new = np.stack([PchipInterpolator(MONEYNESS, row)(mon_q)
                            for row in sd.grids[t]]).ravel()
        else:
            raise ValueError(f"unknown interp {interp!r}")
        out[t - 1] = new - sd.grids[t - 1].ravel()
    return out


def factor_scores(dsigma, model):
    return ((dsigma - model.mu) * model.weights) @ model.components.T


def factor_exposures(bucket_vega, model):
    return (bucket_vega / model.weights) @ model.components.T


def bumped_factor_exposures(price_fn, model, k, eps, curvature=True):
    """Factor exposures by bump-and-revalue, for books whose pricing model
    has no cheap per-pillar vega (simulation-priced payoffs): the central
    difference of the book value along each factor's unit-score surface
    move L_j / w — the raw-vol pillar shift that scores as f_j = 1 and
    f_other = 0 — so in the small-eps limit E_j matches
    factor_exposures(bucket_vega) exactly for any pricer that is smooth in
    the pillar vols.

    price_fn(dsigma) -> book value under the (n_pillars,) pillar vol shift;
    it is the only thing that touches the pricing model, so a simulated
    book plugs in unchanged (2k valuations plus one shared base call).
    eps sizes the bumps in score units (scalar or per-factor): a finite eps
    makes E_j a secant slope over +/-eps, so size it to the moves you care
    about (the fit-sample score std, or a tail quantile for VaR use).

    curvature=True also returns d2V/df1^2 from the PC1 pair at no extra
    valuations — the second-order exposure for a 0.5 * curv * f1^2 term
    (PC1 carries most of the variance, so its square is the first Hessian
    term worth having).
    """
    eps = np.broadcast_to(np.asarray(eps, dtype=float), (k,))
    base = price_fn(np.zeros(model.components.shape[1]))
    expos = np.empty(k)
    curv = None
    for j in range(k):
        d = eps[j] * model.components[j] / model.weights
        up, dn = price_fn(d), price_fn(-d)
        expos[j] = (up - dn) / (2.0 * eps[j])
        if j == 0 and curvature:
            curv = (up - 2.0 * base + dn) / eps[j] ** 2
    return expos, curv


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
