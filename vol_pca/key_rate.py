"""Key-rate vega and skew: fixed-shape bucketed vol sensitivities.

The first alternative to PCA factors under the project goal (VaR from a few
greeks instead of per-scenario revaluation): instead of exposures to fitted
statistical loadings, the book's vol risk is two desk-style sensitivities
per quarterly maturity bucket,

  level   parallel bump of the whole smile at the spread's own maturity
          (its plain vega -- both legs bumped together)
  skew    a moneyness tilt through ATM: +1 vol pt at 105, -1 at 95, i.e.
          (m - 100) / 5 vol pts at moneyness m, UNclamped into the wings

grouped onto the 3/6/9/12M knots with tent (linear-interpolation) weights in
TTM, so the four level sensitivities sum exactly to the book's net parallel
vega and there are no bucket-boundary seams. TTMs below 3M clamp onto the 3M
knot. The realized moves are single-point reads off the day's sticky-strike
pillar changes -- no fitting sample, no fitted loadings, and no per-pillar
vega scatter (hence none of the bucketed-vega side lobes):

  level_q = dsigma at (knot_q, moneyness 100)
  skew_q  = [dsigma(knot_q, 110) - dsigma(knot_q, 90)] / 4

(the /4: the unit tilt moves the 90-110 spread by 4 vol pts, so a score of
0.01 means the literal "+1% at 105" bump happened). Vol P&L estimate =
sum_q L_q level_q + S_q skew_q, and the same grouping applied to u * vanna
gives the key-rate vanna term for VaR. Everything is per-leg shape
EVALUATION, never a pillar-grid representation of the shapes -- a tent
peaking at the 9M knot cannot even be represented on the TTM pillars (no
0.75 node); only the score reads touch the pillar grid, through the same
bicubic interpolant pricing uses. Sensitivities are in $ per 1.00 (decimal
vol) of unit shape; multiply by 0.01 for the per-vol-pt number.
"""

import os

import numpy as np
import pandas as pd

from vol_pca.attribution import _bs
from vol_pca.book import EXPIRY_DAYS, NOTIONAL, SPREAD_WIDTH
from vol_pca.pricing import black76
from vol_pca.surface import N_PILLARS, TTM_PILLARS, grid_lookup
from vol_pca.var import (book_speed, build_book, build_scenarios, hist_var,
                         scenario_dsigma)

KNOTS = np.array([0.25, 0.5, 0.75, 1.0])       # 3/6/9/12M maturity knots
KR_NAMES = [f"{kind}_{m}m" for kind in ("lvl", "skw") for m in (3, 6, 9, 12)]


def tilt(mon):
    """The unit skew shape in moneyness: +1 at 105, -1 at 95, unclamped."""
    return (np.asarray(mon, dtype=float) - 100.0) / 5.0


def tent_weights(ttm):
    """(n, n_knots) hat-function weights of each TTM onto the knots: linear
    interpolation between neighbours, all mass on the end knot outside
    [KNOTS[0], KNOTS[-1]]. Rows sum to 1, so tent-grouped sensitivities sum
    to the ungrouped total."""
    t = np.clip(np.atleast_1d(np.asarray(ttm, dtype=float)),
                KNOTS[0], KNOTS[-1])
    i0 = np.clip(np.searchsorted(KNOTS, t, side="right") - 1, 0,
                 len(KNOTS) - 2)
    frac = (t - KNOTS[i0]) / (KNOTS[i0 + 1] - KNOTS[i0])
    w = np.zeros((len(t), len(KNOTS)))
    rows = np.arange(len(t))
    w[rows, i0] = 1.0 - frac
    w[rows, i0 + 1] = frac
    return w


def key_rate_sens(dollar, mon, ttm):
    """Group per-leg dollar sensitivities (u * vega, or u * vanna for the
    spot-vol cross) onto the 8 key-rate shapes, by evaluating each shape at
    the leg's own (TTM, moneyness): level = tent weight, skew = tent weight
    x tilt(mon). Returns (8,) ordered as KR_NAMES."""
    T = tent_weights(ttm)
    d = np.asarray(dollar, dtype=float)
    return np.concatenate([T.T @ d, T.T @ (d * tilt(mon))])


def score_matrix():
    """(8, N_PILLARS) linear map from a pillar dsigma vector to the key-rate
    scores, built from the pricing interpolant's lookup weights: level rows
    read the ATM column at each knot (only the 9M knot actually
    interpolates in TTM), skew rows read (110-col - 90-col) / 4."""
    dummy = np.zeros((len(TTM_PILLARS), N_PILLARS // len(TTM_PILLARS)))
    ttm_q = np.repeat(KNOTS, 3)
    mon_q = np.tile(np.array([90.0, 100.0, 110.0]), len(KNOTS))
    _, w_idx, w = grid_lookup(dummy, ttm_q, mon_q, want_weights=True)
    R = np.zeros((len(ttm_q), N_PILLARS))
    np.add.at(R, (np.arange(len(ttm_q))[:, None], w_idx), w)
    return np.vstack([R[1::3], (R[2::3] - R[0::3]) / 4.0])


def key_rate_scores(dsigma):
    """(n, 8) realized key-rate moves from (n, N_PILLARS) pillar dsigma rows
    (decimal vols): ATM level move and 90-110 spread move / 4 at each knot."""
    return np.atleast_2d(np.asarray(dsigma, dtype=float)) @ score_matrix().T


def daily_key_rate_sens(sd, notional=NOTIONAL):
    """Previous-close key-rate sensitivities for every P&L day: replays the
    book with exactly simulate_book's leg set and vegas (legs created before
    day t and unexpired at the previous close, including legs that settle at
    t -- the same convention bucket_vega uses), grouped by tent / tilt
    instead of scattered onto pillars. Returns (dates (n-1,), sens (n-1, 8))
    with row t-1 holding the exposures for the move t-1 -> t."""
    n = len(sd)
    created = np.repeat(np.arange(n), 2)
    strike = np.empty(2 * n)
    units = np.empty(2 * n)
    strike[0::2], units[0::2] = sd.spot, -notional / sd.spot
    strike[1::2], units[1::2] = SPREAD_WIDTH * sd.spot, notional / sd.spot
    expiry = sd.dates[created] + np.timedelta64(EXPIRY_DAYS, "D")

    sens = np.zeros((n - 1, 2 * len(KNOTS)))
    for t in range(1, n):
        live = (created < t) & (expiry > sd.dates[t - 1])
        k, u = strike[live], units[live]
        ttm0 = (expiry[live] - sd.dates[t - 1]).astype(float) / 365.0
        mon0 = k / sd.spot[t - 1] * 100.0
        sig0 = grid_lookup(sd.grids[t - 1], ttm0, mon0)
        f0, df0 = sd.forward(t - 1, ttm0), sd.discount(t - 1, ttm0)
        _, _, _, vega0, _ = black76(f0, k, ttm0, sig0, df0,
                                    spot=sd.spot[t - 1], greeks=True)
        sens[t - 1] = key_rate_sens(u * vega0, mon0, ttm0)
    return sd.dates[1:], sens


def key_rate_pnl(book, scen, dsigma_scen=None):
    """Scenario P&L from key-rate greeks -- greeks_pnl with the PCA vol and
    vanna terms replaced by the fixed-shape projection:

        delta.dS + 0.5 gamma dS^2 + sens_vega.scores
        + (sens_vanna.scores) dS + rho.dr + div.dq

    Scores are the scenario's key-rate moves read off scenario_dsigma (the
    as-of-anchored roll-in sticky-strike pillar moves), so they carry the
    full move including roll-down; there is no fit and no mean term.
    Returns (est, parts)."""
    if dsigma_scen is None:
        dsigma_scen = scenario_dsigma(book, scen)
    scores = key_rate_scores(dsigma_scen)
    mon0 = book.strike / book.spot * 100.0
    sv = key_rate_sens(book.units * book.vega0, mon0, book.ttm)
    sx = key_rate_sens(book.units * book.vanna0, mon0, book.ttm)
    dS = book.spot * (scen.ratio - 1.0)
    parts = {
        "delta": book.net_delta * dS,
        "gamma": 0.5 * float((book.units * book.gamma0).sum()) * dS**2,
        "vol": scores @ sv,
        "vanna": (scores @ sx) * dS,
        "rate": scen.dr @ book.rho0,
        "div": scen.dq @ book.dvq0,
        "scores": scores, "sens_vega": sv, "sens_vanna": sx,
    }
    est = (parts["delta"] + parts["gamma"] + parts["vol"] + parts["vanna"]
           + parts["rate"] + parts["div"])
    return est, parts


def bumped_key_rate_sens(book, eps=0.01):
    """The formula-free measurement of the same 8 sensitivities: revalue the
    book with every leg's vol shifted by +/- eps x shape(leg) -- the shape
    evaluated at the leg's own TTM and moneyness, i.e. each spread bumped in
    parallel (level) or by the tilt at its strikes (skew) at its own
    maturity -- and take central differences. 16 revaluations at the
    defaults; eps = 0.01 is the literal "+1 vol pt at 105" bump. Matches
    key_rate_sens(u * vega0, ...) up to O(eps^2) volga."""
    mon0 = book.strike / book.spot * 100.0
    T = tent_weights(book.ttm)
    shapes = np.concatenate([T, T * tilt(mon0)[:, None]], axis=1)
    out = np.empty(shapes.shape[1])
    for j in range(shapes.shape[1]):
        d = eps * shapes[:, j]
        up = float((book.units * _bs(book.spot, book.strike, book.ttm,
                                     np.maximum(book.sig0 + d, 1e-4),
                                     book.r0, book.q0)).sum())
        dn = float((book.units * _bs(book.spot, book.strike, book.ttm,
                                     np.maximum(book.sig0 - d, 1e-4),
                                     book.r0, book.q0)).sum())
        out[j] = (up - dn) / (2.0 * eps)
    return out


_KR_CTX = None


def _kr_init(sd, mask):
    global _KR_CTX
    _KR_CTX = (sd, mask)


def _kr_worker(t):
    sd, mask = _KR_CTX
    book = build_book(sd, t)
    scen = build_scenarios(sd, book, mask=mask)
    est, _ = key_rate_pnl(book, scen)
    dS = book.spot * (scen.ratio - 1.0)
    hedge = book.net_delta * dS
    cube = book_speed(book) / 6.0 * dS**3
    row = {"date": book.date}
    for tag, extra in (("kr", 0.0), ("krc", cube)):
        row[f"{tag}_h"] = hist_var(est + extra - hedge)
        row[f"{tag}_raw"] = hist_var(est + extra)
    return row


def rolling_var_keyrate(sd, start=252, mask=None, n_jobs=None):
    """Rolling hedged and raw VaR99 of the key-rate projection for every
    as-of date from `start` on, with ("krc_*") and without ("kr_*") the
    third-order equity term. Greeks-only and fit-free -- join with the
    cached rolling_var_backtest output on date for the full-reval benchmark
    and the PCA columns."""
    import multiprocessing as mp

    ts = range(start, len(sd))
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 2)
    with mp.get_context("fork").Pool(n_jobs, _kr_init, (sd, mask)) as pool:
        rows = pool.map(_kr_worker, ts, chunksize=4)
    return pd.DataFrame(rows).set_index("date")
