"""Historical 1-day VaR for the call-spread book: full reval vs greeks.

Every historical day pair supplies one joint scenario — the spot return, the
fixed-moneyness pillar-grid vol change, the zero-rate and implied-div-yield
changes at each leg's TTM, and the calendar-day gap dt — applied to a fixed
as-of book. Repricing samples the shocked grid at each leg's NEW moneyness
and dt-DECAYED lookup TTM, so smile slide and term roll-down materialize on
the as-of surface exactly as in the sticky-strike-with-roll factor basis
(sticky_strike_dsigma(include_roll=True), same re-anchoring applied to the
as-of grid in scenario_dsigma). The Black-Scholes time input stays at the
base TTM: scenario P&L is pure market-shock P&L, no deterministic
theta/funding — the attribution's equity + vol + rate + div + vanna, not
`time`.

Two engines over the same scenarios:

  full_reval_pnl   BS-reprice every leg under every scenario
                   (n_scenarios x n_legs valuations)
  greeks_pnl       one greeks pass on the as-of date only — delta, gamma,
                   pillar-bucketed vega and vanna (-> k-factor exposures on a
                   fitted PCAModel), per-leg rho/div by 1bp bump — then
                   project each scenario from (dS, factor scores, dr, dq)
                   with zero further option valuations:

                   pnl_k = delta.dS + 0.5 gamma.dS^2
                           + bucket_vega.mu + sum_{j<=k} E_j f_j
                           + bucket_vanna.(k-factor recon dsigma) x dS
                           + rho.dr + divs.dq

With all factors the vol term reproduces bucket_vega.dsigma_scen exactly, so
the remaining gap to full reval is pure linearization (vega convexity plus
delta/gamma truncation), not factor truncation.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vol_pca.attribution import _bs, _rq
from vol_pca.book import EXPIRY_DAYS, NOTIONAL, SPREAD_WIDTH
from vol_pca.factors import factor_exposures, factor_scores
from vol_pca.surface import MONEYNESS, N_PILLARS, TTM_PILLARS, grid_lookup

_EPS = 1e-6
_DB = 1e-4                      # 1bp bump for rho / div sensitivities


@dataclass
class AsOfBook:
    date: np.datetime64
    t: int                      # index of the as-of date in sd
    spot: float
    grid: np.ndarray            # (n_ttm, n_mon) as-of pillar vols
    strike: np.ndarray          # per leg, signed units: short leg negative
    units: np.ndarray
    expiry: np.ndarray
    ttm: np.ndarray
    sig0: np.ndarray
    r0: np.ndarray
    q0: np.ndarray
    v0: np.ndarray              # per-unit base values and greeks
    delta0: np.ndarray
    gamma0: np.ndarray
    vega0: np.ndarray
    vanna0: np.ndarray
    rho0: np.ndarray            # DOLLAR (units-weighted) dV/dr, dV/dq per leg
    dvq0: np.ndarray
    bucket_vega: np.ndarray     # (N_PILLARS,) dollar vega/vanna scattered
    bucket_vanna: np.ndarray    # onto the pillars via the lookup weights

    @property
    def n_legs(self):
        return len(self.strike)

    @property
    def net_delta(self):        # dollar delta: dP&L per 1.0 move in spot
        return float((self.units * self.delta0).sum())


def build_book(sd, t=-1, notional=NOTIONAL):
    """The close-of-day book at sd.dates[t]: every leg sold on days <= t
    (including day t's new spread) that has not yet expired."""
    t = range(len(sd))[t]
    d, s, grid = sd.dates[t], sd.spot[t], sd.grids[t]

    strikes, units, expiries = [], [], []
    for i in range(t + 1):
        exp_i = sd.dates[i] + np.timedelta64(EXPIRY_DAYS, "D")
        if exp_i <= d:
            continue
        s_i = sd.spot[i]
        for k_new, u_new in ((s_i, -notional / s_i),
                             (SPREAD_WIDTH * s_i, notional / s_i)):
            strikes.append(k_new)
            units.append(u_new)
            expiries.append(exp_i)
    strike = np.array(strikes)
    u = np.array(units)
    expiry = np.array(expiries, dtype="datetime64[D]")

    ttm = (expiry - d).astype(float) / 365.0
    mon = strike / s * 100.0
    sig0, w_idx, w = grid_lookup(grid, ttm, mon, want_weights=True)
    r0, q0 = _rq(sd, t, ttm)
    v0, delta0, gamma0, vega0, vanna0 = _bs(s, strike, ttm, sig0, r0, q0,
                                            greeks=True)
    rho0 = u * (_bs(s, strike, ttm, sig0, r0 + _DB, q0) - v0) / _DB
    dvq0 = u * (_bs(s, strike, ttm, sig0, r0, q0 + _DB) - v0) / _DB

    bucket_vega = np.zeros(N_PILLARS)
    bucket_vanna = np.zeros(N_PILLARS)
    np.add.at(bucket_vega, w_idx.ravel(), (w * (u * vega0)[:, None]).ravel())
    np.add.at(bucket_vanna, w_idx.ravel(), (w * (u * vanna0)[:, None]).ravel())

    return AsOfBook(date=d, t=t, spot=s, grid=grid, strike=strike, units=u,
                    expiry=expiry, ttm=ttm, sig0=sig0, r0=r0, q0=q0, v0=v0,
                    delta0=delta0, gamma0=gamma0, vega0=vega0, vanna0=vanna0,
                    rho0=rho0, dvq0=dvq0, bucket_vega=bucket_vega,
                    bucket_vanna=bucket_vanna)


@dataclass
class ScenarioSet:
    dates: np.ndarray           # (n_scen,) day t of each historical move
    ratio: np.ndarray           # (n_scen,) spot_t / spot_{t-1}
    dgrid: np.ndarray           # (n_scen, n_ttm, n_mon) pillar vol changes
    dt: np.ndarray              # (n_scen,) calendar-day gap / 365
    dr: np.ndarray              # (n_scen, n_legs) rate / div-yield changes
    dq: np.ndarray              # at each leg's base TTM

    def __len__(self):
        return len(self.ratio)


def build_scenarios(sd, book, mask=None):
    """One scenario per historical move t-1 -> t (mask selects moves,
    length len(sd) - 1, e.g. to drop post-data-gap days)."""
    idx = np.arange(1, len(sd))
    if mask is not None:
        idx = idx[np.asarray(mask)]
    ratio = sd.spot[idx] / sd.spot[idx - 1]
    dgrid = sd.grids[idx] - sd.grids[idx - 1]
    dt = (sd.dates[idx] - sd.dates[idx - 1]).astype(float) / 365.0
    dr = np.empty((len(idx), book.n_legs))
    dq = np.empty((len(idx), book.n_legs))
    for j, i in enumerate(idx):
        r1, q1 = _rq(sd, i, book.ttm)
        r0, q0 = _rq(sd, i - 1, book.ttm)
        dr[j], dq[j] = r1 - r0, q1 - q0
    return ScenarioSet(dates=sd.dates[idx], ratio=ratio, dgrid=dgrid, dt=dt,
                       dr=dr, dq=dq)


def full_reval_pnl(book, scen):
    """Full BS revaluation of every leg under every scenario (the benchmark:
    len(scen) x book.n_legs valuations)."""
    pnl = np.empty(len(scen))
    for j in range(len(scen)):
        s1 = book.spot * scen.ratio[j]
        sig1 = grid_lookup(book.grid + scen.dgrid[j],
                           np.maximum(book.ttm - scen.dt[j], _EPS),
                           book.strike / s1 * 100.0)
        v1 = _bs(s1, book.strike, book.ttm, sig1,
                 book.r0 + scen.dr[j], book.q0 + scen.dq[j])
        pnl[j] = (book.units * (v1 - book.v0)).sum()
    return pnl


def scenario_dsigma(book, scen):
    """Each scenario's pillar vol changes in the roll-in sticky-strike basis,
    realized on the AS-OF surface: shocked grid sampled at the pillar's
    re-anchored moneyness and dt-decayed TTM, minus the base pillar vol.
    Needs only grid lookups — no option valuations."""
    nt, nm = len(TTM_PILLARS), len(MONEYNESS)
    out = np.empty((len(scen), nt * nm))
    base = book.grid.ravel()
    for j in range(len(scen)):
        mon_q = np.clip(MONEYNESS / scen.ratio[j], MONEYNESS[0], MONEYNESS[-1])
        out[j] = grid_lookup(book.grid + scen.dgrid[j],
                             np.repeat(TTM_PILLARS - scen.dt[j], nm),
                             np.tile(mon_q, nt)) - base
    return out


def book_reval_fn(book):
    """The single pricing entry of the formula-free path: revalue the book
    under (spot, pillar vol shift, parallel rate/div shift), everything else
    frozen at the base. Vol lookups stay at the BASE moneyness — on a
    strike-quoted surface a spot bump does not re-map the vol lookup; smile
    slide belongs to the factor scores, not the equity greeks (bumping spot
    with re-mapped lookups would double-count the slide and badly distort
    the spot x factor crosses). A real simulation-priced book supplies its
    own closure with this signature."""
    mon0 = book.strike / book.spot * 100.0

    def price(spot=None, dsigma=None, dr=0.0, dq=0.0):
        s = book.spot if spot is None else spot
        if dsigma is None:
            sig = book.sig0
        else:
            sig = grid_lookup(book.grid
                              + np.asarray(dsigma).reshape(book.grid.shape),
                              book.ttm, mon0)
        return float((book.units * _bs(s, book.strike, book.ttm,
                                       np.maximum(sig, 1e-4),
                                       book.r0 + dr, book.q0 + dq)).sum())
    return price


def book_price_fn(book):
    """Vol-only view of book_reval_fn (the price_fn signature that
    bumped_factor_exposures expects)."""
    pf = book_reval_fn(book)
    return lambda dsigma: pf(dsigma=dsigma)


@dataclass
class BumpGreeks:
    """Every sensitivity the formula-free projection uses, measured purely
    by revaluation (n_revals of them) — no closed-form greeks anywhere."""
    v0: float
    delta: float                # spot stencil at fixed vol lookup
    gamma: float
    speed: float
    base_mu: float              # value change under the factor-mean move mu
    expos: np.ndarray           # (k,) dV/df_j, central at +/- eps_j
    curv: np.ndarray            # (k,) d2V/df_j^2, free from the same pairs
    cross: np.ndarray           # (k_cross,) d2V/dS df_j, 4-corner stencil
    rho: float                  # dV per 1.0 parallel rate move
    dvq: float                  # dV per 1.0 parallel div-yield move
    eps: np.ndarray             # the score-unit bump sizes used
    n_revals: int


def bump_greeks(book, model, k=5, k_cross=3, eps=None, h_frac=0.01,
                db=5e-4, price_fn=None):
    """The greeks pass of the bump path, priced-by-revaluation only (the
    main code path — closed-form greeks live in build_book solely for tests
    and validation). Defaults follow the affordability assumptions: k=5 vol
    factors, eps = 1 sigma of each factor's fit-sample score
    (model.score_std), spot bumps of 1%/2% (1% ~ 1 sigma of daily SPX),
    rate/div at +/-5bp, spot x factor crosses for PC1-3 (the rolling
    validation: dropping the crosses floods the VaR tail — 166+ books
    understated beyond -30% — while PC4-5 crosses add nothing).

    Budget: 1 base + 4 spot + 1 mean-move + 2k factor pairs (whose +/-
    pairs also hand over the diagonal curvatures free) + 4*k_cross crosses
    + 4 rate/div = 10 + 2k + 4*k_cross revaluations (32 at the defaults).
    """
    pf = price_fn if price_fn is not None else book_reval_fn(book)
    if eps is None:
        if model.score_std is None:
            raise ValueError("model has no score_std; pass eps explicitly")
        eps = model.score_std[:k]
    eps = np.broadcast_to(np.asarray(eps, dtype=float), (k,))
    S = book.spot
    h = h_frac * S
    v0 = pf()
    vs = {m: pf(spot=S + m * h) for m in (-2, -1, 1, 2)}
    delta = (vs[1] - vs[-1]) / (2.0 * h)
    gamma = (vs[1] - 2.0 * v0 + vs[-1]) / h**2
    speed = (vs[2] - 2.0 * vs[1] + 2.0 * vs[-1] - vs[-2]) / (2.0 * h**3)
    base_mu = pf(dsigma=model.mu) - v0

    expos = np.empty(k)
    curv = np.empty(k)
    for j in range(k):
        d = eps[j] * model.components[j] / model.weights
        up, dn = pf(dsigma=d), pf(dsigma=-d)
        expos[j] = (up - dn) / (2.0 * eps[j])
        curv[j] = (up - 2.0 * v0 + dn) / eps[j] ** 2
    cross = np.empty(k_cross)
    for j in range(k_cross):
        d = eps[j] * model.components[j] / model.weights
        cross[j] = (pf(spot=S + h, dsigma=d) - pf(spot=S + h, dsigma=-d)
                    - pf(spot=S - h, dsigma=d) + pf(spot=S - h, dsigma=-d)
                    ) / (4.0 * h * eps[j])
    rho = (pf(dr=db) - pf(dr=-db)) / (2.0 * db)
    dvq = (pf(dq=db) - pf(dq=-db)) / (2.0 * db)
    return BumpGreeks(v0=v0, delta=delta, gamma=gamma, speed=speed,
                      base_mu=base_mu, expos=expos, curv=curv, cross=cross,
                      rho=rho, dvq=dvq, eps=np.array(eps),
                      n_revals=10 + 2 * k + 4 * k_cross)


def bump_pnl(book, bg, scen, model, dsigma_scen=None, scores=None,
             quad=None, cube=True):
    """Scenario P&L from a BumpGreeks pass — the formula-free projection:

        delta.dS + gamma/2 dS^2 [+ speed/6 dS^3]
        + base_mu + sum_j E_j f_j [+ curvature f^2 terms]
        + sum_{j<k_cross} X_j f_j dS + rho.mean(dr) + dvq.mean(dq)

    Factor scores come from scenario_dsigma (grid lookups, no valuations).
    quad: "pc1" adds 0.5*c1*f1^2, "all" every measured diagonal, None (the
    default) neither — whole-history verdict: the quadratic is a daily-P&L
    tool (vol-estimate R2 0.860 -> 0.875), not a VaR tool (it widens k=5
    quantile gaps slightly), so the projection leaves it off while
    bump_greeks always measures the curvatures (they are free).
    Rate/div apply the parallel slopes to a |units| x TTM weighted
    average of the per-leg scenario moves — the statics proxy for where the
    rate/div sensitivity sits in tenor (both scale ~linearly in T). A plain
    mean is NOT safe: the implied q backed out of forwards carries
    1/T-amplified snap noise at the short end, which near-zero-sensitivity
    short legs would otherwise inject into the book term."""
    if scores is None:
        if dsigma_scen is None:
            dsigma_scen = scenario_dsigma(book, scen)
        scores = factor_scores(dsigma_scen, model)
    f = scores[:, :len(bg.expos)]
    dS = book.spot * (scen.ratio - 1.0)
    wT = np.abs(book.units) * book.ttm
    wT = wT / wT.sum()
    est = (bg.delta * dS + 0.5 * bg.gamma * dS**2
           + bg.base_mu + f @ bg.expos
           + (f[:, :len(bg.cross)] @ bg.cross) * dS
           + bg.rho * (scen.dr @ wT) + bg.dvq * (scen.dq @ wT))
    if cube:
        est = est + bg.speed / 6.0 * dS**3
    if quad == "pc1":
        est = est + 0.5 * bg.curv[0] * f[:, 0] ** 2
    elif quad == "all":
        est = est + 0.5 * (f**2 @ bg.curv)
    elif quad is not None:
        raise ValueError(f"unknown quad {quad!r}")
    return est


def greeks_pnl(book, scen, model, ks, dsigma_scen=None, scores=None):
    """Greeks-projected scenario P&L for each factor count in `ks`.

    Returns (est: {k: (n_scen,)}, parts) where parts holds the shared
    delta/gamma/rate/div terms and the per-k vol and vanna terms. Pass
    `scores` to override the factor scores (e.g. the historical scores from
    the fitting sample instead of the scenario-consistent ones)."""
    if scores is None:
        if dsigma_scen is None:
            dsigma_scen = scenario_dsigma(book, scen)
        scores = factor_scores(dsigma_scen, model)
    dS = book.spot * (scen.ratio - 1.0)
    parts = {
        "delta": book.net_delta * dS,
        "gamma": 0.5 * float((book.units * book.gamma0).sum()) * dS**2,
        "rate": scen.dr @ book.rho0,
        "div": scen.dq @ book.dvq0,
        "scores": scores,
    }
    expos = factor_exposures(book.bucket_vega, model)
    base = book.bucket_vega @ model.mu
    parts["vol"], parts["vanna"] = {}, {}
    est = {}
    for k in ks:
        recon = model.mu + (scores[:, :k] @ model.components[:k]) / model.weights
        parts["vol"][k] = base + scores[:, :k] @ expos[:k]
        parts["vanna"][k] = (recon @ book.bucket_vanna) * dS
        est[k] = (parts["delta"] + parts["gamma"] + parts["rate"]
                  + parts["div"] + parts["vol"][k] + parts["vanna"][k])
    return est, parts


def hist_var(pnl, level=0.99):
    """Historical VaR as a positive loss: -(1-level) empirical quantile."""
    return float(-np.percentile(pnl, 100.0 * (1.0 - level)))


def strike_histogram_weights(book, floor_frac=0.05, width=5.0):
    """Statics-only PCA weights (no greeks): a Gaussian kernel of `width`
    moneyness points at EVERY leg's strike, mass = leg gross notional, legs
    split linearly between their two bracketing TTM pillars; normalized to
    peak 1 with a flat floor so no surface region is fully unweighted.

    Two lessons baked in from the 2020-03-16 study: use the full strike
    histogram, not a per-bucket average (a notional-weighted average strike
    gets dragged toward deep-ITM legs that carry no vol risk), and keep the
    floor LOW (~0.05) — high floors re-admit the noisy short-dated far-wing
    pillars that weighting exists to suppress (floor 1.0 = plain cov PCA)."""
    nt, nm = len(TTM_PILLARS), len(MONEYNESS)
    gross = np.abs(book.units) * book.spot
    mon = book.strike / book.spot * 100.0
    i0 = np.clip(np.searchsorted(TTM_PILLARS, book.ttm) - 1, 0, nt - 2)
    frac = np.clip((book.ttm - TTM_PILLARS[i0])
                   / (TTM_PILLARS[i0 + 1] - TTM_PILLARS[i0]), 0.0, 1.0)
    kern = np.exp(-0.5 * ((MONEYNESS[None, :] - mon[:, None]) / width) ** 2)
    w = np.zeros((nt, nm))
    for row in range(nt):
        g = gross * (np.where(i0 == row, 1 - frac, 0.0)
                     + np.where(i0 + 1 == row, frac, 0.0))
        w[row] = g @ kern
    w /= w.max()
    return np.maximum(w, floor_frac).ravel()


def book_speed(book, h_frac=0.01):
    """Book-level d3V/dS3 (dollar speed) from a 5-point spot stencil at fixed
    vol/rate/div — the third-order equity term is speed/6 x dS^3."""
    h = h_frac * book.spot
    v = {m: float((book.units * _bs(book.spot + m * h, book.strike, book.ttm,
                                    book.sig0, book.r0, book.q0)).sum())
         for m in (-2, -1, 1, 2)}
    return (v[2] - 2 * v[1] + 2 * v[-1] - v[-2]) / (2 * h**3)


_ROLL_CTX = None


def _roll_init(sd, model, ks, mask):
    global _ROLL_CTX
    _ROLL_CTX = (sd, model, ks, mask)


def _roll_worker(t):
    sd, model, ks, mask = _ROLL_CTX
    book = build_book(sd, t)
    scen = build_scenarios(sd, book, mask=mask)
    pnl = full_reval_pnl(book, scen)
    dS = book.spot * (scen.ratio - 1.0)
    hedge = book.net_delta * dS
    est, _ = greeks_pnl(book, scen, model, ks,
                        dsigma_scen=scenario_dsigma(book, scen))
    speed = book_speed(book)
    cube = speed / 6.0 * dS**3
    row = {"date": book.date, "n_legs": book.n_legs,
           "net_vega": float((book.units * book.vega0).sum()) * 0.01,
           "net_gamma": float((book.units * book.gamma0).sum()),
           "speed": speed,
           "full_h": hist_var(pnl - hedge), "full_raw": hist_var(pnl)}
    for k in ks:
        for tag, extra in (("", 0.0), ("c", cube)):
            row[f"g{k}{tag}_h"] = hist_var(est[k] + extra - hedge)
            row[f"g{k}{tag}_raw"] = hist_var(est[k] + extra)
    return row


def rolling_var_backtest(sd, model, ks=(3, 4, 5, 10), start=252, mask=None,
                         n_jobs=None):
    """Hedged and raw VaR99 for every as-of date from `start` on, by full
    revaluation and by greeks projection for each k in `ks` with ("g{k}c_*")
    and without ("g{k}_*") the third-order equity term. The whole-history
    scenario set (in-sample throughout — a methodology comparison, not an
    exceedance backtest) is applied to every day's book."""
    import multiprocessing as mp

    ts = range(start, len(sd))
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 2)
    with mp.get_context("fork").Pool(n_jobs, _roll_init,
                                     (sd, model, ks, mask)) as pool:
        rows = pool.map(_roll_worker, ts, chunksize=4)
    return pd.DataFrame(rows).set_index("date")


_ADAPT_CTX = None


def _adapt_init(sd, fit_dsigma, fit_mask, weights_fn, ks, mask):
    global _ADAPT_CTX
    _ADAPT_CTX = (sd, fit_dsigma, fit_mask, weights_fn, ks, mask)


def _adapt_worker(t):
    from vol_pca.factors import fit_pca

    sd, fit_dsigma, fit_mask, weights_fn, ks, mask = _ADAPT_CTX
    book = build_book(sd, t)
    scen = build_scenarios(sd, book, mask=mask)
    model = fit_pca(fit_dsigma, fit_mask=fit_mask, weights=weights_fn(book))
    est, _ = greeks_pnl(book, scen, model, ks,
                        dsigma_scen=scenario_dsigma(book, scen))
    dS = book.spot * (scen.ratio - 1.0)
    hedge = book.net_delta * dS
    cube = book_speed(book) / 6.0 * dS**3
    row = {"date": book.date}
    for k in ks:
        for tag, extra in (("", 0.0), ("c", cube)):
            row[f"s{k}{tag}_h"] = hist_var(est[k] + extra - hedge)
            row[f"s{k}{tag}_raw"] = hist_var(est[k] + extra)
    return row


_BUMP_CTX = None


def _bump_init(sd, model, mask, kw):
    global _BUMP_CTX
    _BUMP_CTX = (sd, model, mask, kw)


def _bump_worker(t):
    sd, model, mask, kw = _BUMP_CTX
    book = build_book(sd, t)
    scen = build_scenarios(sd, book, mask=mask)
    scores = factor_scores(scenario_dsigma(book, scen), model)
    bg = bump_greeks(book, model, **kw)
    dS = book.spot * (scen.ratio - 1.0)
    hedge = bg.delta * dS
    k = len(bg.expos)
    row = {"date": book.date, "n_revals": bg.n_revals,
           "delta": bg.delta, "vega1": bg.expos[0]}
    for tag, cube in ((f"b{k}c", True), (f"b{k}", False)):
        est = bump_pnl(book, bg, scen, model, scores=scores, cube=cube)
        row[f"{tag}_h"] = hist_var(est - hedge)
        row[f"{tag}_raw"] = hist_var(est)
    return row


def rolling_var_bump(sd, model, start=252, mask=None, n_jobs=None, **bump_kw):
    """Rolling VaR of the formula-free bump path (bump_greeks + bump_pnl at
    their defaults unless overridden via bump_kw): columns "b{k}c_*" (with
    the third-order equity term) and "b{k}_*" (without), hedged with the
    stencil delta. Greeks-only — join with rolling_var_backtest output on
    date for the full-reval benchmark."""
    import multiprocessing as mp

    ts = range(start, len(sd))
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 2)
    with mp.get_context("fork").Pool(n_jobs, _bump_init,
                                     (sd, model, mask, bump_kw)) as pool:
        rows = pool.map(_bump_worker, ts, chunksize=4)
    return pd.DataFrame(rows).set_index("date")


def rolling_var_adaptive(sd, weights_fn, fit_dsigma, fit_mask=None,
                         ks=(4, 10), start=252, mask=None, n_jobs=None):
    """Rolling greeks VaR with book-adaptive weights: every as-of date refits
    the PCA on (fit_dsigma, fit_mask) with weights_fn(book) — no per-date
    option valuations beyond the greeks pass, and no full reval here (join
    with rolling_var_backtest output on date for the benchmark). Columns
    "s{k}_h" etc. mirror the backtest's "g{k}_h" naming."""
    import multiprocessing as mp

    ts = range(start, len(sd))
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 2)
    with mp.get_context("fork").Pool(n_jobs, _adapt_init,
            (sd, fit_dsigma, fit_mask, weights_fn, ks, mask)) as pool:
        rows = pool.map(_adapt_worker, ts, chunksize=4)
    return pd.DataFrame(rows).set_index("date")
