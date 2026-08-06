"""Rainbow-option valuation: implied marginals glued by a Gaussian copula.

The structure under study (user spec 2026-08-05): a call spread on the ranked
basket
    B = 0.5 * best(P_spx, P_sx5e) + 0.3 * worst(P_spx, P_sx5e) + 0.2 * P_hsi,
with P_i = S_i(T) / S_i(0) the gross performance of each index, sold at $1M
notional with strikes 100% / 112% of basket performance (the cap is wider
than the single-index book's 110% because the ranked basket prices cheaper).
Valuation is simulation-shaped on purpose — this is the stand-in for the real
book's simulation-priced exotics:

- Each index's risk-neutral marginal at expiry comes from its own surface by
  Breeden-Litzenberger: undiscounted calls C(x) along a performance grid
  x = strike / spot, smiles read through the standard bicubic pillar
  interpolant; P(X <= x) = 1 + C'(x), density C''(x). Beyond the quoted
  50-150 moneyness range the smile continues with its edge slope, tapered
  to a bounded asymptote — the pricing grid's flat clamp is a kink in
  sigma(x) exactly where the put wing still has vega, which would teleport
  a Dirac of probability mass (~2-3% for SPX) and bias the recovered mean
  ~30bp below the forward, while an untapered slope violates Lee's moment
  bound on rising call wings.
- Dependence is a Gaussian copula over the three marginals with a constant
  historical spot-correlation matrix (the data carries no implied-correlation
  input; multi-day return horizons wash out the asynchronous-close bias).
- FX and quanto adjustments are deliberately ignored: each marginal lives
  under its own forward measure (forwards off its own curve) and the basket
  payoff is discounted on the USD (SPX) curve.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm, qmc

from vol_pca.pricing import black76
from vol_pca.surface import grid_lookup

WEIGHTS = (0.5, 0.3, 0.2)      # best / worst of (SPX, SX5E), then HSI
GEO_WEIGHTS = (0.4, 0.4, 0.2)  # fixed-weight control variate basket
K_LO, K_HI = 1.00, 1.12
NOTIONAL = 1_000_000.0


@dataclass
class ImpliedMarginal:
    x: np.ndarray          # performance grid K / S0
    vols: np.ndarray       # smile along x (decimals), flat beyond 50-150
    call: np.ndarray       # undiscounted E[(X - x)+]
    cdf: np.ndarray        # P(X <= x), monotone-adjusted
    pdf: np.ndarray        # raw Breeden-Litzenberger density C''(x)
    fwd_ratio: float       # F(T) / S0, the model-free E[X]
    mean: float            # E[X] recovered from the cdf (consistency check)

    def quantile(self, u):
        keep = np.concatenate([[True], np.diff(self.cdf) > 1e-12])
        return np.interp(u, self.cdf[keep], self.x[keep])


def date_index(sd, date):
    i = int(np.searchsorted(sd.dates, np.datetime64(date, "D")))
    if i >= len(sd.dates) or sd.dates[i] != np.datetime64(date, "D"):
        raise KeyError(f"{date} not in surface dates")
    return i


def implied_marginal(sd, i, ttm, n=2001, n_sig=10.0):
    """Risk-neutral distribution of S_T / S0 implied by date i's surface.

    The grid spans +/- n_sig * (worst wing vol) * sqrt(T) in log-performance
    around the forward, so both cdf ends are numerically 0/1 at any TTM.
    Wings beyond the quoted moneyness range continue with the smile's edge
    slope, tapered exponentially (scale 50 moneyness points) so sigma stays
    C1 at the join — no clamp kink polluting the density — but asymptotes to
    a bounded level: an unbounded linear wing violates Lee's moment bound
    whenever the call-wing slope is positive (HSI's is) and manufactures
    far-right probability mass. A floor at a quarter of the edge vol guards
    steep falling wings; any floor kink lands where vega is ~0.
    """
    f = float(sd.forward(i, ttm) / sd.spot[i])
    grid = sd.grids[i]
    lo, hi = sd.moneyness[0], sd.moneyness[-1]
    # the bicubic lookup at fixed ttm is exactly a moneyness CubicSpline over
    # the TTM-collapsed pillar row (lookups at pillar moneyness collapse it)
    row = grid_lookup(grid, np.full(len(sd.moneyness), float(ttm)), sd.moneyness)
    smile = CubicSpline(sd.moneyness, row)
    half = max(n_sig * row.max() * np.sqrt(ttm), 1.0)
    x = np.geomspace(f * np.exp(-half), f * np.exp(half), n)
    mon = 100.0 * x
    vols = smile(np.clip(mon, lo, hi))
    s_lo, s_hi = smile(lo, 1), smile(hi, 1)   # exact edge slopes, vol / mon point
    lam = 50.0                                # taper scale, moneyness points
    vols = np.where(mon < lo, row[0] + s_lo * lam * np.expm1(-(lo - mon) / lam), vols)
    vols = np.where(mon > hi, row[-1] - s_hi * lam * np.expm1(-(mon - hi) / lam), vols)
    vols = np.maximum(vols, 0.25 * min(row[0], row[-1]))
    call = black76(f, x, ttm, vols, 1.0)
    cdf_raw = 1.0 + np.gradient(call, x)
    pdf = np.gradient(cdf_raw, x)
    cdf = np.clip(np.maximum.accumulate(cdf_raw), 0.0, 1.0)
    mean = x[0] + np.trapezoid(1.0 - cdf, x)   # E[X] = int (1 - F); F ~ 0 below x[0]
    return ImpliedMarginal(x=x, vols=vols, call=call, cdf=cdf, pdf=pdf,
                           fwd_ratio=f, mean=float(mean))


def spot_panel(sds):
    """Close series aligned on the dates every surface quotes (inner join)."""
    ser = {name: pd.Series(sd.spot, index=pd.DatetimeIndex(sd.dates))
           for name, sd in sds.items()}
    return pd.DataFrame(ser).dropna()


def historical_corr(spots, horizon=1):
    """Pearson correlation of horizon-day log returns (overlapping if > 1).

    HSI (and to a lesser degree SX5E) closes hours before SPX, so
    same-calendar-day returns understate the true co-movement; multi-day
    horizons wash the asynchronous-close bias out.
    """
    return np.log(spots).diff(horizon).dropna().corr()


def sample_performances(marginals, corr, n_paths, seed=0, method="pseudo"):
    """Gaussian-copula draw: correlated normals -> uniforms -> each marginal's
    quantile. Returns performances of shape (n_paths, n_indices).

    method="sobol" swaps pseudo-random normals for scrambled Sobol points:
    the identical estimator, but quasi-MC integration error decays ~1/N
    instead of 1/sqrt(N) in this low dimension. Use powers of two for
    n_paths to keep the point set balanced.
    """
    chol = np.linalg.cholesky(np.asarray(corr, dtype=float))
    if method == "sobol":
        eng = qmc.Sobol(d=len(marginals), scramble=True, seed=seed)
        k2 = int(np.log2(n_paths))
        z = norm.ppf(eng.random_base2(k2) if 2 ** k2 == n_paths
                     else eng.random(n_paths))
    elif method == "pseudo":
        z = np.random.default_rng(seed).standard_normal((n_paths, len(marginals)))
    else:
        raise ValueError(f"unknown method {method!r}")
    u = norm.cdf(z @ chol.T)
    return np.column_stack([m.quantile(u[:, j]) for j, m in enumerate(marginals)])


def rainbow_basket(perf, weights=WEIGHTS):
    """Ranked basket: w_best / w_worst on the first two columns, w3 on the
    third. Equals the plain (w_best+w_worst)/2-weighted pair basket plus
    (w_best - w_worst)/2 * |p1 - p2| — the rank premium is a straddle on the
    performance spread."""
    w_best, w_worst, w3 = weights
    p1, p2 = perf[:, 0], perf[:, 1]
    return (w_best * np.maximum(p1, p2) + w_worst * np.minimum(p1, p2)
            + w3 * perf[:, 2])


def call_spread_payoff(b, k_lo=K_LO, k_hi=K_HI):
    return np.clip(b - k_lo, 0.0, k_hi - k_lo)


def geo_basket(perf, weights_geo=GEO_WEIGHTS):
    """Fixed-weight geometric basket prod p_i^w_i — the control-variate
    underlying. 40:40:20 replaces the 50:30 rank weights on the (SPX, SX5E)
    pair by their average, removing the best/worst kink; the geometric mean
    replaces the arithmetic one so the basket is a product, which is what
    makes its call spread integrable one index at a time."""
    return np.exp(np.log(perf) @ np.asarray(weights_geo, dtype=float))


def price_rainbow(marginals, corr, df, n_paths=400_000, seed=0, weights=WEIGHTS,
                  k_lo=K_LO, k_hi=K_HI, notional=NOTIONAL, method="pseudo",
                  perf_to_date=(1.0, 1.0, 1.0), cv=None, cv_beta=1.0,
                  cv_nodes=(24, 24, 257), weights_geo=GEO_WEIGHTS):
    """One-call Monte Carlo valuation; returns price and distribution stats.
    (The reported se is the pseudo-random formula; under method="sobol" the
    actual integration error is far smaller.)

    perf_to_date seasons the option: per-index S_asof / S_sale factors, so a
    leg's total performance is g * X with X drawn from the as-of marginal at
    the remaining TTM, while the strikes stay in sale-date units.

    cv="geo" turns on the control variate: the same call spread on the
    fixed-weight geometric basket, priced on the same paths, its noisy MC
    mean replaced by the exact price_geo_quad expectation —
    pv = MC[Y] - beta * (MC[C] - E[C]). cv_beta is 1.0 (unbiased for any
    fixed value) or "fit" for the per-run least-squares coefficient.

    Measured verdict (2026-08-06, 24 scrambles per config): the CV is
    textbook-strong under method="pseudo" (std $1,432 -> $170 at 2,048
    paths, calm 1Y; ~30-70x variance reduction) but does NOT compose with
    method="sobol", the project's engine: scrambled Sobol has already
    integrated the smooth component the kink-free control shares, so the
    correction only adds the control's independent noise (calm std $47 ->
    $58 at 2,048; ratio >= 1 at every N on covid surfaces, oracle anchor).
    Plain Sobol dominates pseudo+CV at equal paths everywhere measured, so
    this stays a documented negative result, not a production path.
    """
    perf = sample_performances(marginals, corr, n_paths, seed=seed, method=method)
    perf = perf * np.asarray(perf_to_date, dtype=float)
    b = rainbow_basket(perf, weights)
    pay = call_spread_payoff(b, k_lo, k_hi)
    pv = df * pay.mean() * notional
    out = {"pv": float(pv), "se": float(df * pay.std() / np.sqrt(n_paths) * notional),
           "pct_notional": float(pv / notional), "mean_basket": float(b.mean()),
           "p_itm": float((b > k_lo).mean()), "p_capped": float((b > k_hi).mean())}
    if cv == "geo":
        c_pay = call_spread_payoff(geo_basket(perf, weights_geo), k_lo, k_hi)
        ec = price_geo_quad(marginals, corr, df, *cv_nodes,
                            weights_geo=weights_geo, k_lo=k_lo, k_hi=k_hi,
                            notional=notional, perf_to_date=perf_to_date)["pv"]
        beta = (float(np.cov(pay, c_pay)[0, 1] / max(np.var(c_pay), 1e-300))
                if cv_beta == "fit" else float(cv_beta))
        mc_c = float(df * c_pay.mean() * notional)
        out["pv_plain"] = out["pv"]
        out["pv"] = float(pv - beta * (mc_c - ec))
        out["pct_notional"] = out["pv"] / notional
        out.update(cv_ec=float(ec), cv_mc=mc_c, cv_beta=beta,
                   cv_corr=float(np.corrcoef(pay, c_pay)[0, 1]))
    elif cv is not None:
        raise ValueError(f"unknown control variate {cv!r}")
    return out


def marginal_call_spread(marg, df, k_lo=K_LO, k_hi=K_HI, n=2001):
    """Single-index call spread by quadrature on the marginal, no paths:
    E[(X - a)+ - (X - b)+] = int_a^b P(X > s) ds."""
    xs = np.linspace(k_lo, k_hi, n)
    return float(df * np.trapezoid(1.0 - np.interp(xs, marg.x, marg.cdf), xs))


def price_rainbow_quad(marginals, corr, df, n_outer=48, n_inner=48,
                       weights=WEIGHTS, k_lo=K_LO, k_hi=K_HI, notional=NOTIONAL,
                       perf_to_date=(1.0, 1.0, 1.0)):
    """Deterministic pricer: same model as price_rainbow, no paths, no seed.

    The third marginal is integrated out analytically conditional on the
    first two copula normals — the conditional payoff is a call spread in
    P3 with shifted strikes, priced as int_a^b P(P3 > p | .) dp on the
    marginal's own grid via its normal-score table. The remaining 2-D
    integral runs Gauss-Hermite in the first normal and, per outer node,
    Gauss-Legendre in copula-uniform space for the second, SPLIT at the
    best/worst rank kink P1 = P2 so every quadrature panel sees a smooth
    integrand (an unsplit rule would only converge algebraically). Node
    count n_outer * 2 * n_inner; agreement with the MC engines is floored
    by the marginals' grid resolution, not by quadrature error.
    perf_to_date seasons the option exactly as in price_rainbow.
    """
    w_b, w_w, w3 = weights
    g1, g2, g3 = perf_to_date
    m1, m2, m3 = marginals
    L = np.linalg.cholesky(np.asarray(corr, dtype=float))
    eps = 1e-16

    z1, w1 = np.polynomial.hermite_e.hermegauss(n_outer)
    w1 = w1 / np.sqrt(2.0 * np.pi)
    p1 = g1 * m1.quantile(norm.cdf(z1))

    # normal-score tables Phi^-1(F(x)) turn conditional copula probabilities
    # into plain Gaussian cdf evaluations
    psi2 = norm.ppf(np.clip(m2.cdf, eps, 1.0 - eps))
    psi3 = norm.ppf(np.clip(m3.cdf, eps, 1.0 - eps))

    # inner split point: g2 * P2 crosses the (seasoned) p1 at y2* in the
    # space of the second independent normal (z2 = L21 z1 + L22 y2)
    v_star = norm.cdf((np.interp(p1 / g2, m2.x, psi2) - L[1, 0] * z1) / L[1, 1])
    gx, gw = np.polynomial.legendre.leggauss(n_inner)
    gx, gw = 0.5 * (gx + 1.0), 0.5 * gw
    v = np.concatenate([v_star[:, None] * gx,
                        v_star[:, None] + (1.0 - v_star)[:, None] * gx], axis=1)
    wv = np.concatenate([v_star[:, None] * gw, (1.0 - v_star)[:, None] * gw], axis=1)

    y2 = norm.ppf(np.clip(v, eps, 1.0 - eps))
    z2 = L[1, 0] * z1[:, None] + L[1, 1] * y2
    p2 = g2 * m2.quantile(norm.cdf(z2).ravel()).reshape(z2.shape)

    A = (w_b * np.maximum(p1[:, None], p2) + w_w * np.minimum(p1[:, None], p2)).ravel()
    mu = (L[2, 0] * z1[:, None] + L[2, 1] * y2).ravel()
    s3 = max(L[2, 2], 1e-12)

    w3g = w3 * g3
    a, b = (k_lo - A) / w3g, (k_hi - A) / w3g      # call-spread strikes in X3
    x3 = m3.x
    S = norm.cdf((mu[:, None] - psi3[None, :]) / s3)          # P(P3 > x | .)
    C = np.concatenate([np.zeros((len(mu), 1)),
                        np.cumsum(0.5 * (S[:, 1:] + S[:, :-1]) * np.diff(x3),
                                  axis=1)], axis=1)

    def cum_at(q):
        qc = np.clip(q, x3[0], x3[-1])
        i = np.clip(np.searchsorted(x3, qc), 1, len(x3) - 1)
        t = np.clip((qc - x3[i - 1]) / (x3[i] - x3[i - 1]), 0.0, 1.0)
        c0 = np.take_along_axis(C, (i - 1)[:, None], 1)[:, 0]
        return c0 + t * (np.take_along_axis(C, i[:, None], 1)[:, 0] - c0)

    below = np.maximum(np.minimum(b, x3[0]) - a, 0.0)   # survival == 1 under the grid
    cond = w3g * (below + cum_at(b) - cum_at(a))
    pv = df * float((w1[:, None] * wv * cond.reshape(wv.shape)).sum()) * notional
    return {"pv": pv, "n_nodes": int(wv.size)}


def price_geo_quad(marginals, corr, df, n_outer=24, n_inner=24, n_y=257,
                   weights_geo=GEO_WEIGHTS, k_lo=K_LO, k_hi=K_HI,
                   notional=NOTIONAL, perf_to_date=(1.0, 1.0, 1.0)):
    """Deterministic price of the call spread on the fixed-weight geometric
    basket — the exact expectation the control variate is anchored to, under
    the same copula and implied marginals as the MC engines.

    Because the geometric basket is a product, conditional on the first two
    copula normals it is a monotone power of P3, so its call spread prices
    as int_{k_lo}^{k_hi} P(G > y | .) dy — the survival curve read through
    the third marginal's normal-score table on one shared y grid. No rank
    kink means no panel split, and the payoff's compact support [k_lo, k_hi]
    means n_y stays two orders of magnitude under the marginal grid: the
    integrand tensor is (n_outer * n_inner, n_y), ~15x lighter than the
    rainbow quadrature at equal outer resolution. Outer x inner run plain
    Gauss-Hermite in the two independent normals (smooth integrand — the
    conditional expectation mollifies the strike kinks).

    Validated against the closed-form Black price in the flat-lognormal
    world (~2bp, the marginal-grid floor). Calm-date anchor error vs dense
    Sobol is ~$10 at the default nodes, but crash surfaces inherit the
    rainbow quadrature's documented oscillatory tail (2020-03-16: -$513 at
    24x24, +$756 at 96x96) — the violence lives in the quantile tables, so
    removing the rank kink does not cure it."""
    w1g, w2g, w3g = weights_geo
    g1, g2, g3 = perf_to_date
    m1, m2, m3 = marginals
    L = np.linalg.cholesky(np.asarray(corr, dtype=float))
    eps = 1e-16

    z1, h1 = np.polynomial.hermite_e.hermegauss(n_outer)
    y2, h2 = np.polynomial.hermite_e.hermegauss(n_inner)
    h1, h2 = h1 / np.sqrt(2.0 * np.pi), h2 / np.sqrt(2.0 * np.pi)

    p1 = g1 * m1.quantile(norm.cdf(z1))
    z2 = L[1, 0] * z1[:, None] + L[1, 1] * y2[None, :]
    p2 = g2 * m2.quantile(norm.cdf(z2).ravel()).reshape(z2.shape)
    A = (p1[:, None] ** w1g * p2 ** w2g).ravel()
    mu = (L[2, 0] * z1[:, None] + L[2, 1] * y2[None, :]).ravel()
    s3 = max(L[2, 2], 1e-12)

    psi3 = norm.ppf(np.clip(m3.cdf, eps, 1.0 - eps))
    y = np.linspace(k_lo, k_hi, n_y)
    xq = (y[None, :] / A[:, None]) ** (1.0 / w3g) / g3     # G > y  <=>  X3 > xq
    S = norm.cdf((mu[:, None] - np.interp(xq, m3.x, psi3)) / s3)
    cond = np.trapezoid(S, y, axis=1)
    pv = df * float((h1[:, None] * h2[None, :]).ravel() @ cond) * notional
    return {"pv": pv, "n_nodes": n_outer * n_inner}


def bs_implied_vol(price, fwd, strike, ttm, df=1.0, hi=5.0):
    """Black-76 implied vol, elementwise by root bracketing; NaN where the
    price sits outside the attainable (intrinsic, forward) band. Used to
    re-imply a smile from simulated vanilla prices as a round-trip check."""
    price, fwd, strike = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, dtype=float)) for a in (price, fwd, strike)])
    out = np.full(price.shape, np.nan)
    for i in np.ndindex(price.shape):
        p, f, k = price[i], fwd[i], strike[i]
        if df * max(f - k, 0.0) < p < df * f:
            out[i] = brentq(lambda s: black76(f, k, ttm, s, df)[0] - p, 1e-6, hi)
    return out
