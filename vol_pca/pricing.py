"""Black-76 call pricing on the forward, with spot greeks.

Spot delta/gamma assume the forward ratio F/S stays fixed as spot moves.
Expired or zero-vol inputs collapse to discounted intrinsic on the forward.
"""

import numpy as np
from scipy.stats import norm


def black76(fwd, strike, ttm, sigma, df, spot=None, greeks=False):
    fwd, strike, ttm, sigma, df = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, dtype=float)) for a in (fwd, strike, ttm, sigma, df)])
    live = (ttm > 0) & (sigma > 0)
    st = np.where(live, sigma * np.sqrt(np.maximum(ttm, 1e-12)), 1.0)
    d1 = np.where(live, (np.log(fwd / strike) + 0.5 * st**2) / st, np.where(fwd > strike, np.inf, -np.inf))
    d2 = d1 - st
    price = df * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    if not greeks:
        return price
    fs = fwd / np.asarray(spot, dtype=float)
    delta = df * norm.cdf(d1) * fs
    gamma = np.where(live, df * norm.pdf(d1) / (fwd * st) * fs**2, 0.0)
    vega = np.where(live, df * fwd * norm.pdf(d1) * np.sqrt(np.maximum(ttm, 0.0)), 0.0)
    vanna = np.where(live, -df * norm.pdf(d1) * d2 / sigma * fs, 0.0)   # d2V/dS dsigma
    return price, delta, gamma, vega, vanna
