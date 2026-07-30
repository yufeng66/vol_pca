"""Independent Black-Scholes input attribution for the call-spread book.

Unlike book.py's sequential (waterfall) attribution, every component here is
an independent single-input revaluation from the same previous-close base:
V(one input at its new value, every other input held) - V0. The inputs are
the Black-Scholes arguments:

  equity  spot S0 -> S1 at fixed vol; split into delta (dV/dS x dS),
          gamma (0.5 d2V/dS2 x dS^2) and higher order
  vol     the option's own implied vol input sigma0 -> sigma1, where sigma1
          is looked up at the NEW date/TTM/moneyness -- so this component
          carries surface change, term-structure roll and smile slide
  rate    zero rate r (from the discount curve) at the old TTM
  div     dividend yield q (backed out of forward vs spot) at the old TTM
  time    TTM shrinks, all other inputs held
  vanna   exact spot-vol cross: V(S1,sig1) - V(S1,sig0) - V(S0,sig1) + V0

The time component is additionally split via the BS PDE identity
theta = rV - (r-q)S delta - 0.5 sigma^2 S^2 gamma: `time_funding` is the
first two terms x dt (financing the option premium and carrying the delta
hedge), so `time - time_funding` is the pure gamma theta. A delta-hedged
book removes `eq_delta` and `time_funding` and keeps everything else.

residual = actual P&L - sum(components): the remaining cross terms (e.g.
rate x time, equity x time pin risk on settlement days). It should be small.

Position logic (daily $1M 100/110 call-spread sale, 365d expiry, intrinsic
settlement) mirrors book.py; daily total P&L matches simulate_book exactly.
"""

import numpy as np
import pandas as pd

from vol_pca.book import EXPIRY_DAYS, NOTIONAL, SPREAD_WIDTH
from vol_pca.pricing import black76
from vol_pca.surface import grid_lookup

_EPS = 1e-6


def _rq(sd, i, ttm):
    """Zero rate and implied dividend yield at horizon ttm (arrays ok)."""
    ttm = np.maximum(ttm, _EPS)
    r = -np.log(sd.discount(i, ttm)) / ttm
    q = r - np.log(sd.forward(i, ttm) / sd.spot[i]) / ttm
    return r, q


def _bs(spot, strike, ttm, sigma, r, q, greeks=False):
    fwd = spot * np.exp((r - q) * ttm)
    df = np.exp(-r * ttm)
    return black76(fwd, strike, ttm, sigma, df, spot=spot, greeks=greeks)


def independent_attribution(sd, notional=NOTIONAL, detail_dates=()):
    """Returns (daily book-level DataFrame, {date: per-leg detail DataFrame})."""
    detail_dates = {np.datetime64(d, "D") for d in detail_dates}
    details = {}

    n = len(sd)
    cap = 2 * n + 2
    strike = np.zeros(cap)
    units = np.zeros(cap)
    expiry = np.zeros(cap, dtype="datetime64[D]")
    created = np.full(cap, -1)
    val_prev = np.zeros(cap)
    alive = np.zeros(cap, dtype=bool)
    n_legs = 0

    daily = []
    for t in range(n):
        d, s = sd.dates[t], sd.spot[t]
        grid = sd.grids[t]

        if t > 0:
            d0, s0 = sd.dates[t - 1], sd.spot[t - 1]
            grid0 = sd.grids[t - 1]
            ds = s - s0
            live = np.flatnonzero(alive[:n_legs] & (created[:n_legs] < t))

            k, u, exp_d = strike[live], units[live], expiry[live]
            ttm0 = (exp_d - d0).astype(float) / 365.0
            ttm1 = np.maximum((exp_d - d).astype(float) / 365.0, 0.0)
            settled = ttm1 <= 0

            sig0 = grid_lookup(grid0, ttm0, k / s0 * 100.0)
            r0, q0 = _rq(sd, t - 1, ttm0)
            v0, delta0, gamma0, _, _ = _bs(s0, k, ttm0, sig0, r0, q0, greeks=True)

            # new values of each input (rate/div at the OLD ttm for independence)
            sig1 = np.where(settled, sig0,
                            grid_lookup(grid, np.maximum(ttm1, _EPS), k / s * 100.0))
            r1, q1 = _rq(sd, t, ttm0)

            # actual end-of-day mark (matches book.py)
            r1a, q1a = _rq(sd, t, ttm1)
            v1 = np.where(settled, np.maximum(s - k, 0.0),
                          _bs(s, k, ttm1, sig1, r1a, q1a))

            # independent single-input revaluations
            v_s = _bs(s, k, ttm0, sig0, r0, q0)
            v_sig = _bs(s0, k, ttm0, sig1, r0, q0)
            v_r = _bs(s0, k, ttm0, sig0, r1, q0)
            v_q = _bs(s0, k, ttm0, sig0, r0, q1)
            v_t = _bs(s0, k, ttm1, sig0, r0, q0)
            v_ssig = _bs(s, k, ttm0, sig1, r0, q0)

            eq = u * (v_s - v0)
            eq_delta = u * delta0 * ds
            eq_gamma = 0.5 * u * gamma0 * ds**2
            vol = u * (v_sig - v0)
            rate = u * (v_r - v0)
            div = u * (v_q - v0)
            time = u * (v_t - v0)
            vanna = u * ((v_ssig - v_s) - (v_sig - v0))
            pl = u * v1 - val_prev[live]
            resid = pl - eq - vol - rate - div - time - vanna
            # funding part of theta (BS PDE), over the leg's actual decay
            time_funding = u * (r0 * v0 - (r0 - q0) * s0 * delta0) * (ttm0 - ttm1)

            row = {
                "date": d, "spot": s, "gap_days": int((d - d0).astype(int)),
                "n_spreads": int((~settled).sum()) // 2,
                "pl": pl.sum(), "equity": eq.sum(), "eq_delta": eq_delta.sum(),
                "eq_gamma": eq_gamma.sum(),
                "eq_higher": (eq - eq_delta - eq_gamma).sum(),
                "vol": vol.sum(), "rate": rate.sum(), "div": div.sum(),
                "time": time.sum(), "time_funding": time_funding.sum(),
                "vanna": vanna.sum(), "resid": resid.sum(),
            }
            daily.append(row)

            if d in detail_dates:
                details[pd.Timestamp(d)] = pd.DataFrame({
                    "strike": k, "units": u, "expiry": exp_d, "settled": settled,
                    "ttm0": ttm0, "mon0": k / s0 * 100.0, "mon1": k / s * 100.0,
                    "sig0": sig0, "sig1": sig1, "r0": r0, "r1": r1,
                    "q0": q0, "q1": q1, "v0": u * v0, "v1": u * v1,
                    "pl": pl, "equity": eq, "eq_delta": eq_delta,
                    "eq_gamma": eq_gamma, "eq_higher": eq - eq_delta - eq_gamma,
                    "vol": vol, "rate": rate, "div": div, "time": time,
                    "time_funding": time_funding, "vanna": vanna, "resid": resid,
                })

            val_prev[live] = u * v1
            alive[live[settled]] = False

        exp_new = d + np.timedelta64(EXPIRY_DAYS, "D")
        for k_new, u_new in ((s, -notional / s), (SPREAD_WIDTH * s, notional / s)):
            ttm_new = EXPIRY_DAYS / 365.0
            sig_new = grid_lookup(grid, ttm_new, k_new / s * 100.0)
            r_new, q_new = _rq(sd, t, ttm_new)
            v_new = _bs(s, k_new, ttm_new, sig_new, r_new, q_new)
            strike[n_legs], units[n_legs], expiry[n_legs] = k_new, u_new, exp_new
            created[n_legs], val_prev[n_legs], alive[n_legs] = t, u_new * v_new[0], True
            n_legs += 1

    return pd.DataFrame(daily).set_index("date"), details
