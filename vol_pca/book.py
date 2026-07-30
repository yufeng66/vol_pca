"""Daily call-spread book simulation with P&L attribution.

Every trading day the book sells one $1M-notional call spread (short the 100%
strike, long the 110% strike) expiring 365 calendar days later, so at steady
state it carries ~250 spreads (~500 option legs). Positions are marked on the
pillar-grid surface; a leg expiring on or before the current date settles at
intrinsic on that date's spot.

Daily P&L per leg is decomposed sequentially against the previous date's
state (sticky-TTM conventions):
  theta      reprice at the new TTM on the OLD surface/curves
  delta      old smile delta x dS; the surface is quoted in moneyness (% of
             spot), so a fixed strike slides along the smile as spot moves:
             smile delta = BS delta + vega x dsigma/dS from the slide
  gamma      0.5 x old gamma x dS^2
  vega_full  reprice with the NEW date's vol (looked up at the old TTM and
             old moneyness) on the old curves — the "full calculation" of the
             vol-move impact per option
  vanna      old d2V/dSdsigma x dS x dsigma (spot-vol cross term; SPX spot
             and vol moves are strongly anticorrelated so it has drift)
  residual   everything else (higher-order terms, curve moves, wing clamps)

vega_lin (old vega x vol change at the same lookup point) equals
bucket_vega . dsigma exactly because the grid lookup is linear in pillar vols.
"""

import numpy as np
import pandas as pd

from vol_pca.pricing import black76
from vol_pca.surface import N_PILLARS, grid_lookup

NOTIONAL = 1_000_000.0
SPREAD_WIDTH = 1.10          # long strike as multiple of spot
EXPIRY_DAYS = 365


def simulate_book(sd, notional=NOTIONAL):
    n = len(sd)
    cap = 2 * n + 2
    strike = np.zeros(cap)
    units = np.zeros(cap)               # signed: short leg negative
    expiry = np.zeros(cap, dtype="datetime64[D]")
    created = np.full(cap, -1)
    val_prev = np.zeros(cap)            # mark at the previous close
    alive = np.zeros(cap, dtype=bool)
    n_legs = 0

    daily = []
    bucket_vega = np.zeros((n, N_PILLARS))   # start-of-period exposures, row t = move t-1 -> t
    dsigma = np.zeros((n, N_PILLARS))

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
            mon0 = k / s0 * 100.0

            f0, df0 = sd.forward(t - 1, ttm0), sd.discount(t - 1, ttm0)
            sig0, w_idx, w = grid_lookup(grid0, ttm0, mon0, want_weights=True)
            v0, delta0, gamma0, vega0, vanna0 = black76(f0, k, ttm0, sig0, df0, spot=s0, greeks=True)

            # smile slide: dm/dS = -m/S at fixed strike
            dm = 1.0
            dsig_dm = (grid_lookup(grid0, ttm0, mon0 + dm) -
                       grid_lookup(grid0, ttm0, mon0 - dm)) / (2 * dm)
            smile_delta = delta0 + vega0 * dsig_dm * (-mon0 / s0)

            # theta: roll TTM forward on the old surface and old curves
            f0r, df0r = sd.forward(t - 1, ttm1), sd.discount(t - 1, ttm1)
            sig_r = grid_lookup(grid0, np.maximum(ttm1, 1e-6), mon0)
            v_time = black76(f0r, k, ttm1, np.where(ttm1 > 0, sig_r, 0.0), df0r)

            # vega: new date's vols at the old lookup point, all else unchanged
            sig1 = grid_lookup(grid, ttm0, mon0)
            v_vega = black76(f0, k, ttm0, sig1, df0)

            # end-of-day marks: settle expired legs at intrinsic on today's spot
            settled = ttm1 <= 0
            v1 = np.where(settled, np.maximum(s - k, 0.0), 0.0)
            live_idx = ~settled
            if live_idx.any():
                ttm_l, k_l = ttm1[live_idx], k[live_idx]
                sig_l = grid_lookup(grid, ttm_l, k_l / s * 100.0)
                v1[live_idx] = black76(sd.forward(t, ttm_l), k_l, ttm_l, sig_l, sd.discount(t, ttm_l))

            pl = u * v1 - val_prev[live]
            pl_theta = u * (v_time - v0)
            pl_delta = u * smile_delta * ds
            pl_gamma = 0.5 * u * gamma0 * ds**2
            pl_vega = u * (v_vega - v0)
            pl_vega_lin = u * vega0 * (sig1 - sig0)
            pl_vanna = u * vanna0 * ds * (sig1 - sig0)

            np.add.at(bucket_vega[t], w_idx.ravel(), (w * (u * vega0)[:, None]).ravel())
            dsigma[t] = (grid - grid0).ravel()

            val_prev[live] = u * v1
            alive[live[settled]] = False

            daily.append({
                "date": d, "spot": s, "gap_days": int((d - d0).astype(int)),
                "n_spreads": int(live_idx.sum()) // 2,
                "book_value": float(val_prev[:n_legs][alive[:n_legs]].sum()),
                "pl": pl.sum(), "pl_theta": pl_theta.sum(), "pl_delta": pl_delta.sum(),
                "pl_gamma": pl_gamma.sum(), "pl_vega_full": pl_vega.sum(),
                "pl_vanna": pl_vanna.sum(), "pl_vega_lin": pl_vega_lin.sum(),
                "pl_resid": (pl - pl_theta - pl_delta - pl_gamma - pl_vega - pl_vanna).sum(),
            })

        # sell today's spread: short 100% call, long 110% call
        exp_new = d + np.timedelta64(EXPIRY_DAYS, "D")
        for k_new, u_new in ((s, -notional / s), (SPREAD_WIDTH * s, notional / s)):
            ttm_new = EXPIRY_DAYS / 365.0
            sig_new = grid_lookup(grid, ttm_new, k_new / s * 100.0)
            v_new = black76(sd.forward(t, ttm_new), k_new, ttm_new, sig_new, sd.discount(t, ttm_new))
            strike[n_legs], units[n_legs], expiry[n_legs] = k_new, u_new, exp_new
            created[n_legs], val_prev[n_legs], alive[n_legs] = t, u_new * v_new[0], True
            n_legs += 1

    res = pd.DataFrame(daily).set_index("date")
    return res, bucket_vega[1:], dsigma[1:]
