"""Dupire local volatility on the SPX surface, and a RILA-style reset package.

Valuation-only thread (user spec 2026-08-27): no greeks, no VaR, no book
simulation. The product is the insurer's hedge asset behind a registered
index-linked annuity — at issue, basis K = S0 and maturity T = 6y, terminal
payoff on the *final* basis

    (0.8 K - S_T)+  -  (S_T - K)+  +  (S_T - 2K)+

(long the 80% put, short the 100/200 call spread; the net value is usually
NEGATIVE for the hedge-asset side, and is reported signed). The path
dependence is a **ratchet**: at each of the six monthly anniversaries
t_i = i/12 the basis resets K <- max(K, S_{t_i}), so the final basis is
max(S0, S_{1/12}, ..., S_{1/2}) and all three strikes move with it. The
payoff is continuous (kinks only, no digitals), which is what makes it a
good QMC / common-random-number target.

Design notes that the rest of the module hangs off:

- **The pillar grid is not enough.** vol_pca.surface's 8 TTM pillars top out
  at 1.0y; a 6y product needs the whole quoted term structure, so the
  calibration input is built straight from the cleaned raw rows
  (`load_term_surfaces` -> `TermSurface`, ~17 terms per date out to ~9.4y).
  Moneyness interpolation stays the house cubic spline over the 13 quoted
  columns.
- **Total variance is interpolated at fixed log-forward-moneyness y**, which
  is the variable Dupire's calendar derivative is taken in: for each y the
  per-term moneyness m_j = 100 e^y F(tau_j)/S is looked up on that term's
  smile, w_j = sigma_j^2 tau_j, and w(., y) is a shape-preserving PCHIP
  through those knots plus the exact w(0, y) = 0 anchor. PCHIP (not
  not-a-knot cubic) because a cubic overshoots between the 0 anchor and the
  first ~1-month term and manufactures d_tau w < 0 — calendar arbitrage the
  interpolant invented rather than the data. `tau_interp="cubic"` keeps the
  house bicubic convention available for comparison.
- **Wings.** The 200%-of-basis call strike sits outside the quoted 50-150
  moneyness range at every maturity, so the wing convention is a real
  pricing input. `wing="clamp"` = flat implied vol in strike beyond the
  edges (the house `grid_lookup` convention, and the spec's stated
  baseline); `wing="taper"` mirrors rainbow.implied_marginal's C1
  edge-slope taper. **The default is the taper, on measurement**: the flat
  clamp leaves a kink in sigma(y) exactly at the 50/150 join, Dupire's
  d_yy w reads that kink as local butterfly arbitrage (denominator < 0),
  the guard floors sigma_loc in a band right where the long-dated put wing
  still has vega, and vanilla repricing degrades from ~0.06-0.12 to
  ~0.36-0.42 vol points with individual 5y/70-moneyness errors near -2
  points. rainbow.py had already documented the same clamp pathology from
  the density side ("teleports a 2-3% Dirac of mass"). Both conventions are
  computed on every date and the package-value gap is the reported wing
  sensitivity.
- **Rates/divs** come from the date's own log-linear discount and forward
  curves, exactly as everywhere else in the repo: the risk-neutral drift is
  d ln F/dt and discounting is off D(tau), so no separate q is needed
  (F(t) = S0 exp(int (r-q))) and E[S_T] = F(T) holds by construction.

Engines, cheapest first:

- `european_package_surface` — the post-window (r = 0) state is three
  European vanillas; priced by direct surface reads through black76, no
  simulation. Doubles as the LV engine's calibration anchor.
- `pde_*` — Crank-Nicolson in log S (fp64, Rannacher-started). Reprices
  vanillas for the calibration test, prices the frozen-basis package, builds
  the hybrid engine's tail table T(S_w, K), and — because the ratchet only
  bites on six dates — runs the full reset package as a deterministic
  reference (`pde_reset_price`: n_K independent 1-D solves sharing one
  tridiagonal factorisation, coupled only by the K-jump at each fixing).
- `price_rila_mc` — plain pseudo-random log-Euler, the CPU baseline whose
  convergence the study measures. `cv="frozen"` adds the frozen-basis
  control variate anchored on the PDE mean (legitimate here: the repo's
  earlier negative CV verdict was specific to *scrambled Sobol*, where the
  control only adds noise; under pseudo-MC it is textbook).
- `flat_reset_oracle` — closed-machinery check: under constant (sigma, r, q)
  the reset value satisfies a 1-D backward recursion in x = S/K.

No torch at module scope: this file imports numpy/scipy only. The GPU rungs
live in vol_pca/rila_torch.py.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import solve_banded

from vol_pca.data import load_clean_frame
from vol_pca.pricing import black76
from vol_pca.surface import MONEYNESS

# ---------------------------------------------------------------- product

T_RILA = 6.0                 # years, ACT/365
N_RESET = 6                  # monthly ratchet dates
RESET_STEP = 1.0 / 12.0
K_PUT, K_CALL_LO, K_CALL_HI = 0.8, 1.0, 2.0
NOTIONAL = 1_000_000.0

_MON_COLS = [str(int(m)) for m in MONEYNESS]


def package_payoff(s, k):
    """Terminal payoff on spot `s` with final basis `k` (both absolute)."""
    return (np.maximum(K_PUT * k - s, 0.0)
            - np.maximum(s - K_CALL_LO * k, 0.0)
            + np.maximum(s - K_CALL_HI * k, 0.0))


@dataclass(frozen=True)
class RilaState:
    """A valuation state of the package as seen from the calibration date.

    basis: current strike basis K (absolute); spot: current S; ttm: remaining
    time to maturity; fixings: remaining ratchet times in years from now,
    ascending and strictly inside (0, ttm). Newly issued = basis == spot,
    ttm == 6, six fixings; post-window = no fixings left (three vanillas).
    """
    basis: float
    spot: float
    ttm: float
    fixings: tuple = ()

    @property
    def resets_left(self):
        return len(self.fixings)

    @property
    def k_over_s(self):
        return self.basis / self.spot

    @property
    def window_end(self):
        return self.fixings[-1] if self.fixings else 0.0


def rila_state(ts, resets_left=N_RESET, k_over_s=1.0, ttm=None):
    """Seasoned state on the original monthly schedule: `resets_left` fixings
    still ahead means (N_RESET - resets_left) months have elapsed, so the
    remaining fixings sit at 1/12 .. resets_left/12 and the remaining TTM is
    6y minus the elapsed months. `ttm` overrides that (post-window states,
    where the elapsed time is no longer pinned by the fixing count)."""
    elapsed = (N_RESET - resets_left) * RESET_STEP
    if ttm is None:
        ttm = T_RILA - elapsed
    fix = tuple(RESET_STEP * (i + 1) for i in range(resets_left))
    return RilaState(basis=k_over_s * ts.spot, spot=float(ts.spot),
                     ttm=float(ttm), fixings=fix)


# ------------------------------------------------------- implied surface

@dataclass
class TermSurface:
    """One date's full quoted term structure (not the 1y pillar grid)."""
    date: np.datetime64
    spot: float
    taus: np.ndarray            # (n_term,) ACT/365
    vols: np.ndarray            # (n_term, 13) decimals
    curve_ttm: np.ndarray       # (n_term+1,) incl. leading 0
    curve_lndf: np.ndarray
    curve_lnfr: np.ndarray
    mon: np.ndarray = field(default_factory=lambda: MONEYNESS.copy())
    _spl: object = field(default=None, repr=False, compare=False)

    # --- curves (log-linear, the repo convention)
    def discount(self, tau):
        return np.exp(np.interp(tau, self.curve_ttm, self.curve_lndf))

    def fwd_ratio(self, tau):
        return np.exp(np.interp(tau, self.curve_ttm, self.curve_lnfr))

    def forward(self, tau):
        return self.spot * self.fwd_ratio(tau)

    # --- smile
    @property
    def splines(self):
        if self._spl is None:
            object.__setattr__(self, "_spl",
                               CubicSpline(self.mon, self.vols, axis=1))
        return self._spl

    def smile_vol(self, j, mon, wing="taper"):
        """Term j's smile at moneyness `mon` (% of spot), wings per `wing`."""
        lo, hi = self.mon[0], self.mon[-1]
        spl, row = self.splines, self.vols[j]
        mon = np.asarray(mon, dtype=float)
        v = spl(np.clip(mon, lo, hi))[j]
        if wing == "clamp":
            return v
        if wing != "taper":
            raise ValueError(f"unknown wing {wing!r}")
        s_lo = spl(lo, 1)[j]
        s_hi = spl(hi, 1)[j]
        lam = 50.0                                   # rainbow.py taper scale
        v = np.where(mon < lo, row[0] + s_lo * lam * np.expm1(-(lo - mon) / lam), v)
        v = np.where(mon > hi, row[-1] - s_hi * lam * np.expm1(-(mon - hi) / lam), v)
        return np.maximum(v, 0.25 * min(row[0], row[-1]))

    # --- total variance in (tau, y)
    def total_var(self, taus, ys, wing="taper", tau_interp="pchip"):
        """w(tau, y) = sigma_imp^2 tau at y = log(K / F(tau)), shape
        (n_tau, n_y). Per y the quoted terms give exact knots (the query
        moneyness m_j = 100 e^y F(tau_j)/S is read off term j's own smile);
        the tau interpolation runs on total variance through those knots plus
        the exact w(0, y) = 0 anchor."""
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        fr = self.fwd_ratio(self.taus)                       # (n_term,)
        knots = np.empty((len(self.taus) + 1, len(ys)))
        knots[0] = 0.0
        for j in range(len(self.taus)):
            mon = 100.0 * np.exp(ys) * fr[j]
            knots[j + 1] = self.smile_vol(j, mon, wing) ** 2 * self.taus[j]
        nodes = np.concatenate([[0.0], self.taus])
        cls = {"pchip": PchipInterpolator, "cubic": CubicSpline}[tau_interp]
        return cls(nodes, knots, axis=0)(taus)

    def implied_vol(self, taus, ys, **kw):
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        return np.sqrt(self.total_var(taus, ys, **kw) / taus[:, None])


def load_term_surfaces(csv_path, dates=None):
    """Per-date TermSurface off the cleaned raw rows (no pillar collapse)."""
    df, _, _ = load_clean_frame(csv_path)
    if dates is not None:
        want = pd.to_datetime(list(dates))
        df = df[df["asof"].isin(want)]
    out = {}
    for asof, day in df.groupby("asof", sort=True):
        s = float(day["ImpliedSpot"].iloc[0])
        taus = day["ttm"].to_numpy()
        out[str(np.datetime64(asof, "D"))] = TermSurface(
            date=np.datetime64(asof, "D"), spot=s, taus=taus,
            vols=day[_MON_COLS].to_numpy(dtype=float) / 100.0,
            curve_ttm=np.concatenate([[0.0], taus]),
            curve_lndf=np.concatenate([[0.0], np.log(day["DiscountFactor"].to_numpy())]),
            curve_lnfr=np.concatenate([[0.0], np.log(day["Forward"].to_numpy() / s)]))
    return out


def flat_term_surface(vol=0.20, spot=100.0, rate=0.0, div=0.0, n_term=12,
                      tau_max=9.0):
    """Synthetic flat-vol surface with constant r, q — the oracle fixture."""
    taus = np.linspace(tau_max / n_term, tau_max, n_term)
    ct = np.concatenate([[0.0], taus])
    return TermSurface(date=np.datetime64("2024-01-01"), spot=spot, taus=taus,
                       vols=np.full((n_term, len(MONEYNESS)), float(vol)),
                       curve_ttm=ct, curve_lndf=-rate * ct,
                       curve_lnfr=(rate - div) * ct)


# ----------------------------------------------------------- Dupire step

@dataclass
class LocalVolSurface:
    """sigma_loc on a dense (tau, y) grid + the curves it was built with."""
    taus: np.ndarray            # (nt,) ascending
    ys: np.ndarray              # (ny,) uniform
    sig: np.ndarray             # (nt, ny) local vol
    w: np.ndarray               # (nt, ny) total implied variance (diagnostic)
    spot: float
    curve_ttm: np.ndarray
    curve_lndf: np.ndarray
    curve_lnfr: np.ndarray
    diag: dict = field(default_factory=dict)
    date: str = ""

    # --- curves
    def ln_fwd(self, t):
        return np.log(self.spot) + np.interp(t, self.curve_ttm, self.curve_lnfr)

    def ln_df(self, t):
        return np.interp(t, self.curve_ttm, self.curve_lndf)

    def discount(self, t):
        return np.exp(self.ln_df(t))

    def fwd(self, t):
        return np.exp(self.ln_fwd(t))

    def atm_vol(self, tau):
        """sqrt(w(tau, 0)/tau) off the calibration grid."""
        j = int(np.argmin(np.abs(self.ys)))
        return float(np.sqrt(np.interp(tau, self.taus, self.w[:, j])
                             / max(float(np.atleast_1d(tau)[0]), 1e-12)))

    # --- lookups
    def sigma_row(self, t):
        """(ny,) local-vol slice at calendar time t, clamped at the tau ends."""
        tc = float(np.clip(t, self.taus[0], self.taus[-1]))
        i = int(np.clip(np.searchsorted(self.taus, tc) - 1, 0, len(self.taus) - 2))
        f = (tc - self.taus[i]) / (self.taus[i + 1] - self.taus[i])
        return self.sig[i] * (1.0 - f) + self.sig[i + 1] * f

    def sigma_at(self, t, s, row=None):
        """sigma_loc(t, S) for absolute spot(s) `s`, bilinear + edge clamp."""
        row = self.sigma_row(t) if row is None else row
        y = np.log(np.asarray(s, dtype=float)) - self.ln_fwd(t)
        dy = self.ys[1] - self.ys[0]
        u = np.clip((y - self.ys[0]) / dy, 0.0, len(self.ys) - 1.0000001)
        i = u.astype(np.intp)
        f = u - i
        return row[i] * (1.0 - f) + row[i + 1] * f


def dupire_local_vol(ts, n_tau=120, n_y=141, tau_min=1.0 / 365.0, tau_max=6.05,
                     y_lim=3.0, floor=0.05, cap_mult=4.0, wing="taper",
                     tau_interp="pchip"):
    """Gatheral total-variance form of Dupire, by finite differences of the
    smooth w on a dense (tau, y) grid:

        sig_loc^2 = d_tau w / [1 - (y/w) d_y w
                               + 1/4 (-1/4 - 1/w + y^2/w^2) (d_y w)^2
                               + 1/2 d_yy w]

    Local variance is floored at `floor`^2 and local vol capped at
    `cap_mult` x that maturity's ATM implied vol; `diag` carries the node
    counts that hit each guard, plus the raw calendar/butterfly violations
    (d_tau w <= 0, denominator <= 0) so a crash surface's arbitrage shows up
    in the report instead of being silently absorbed.
    """
    taus = np.geomspace(tau_min, tau_max, n_tau)     # short end resolved
    ys = np.linspace(-y_lim, y_lim, n_y)
    w = ts.total_var(taus, ys, wing=wing, tau_interp=tau_interp)

    h = ys[1] - ys[0]
    dwt = np.gradient(w, taus, axis=0)
    dwy = np.gradient(w, h, axis=1)
    d2wy = np.empty_like(w)
    d2wy[:, 1:-1] = (w[:, 2:] - 2.0 * w[:, 1:-1] + w[:, :-2]) / h**2
    d2wy[:, 0], d2wy[:, -1] = d2wy[:, 1], d2wy[:, -2]

    ws = np.maximum(w, 1e-10)
    Y = ys[None, :]
    den = (1.0 - Y / ws * dwy
           + 0.25 * (-0.25 - 1.0 / ws + Y**2 / ws**2) * dwy**2
           + 0.5 * d2wy)

    bad_cal = dwt <= 0.0                      # calendar arbitrage in the data
    bad_bfly = den <= 0.0                     # butterfly arbitrage
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(bad_cal | bad_bfly, -1.0, dwt / den)
    n_floor = int((var < floor**2).sum())
    sig = np.sqrt(np.maximum(var, floor**2))

    j0 = int(np.argmin(np.abs(ys)))
    cap = cap_mult * np.sqrt(w[:, j0] / taus)[:, None]
    n_cap = int((sig > cap).sum())
    sig = np.minimum(sig, cap)

    diag = {"n_nodes": sig.size, "n_floor": n_floor, "n_cap": n_cap,
            "n_calendar_arb": int(bad_cal.sum()), "n_bfly_arb": int(bad_bfly.sum()),
            "pct_floor": 100.0 * n_floor / sig.size,
            "pct_cap": 100.0 * n_cap / sig.size,
            "sig_min": float(sig.min()), "sig_max": float(sig.max()),
            "wing": wing, "tau_interp": tau_interp}
    return LocalVolSurface(taus=taus, ys=ys, sig=sig, w=w, spot=float(ts.spot),
                           curve_ttm=ts.curve_ttm, curve_lndf=ts.curve_lndf,
                           curve_lnfr=ts.curve_lnfr, diag=diag,
                           date=str(ts.date))


def flat_local_vol(vol=0.20, spot=100.0, rate=0.0, div=0.0, tau_max=6.05,
                   n_tau=40, n_y=41, y_lim=3.0):
    """Constant-sigma LV surface with constant r, q (test fixture)."""
    taus = np.geomspace(1.0 / 365.0, tau_max, n_tau)
    ys = np.linspace(-y_lim, y_lim, n_y)
    ct = np.array([0.0, tau_max * 2.0])
    return LocalVolSurface(
        taus=taus, ys=ys, sig=np.full((n_tau, n_y), float(vol)),
        w=np.outer(taus, np.ones(n_y)) * vol**2, spot=float(spot),
        curve_ttm=ct, curve_lndf=-rate * ct, curve_lnfr=(rate - div) * ct,
        diag={"n_floor": 0, "n_cap": 0}, date="flat")


# ------------------------------------------------------ European anchors

def european_package_surface(ts, basis, ttm, wing="taper", tau_interp="pchip"):
    """Post-window value: three vanillas read straight off the implied
    surface (no simulation, no local vol). Returns the signed package value
    and its legs, all in absolute currency for one contract."""
    f = float(ts.forward(ttm))
    d = float(ts.discount(ttm))
    strikes = np.array([K_PUT, K_CALL_LO, K_CALL_HI]) * basis
    sig = ts.implied_vol([ttm], np.log(strikes / f), wing=wing,
                         tau_interp=tau_interp)[0]
    call = black76(f, strikes, ttm, sig, d)
    put = call[0] - d * (f - strikes[0])              # put-call parity
    return {"pv": float(put - call[1] + call[2]), "put_80": float(put),
            "call_100": float(-call[1]), "call_200": float(call[2]),
            "vols": sig, "fwd": f, "df": d}


# ---------------------------------------------------------- CN PDE engine

def _x_grid(lv, s_center, t_hi, strikes, n_x=801, n_std=6.0, pad=0.55):
    """Uniform log-spot grid, centred on ln(s_center) (odd n_x puts it on a
    node) and wide enough for `n_std` diffusive std devs, the forward drift
    and every strike kink."""
    sig = max(lv.atm_vol(t_hi), 0.05)
    hw = n_std * sig * np.sqrt(t_hi) + abs(lv.ln_fwd(t_hi) - lv.ln_fwd(0.0)) + pad
    if len(strikes):
        hw = max(hw, np.max(np.abs(np.log(np.asarray(strikes, float) / s_center))) + pad)
    n_x = n_x if n_x % 2 else n_x + 1
    return np.linspace(np.log(s_center) - hw, np.log(s_center) + hw, n_x)


def _pde_times(t_hi, t_lo, per_year):
    n = max(2, int(round((t_hi - t_lo) * per_year)))
    return np.linspace(t_hi, t_lo, n + 1)


def cn_rollback(lv, v, x, times, rannacher=2):
    """Crank-Nicolson backward roll of `v` (n_x, m) along the decreasing
    `times`.

    ln S coordinates: d_t V + (mu - sig^2/2) d_x V + (sig^2/2) d_xx V - r V = 0,
    with mu and r the step-average forward drift / short rate implied by the
    date's own curves. The first `rannacher` steps run fully implicit to damp
    the payoff kinks (plain CN rings on them). The tridiagonal operator does
    not depend on the right-hand-side column, so a whole family of terminal
    conditions (strikes, or basis nodes K) rolls back in one banded solve per
    step — that is what makes the tail table and the reset reference cheap.

    Boundaries use the **linearity condition** (V affine in S at both ends,
    imposed through the three outermost nodes so the system stays
    tridiagonal). That is exact for every leg of this study — a call is
    affine in S at both extremes, and so is the package, whose S -> inf limit
    is the constant -K with the basis frozen but -S_ratchet once a fixing can
    still fire. Hard-coding the frozen-basis Dirichlet value -K*D instead
    biases the reset reference by ~1bp of spot; the linearity condition needs
    no case analysis and lands on the flat-vol oracle.
    """
    v = np.array(v, dtype=float, copy=True)
    if v.ndim == 1:
        v = v[:, None]
    dx = x[1] - x[0]
    s = np.exp(x)
    n = len(x)
    # V_0 = a0 V_1 + b0 V_2 ;  V_{n-1} = a1 V_{n-2} + b1 V_{n-3}
    b0 = (s[0] - s[1]) / (s[2] - s[1])
    a0 = 1.0 - b0
    rho = (s[-1] - s[-2]) / (s[-2] - s[-3])
    a1, b1 = 1.0 + rho, -rho
    ab = np.empty((3, n - 2))
    for k in range(len(times) - 1):
        t1, t0 = times[k], times[k + 1]
        dt = t1 - t0
        sig = lv.sigma_at(0.5 * (t0 + t1), s)
        mu = (lv.ln_fwd(t1) - lv.ln_fwd(t0)) / dt
        r = -(lv.ln_df(t1) - lv.ln_df(t0)) / dt
        a = 0.5 * sig**2
        b = mu - a
        lo = a / dx**2 - b / (2.0 * dx)
        di = -2.0 * a / dx**2 - r
        up = a / dx**2 + b / (2.0 * dx)
        th = 1.0 if k < rannacher else 0.5

        # explicit part on the old solution (its boundary rows are already set)
        rhs = v[1:-1] + (1.0 - th) * dt * (
            lo[1:-1, None] * v[:-2] + di[1:-1, None] * v[1:-1] + up[1:-1, None] * v[2:])

        ab[0, 1:] = -th * dt * up[1:-2]        # super-diagonal
        ab[1, :] = 1.0 - th * dt * di[1:-1]
        ab[2, :-1] = -th * dt * lo[2:-1]       # sub-diagonal
        # fold the linearity conditions into the two end rows
        ab[1, 0] -= th * dt * lo[1] * a0
        ab[0, 1] -= th * dt * lo[1] * b0
        ab[1, -1] -= th * dt * up[-2] * a1
        ab[2, -2] -= th * dt * up[-2] * b1
        v[1:-1] = solve_banded((1, 1), ab, rhs)
        v[0] = a0 * v[1] + b0 * v[2]
        v[-1] = a1 * v[-2] + b1 * v[-3]
    return v


def pde_package(lv, basis, ttm, t_end=0.0, n_x=801, per_year=120,
                rannacher=2, spot=None, full=False):
    """European (frozen-basis) package value under the LV surface.

    `basis` may be an array of K nodes: they share the tridiagonal solve, so
    the tail table T(S_w, K) of the hybrid engine is this call with
    t_end = the window end and `full=True` (returns the whole (n_x, n_K)
    slice plus the x grid instead of the value at `spot`).
    """
    k = np.atleast_1d(np.asarray(basis, dtype=float))
    s0 = float(lv.spot if spot is None else spot)
    x = _x_grid(lv, s0, ttm, np.concatenate([K_PUT * k, K_CALL_HI * k]), n_x)
    v = package_payoff(np.exp(x)[:, None], k[None, :])
    v = cn_rollback(lv, v, x, _pde_times(ttm, t_end, per_year), rannacher)
    if full:
        return {"x": x, "k": k, "v": v}
    j = int(np.argmin(np.abs(x - np.log(s0))))
    return float(v[j, 0]) if k.size == 1 else v[j]


def pde_vanilla_calls(lv, ttm, strikes, n_x=801, per_year=120, rannacher=2,
                      spot=None):
    """Call prices at one maturity, all strikes in one banded solve."""
    k = np.atleast_1d(np.asarray(strikes, dtype=float))
    s0 = float(lv.spot if spot is None else spot)
    x = _x_grid(lv, s0, ttm, k, n_x)
    s = np.exp(x)
    v = np.maximum(s[:, None] - k[None, :], 0.0)
    v = cn_rollback(lv, v, x, _pde_times(ttm, 0.0, per_year), rannacher)
    j = int(np.argmin(np.abs(x - np.log(s0))))
    return v[j]


def pde_reprice_table(lv, ts, taus=None, mons=(70, 80, 90, 100, 110, 120, 130),
                      n_x=801, per_year=120, wing="taper", tau_interp="pchip"):
    """Calibration test: reprice vanillas on the LV surface and convert the
    price error back into implied-vol points against the input surface."""
    if taus is None:
        taus = ts.taus[ts.taus <= T_RILA]
    rows = []
    for tau in np.atleast_1d(taus):
        f, d = float(ts.forward(tau)), float(ts.discount(tau))
        strikes = np.asarray(mons, dtype=float) / 100.0 * ts.spot
        target = ts.implied_vol([tau], np.log(strikes / f), wing=wing,
                                tau_interp=tau_interp)[0]
        pde = pde_vanilla_calls(lv, float(tau), strikes, n_x=n_x, per_year=per_year)
        ref = black76(f, strikes, tau, target, d)
        vega = d * f * np.exp(-0.5 * ((np.log(f / strikes) + 0.5 * target**2 * tau)
                                      / (target * np.sqrt(tau)))**2) \
            / np.sqrt(2 * np.pi) * np.sqrt(tau)
        rows.append(pd.DataFrame({
            "tau": float(tau), "mon": np.asarray(mons, dtype=float),
            "vol_in": target, "pde": pde, "black": ref,
            "d_vol_pts": 100.0 * (pde - ref) / np.maximum(vega, 1e-12)}))
    return pd.concat(rows, ignore_index=True)


def pde_reset_price(lv, state, n_k=97, k_hi_mult=2.4, n_x=801, per_year=120,
                    rannacher=2):
    """Deterministic reference for the *reset* package.

    The ratchet only fires on the six fixing dates, so between them K is a
    plain parameter and the problem is n_K independent 1-D PDEs sharing one
    operator. At a fixing the coupling is a pure relabelling,
    V(t^-, S, K) = V(t^+, S, max(K, S)), i.e. a gather along the (log-spaced)
    K axis — no diffusion in K, no 2-D solve. `tail=True` stops at the window
    end and returns the (x, K, V) tail table the hybrid MC engine integrates
    against.
    """
    k = np.geomspace(state.basis, k_hi_mult * state.spot, n_k)
    x = _x_grid(lv, state.spot, state.ttm,
                np.concatenate([K_PUT * k, K_CALL_HI * k]), n_x)
    s = np.exp(x)
    v = package_payoff(s[:, None], k[None, :])

    lnk = np.log(k)
    dlnk = lnk[1] - lnk[0]
    stops = list(state.fixings)[::-1] + [0.0]
    t_hi = state.ttm
    for t_lo in stops:
        # every leg gets its own Rannacher start: the ratchet relabelling
        # leaves a kink in S at S = K (the delta jumps by d_K V), so each
        # post-fixing roll restarts from a non-smooth profile just like the
        # terminal payoff does
        v = cn_rollback(lv, v, x, _pde_times(t_hi, t_lo, per_year), rannacher)
        if t_lo == 0.0:
            break
        # ratchet: relabel to the new basis max(K, S) — locate the target on
        # the log-spaced K axis but interpolate LINEARLY IN K, and let the
        # top node extrapolate instead of clamping: for S well above the K
        # range the package value is affine in K (slope -D with the basis
        # frozen, ~0 while a later fixing can still overwrite it), so the
        # local end slope carries S > k_max correctly. Clamping there would
        # value a 10-sigma node at -k_max instead of ~-S.
        tgt = np.maximum(k[None, :], s[:, None])
        u = np.clip((np.log(tgt) - lnk[0]) / dlnk, 0.0, n_k - 2.0)
        i = u.astype(np.intp)
        k0 = k[i]
        f = (tgt - k0) / (k[i + 1] - k0)
        v = np.take_along_axis(v, i, 1) * (1 - f) + np.take_along_axis(v, i + 1, 1) * f
        t_hi = t_lo
    j = int(np.argmin(np.abs(x - np.log(state.spot))))
    return {"pv": float(v[j, 0]), "x": x, "k": k, "v": v,
            "extrap": float((s > k[-1]).mean())}


def pde_tail_table(lv, state, n_k=97, k_hi_mult=2.4, n_x=801, per_year=120,
                   rannacher=2):
    """T(S_w, K) = LV value at the window end of the European package with
    basis K — the hybrid engine's exact 5.5y tail."""
    t_win = state.window_end
    k = np.geomspace(min(state.basis, state.spot), k_hi_mult * state.spot, n_k)
    out = pde_package(lv, k, state.ttm, t_end=t_win, n_x=n_x,
                      per_year=per_year, rannacher=rannacher, spot=state.spot,
                      full=True)
    out["t_window"] = t_win
    return out


# ------------------------------------------------------------- CPU Monte

def mc_time_grid(state, per_year):
    """Step grid that contains every remaining fixing and maturity exactly:
    each inter-fixing segment is subdivided on its own, so the fixings are
    grid nodes by construction rather than by floating-point luck."""
    anchors = np.unique(np.concatenate([[0.0], np.asarray(state.fixings, float),
                                        [state.ttm]]))
    segs = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        n = max(1, int(round((b - a) * per_year)))
        segs.append(np.linspace(a, b, n + 1)[:-1])
    t = np.concatenate(segs + [[state.ttm]])
    fix_idx = np.searchsorted(t, np.asarray(state.fixings, float))
    return t, fix_idx


def mc_step_data(lv, t_grid):
    """Per-step drift increments and local-vol rows, precomputed once."""
    dlnf = np.diff(lv.ln_fwd(t_grid))
    rows = np.stack([lv.sigma_row(t) for t in t_grid[:-1]])
    return dlnf, rows


def simulate_terminal(lv, state, t_grid, fix_idx, n_paths, seed=0,
                      chunk=1 << 18, normals=None, reset=True, step=None):
    """Log-Euler on the fixed grid, returning terminal spot and final basis.

    ln S_{k+1} = ln S_k + (ln F_{k+1} - ln F_k) - sig^2 dt/2 + sig sqrt(dt) Z
    with sig = sigma_loc(t_k, S_k) read bilinearly off the calibration grid.
    The drift increment is the *exact* forward log-increment over the step,
    so S/F stays an exact martingale under the scheme (E[S_T] = F(T) with no
    discretisation bias — the martingale test checks pure MC error).

    `normals` (n_paths, n_steps) forces given draws (CPU/GPU CRN parity).
    """
    dlnf, rows = mc_step_data(lv, t_grid) if step is None else step
    dt = np.diff(t_grid)
    sq = np.sqrt(dt)
    dy = lv.ys[1] - lv.ys[0]
    lnf = lv.ln_fwd(t_grid[:-1])
    ny = len(lv.ys)
    rng = np.random.default_rng(seed)
    s_out, k_out = [], []
    fixset = set(int(i) for i in fix_idx)
    for i0 in range(0, n_paths, chunk):
        m = min(chunk, n_paths - i0)
        lns = np.full(m, np.log(state.spot))
        kk = np.full(m, float(state.basis))
        for step_i in range(len(dt)):
            u = np.clip((lns - lnf[step_i] - lv.ys[0]) / dy, 0.0, ny - 1.0000001)
            j = u.astype(np.intp)
            f = u - j
            row = rows[step_i]
            sig = row[j] * (1.0 - f) + row[j + 1] * f
            z = (rng.standard_normal(m) if normals is None
                 else normals[i0:i0 + m, step_i])
            lns += dlnf[step_i] - 0.5 * sig**2 * dt[step_i] + sig * sq[step_i] * z
            if reset and (step_i + 1) in fixset:
                np.maximum(kk, np.exp(lns), out=kk)
        s_out.append(np.exp(lns))
        k_out.append(kk)
    return np.concatenate(s_out), np.concatenate(k_out)


def price_rila_mc(lv, state, n_paths=1 << 16, per_year=252, seed=0,
                  chunk=1 << 18, cv=None, cv_beta="fit", reset=True,
                  normals=None, step=None, t_grid=None, fix_idx=None,
                  cv_exact=None, pde_kw=None):
    """Plain pseudo-random Monte Carlo value of the package (one contract).

    cv="frozen" turns on the frozen-basis control variate: the same paths'
    payoff with the basis held at K_0, whose exact mean comes from the LV
    PDE. Under pseudo-MC that is a textbook variance reduction — the repo's
    2026-08-06 negative CV verdict was specific to *scrambled Sobol*, where
    the QMC rule has already integrated the smooth part the control shares
    and the correction only injects its own noise.
    """
    if t_grid is None:
        t_grid, fix_idx = mc_time_grid(state, per_year)
    s_t, k_t = simulate_terminal(lv, state, t_grid, fix_idx, n_paths, seed=seed,
                                 chunk=chunk, normals=normals, reset=reset,
                                 step=step)
    d = float(lv.discount(state.ttm))
    pay = package_payoff(s_t, k_t)
    out = {"pv": float(d * pay.mean()), "se": float(d * pay.std(ddof=1) / np.sqrt(n_paths)),
           "n_paths": n_paths, "n_steps": len(t_grid) - 1,
           "mean_s": float(s_t.mean()), "fwd": float(lv.fwd(state.ttm)),
           "p_ratchet": float((k_t > state.basis * (1 + 1e-12)).mean())}
    if cv == "frozen":
        c = package_payoff(s_t, state.basis)
        ec = (pde_package(lv, state.basis, state.ttm, **(pde_kw or {})) / d
              if cv_exact is None else float(np.ravel(cv_exact)[0]) / d)
        beta = (float(np.cov(pay, c)[0, 1] / max(np.var(c), 1e-300))
                if cv_beta == "fit" else float(cv_beta))
        res = pay - beta * (c - ec)
        out["pv_plain"], out["pv"] = out["pv"], float(d * res.mean())
        out["se"] = float(d * res.std(ddof=1) / np.sqrt(n_paths))
        out.update(cv_beta=beta, cv_corr=float(np.corrcoef(pay, c)[0, 1]),
                   cv_exact=float(d * ec))
    elif cv is not None:
        raise ValueError(f"unknown control variate {cv!r}")
    return out


def rila_value(ts, lv, state, engine="auto", **kw):
    """Dispatcher. A post-window state (no fixings left) is three European
    vanillas: it is returned by *direct surface reads*, never simulated."""
    if not state.fixings and engine in ("auto", "surface"):
        out = european_package_surface(ts, state.basis, state.ttm,
                                       wing=kw.pop("wing", "taper"))
        out["engine"] = "surface"
        return out
    if engine in ("auto", "pde"):
        out = pde_reset_price(lv, state, **kw)
        out["engine"] = "pde"
        return out
    if engine == "mc":
        out = price_rila_mc(lv, state, **kw)
        out["engine"] = "mc"
        return out
    raise ValueError(f"unknown engine {engine!r}")


def per_100(value, basis):
    """House quoting unit: value per 100 of the current basis K."""
    return 100.0 * value / basis


# ------------------------------------------------------------ flat oracle

def _flat_package_bs(x, ttm, sigma, rate, div):
    """BS value per unit basis of the frozen-basis package at spot x."""
    x = np.asarray(x, dtype=float)
    f = x * np.exp((rate - div) * ttm)
    d = np.exp(-rate * ttm)
    c08 = black76(f, K_PUT, ttm, sigma, d)
    c10 = black76(f, K_CALL_LO, ttm, sigma, d)
    c20 = black76(f, K_CALL_HI, ttm, sigma, d)
    put = c08 - d * (f - K_PUT)
    return put - c10 + c20


def flat_reset_oracle(sigma, rate, div, ttm, fixings, x0=1.0, n_x=1601,
                      x_lo=0.03, n_gl=64):
    """Exact-machinery oracle: value per unit basis under constant sigma, r, q.

    Just after a fixing the basis is K >= S, so with x = S/K and v = V/K the
    Black-Scholes scale invariance collapses the state to one dimension: the
    ratchet K <- max(K, S g) maps x -> min(x g, 1) and rescales the value by
    max(1, x g), giving the backward recursion

        v_i(x) = df * E_g[ max(1, x g) * v_{i+1}( min(x g, 1) ) ]

    with terminal v_n(x) = the Black-Scholes package value over the tail
    (ttm - last fixing) at spot x, unit basis.

    The expectation is **split at the ratchet kink** x g = 1 (the house
    quadrature discipline): below it the integrand is the smooth
    v_{i+1}(x g), integrated by Gauss-Legendre in copula-uniform space;
    above it min(x g, 1) == 1 so the whole branch collapses to the closed
    form v_{i+1}(1) * x * E[g 1_{g > 1/x}], a lognormal partial expectation.
    Plain Gauss-Hermite over the un-split, unbounded max(1, x g) integrand
    is only algebraically convergent and was still drifting ~1bp of spot at
    128 nodes.

    v lives on a log-spaced x grid on (x_lo, 1] with affine extrapolation
    below (as x -> 0 the package is 0.8 df - x e^{-qT}, exactly affine, so
    the extrapolation is free). x_lo defaults well below the spec's 0.3:
    compounding down-months at the outer quadrature nodes reach ~0.05, and
    clipping them would bias the oracle the tests are supposed to trust.
    """
    from scipy.stats import norm

    fix = np.asarray(fixings, dtype=float)
    n = len(fix)
    x = np.geomspace(x_lo, 1.0, n_x)
    lnx = np.log(x)
    dlnx = lnx[1] - lnx[0]
    gx, gw = np.polynomial.legendre.leggauss(n_gl)
    gx, gw = 0.5 * (gx + 1.0), 0.5 * gw

    v = _flat_package_bs(x, ttm - fix[-1], sigma, rate, div)

    def step(x_from, v_tab, delta):
        """One ratchet step of length `delta` applied to the table v_tab."""
        m = (rate - div - 0.5 * sigma**2) * delta
        s = sigma * np.sqrt(delta)
        xf = np.atleast_1d(np.asarray(x_from, dtype=float))
        z_star = (-np.log(xf) - m) / s                    # x g = 1
        u_star = norm.cdf(z_star)
        u = u_star[:, None] * gx[None, :]
        wq = u_star[:, None] * gw[None, :]
        q = xf[:, None] * np.exp(m + s * norm.ppf(np.clip(u, 1e-300, 1.0)))
        idx = (np.log(q) - lnx[0]) / dlnx
        i = np.clip(idx, 0.0, n_x - 1.0000001).astype(np.intp)
        f = np.clip(idx, 0.0, n_x - 1.0000001) - i
        val = v_tab[i] * (1.0 - f) + v_tab[i + 1] * f
        slope = (v_tab[1] - v_tab[0]) / (x[1] - x[0])     # affine below x_lo
        val = np.where(q < x[0], v_tab[0] + (q - x[0]) * slope, val)
        body = (val * wq).sum(1)
        tail = v_tab[-1] * xf * np.exp(m + 0.5 * s**2) * norm.cdf(s - z_star)
        return np.exp(-rate * delta) * (body + tail)

    gaps = np.diff(np.concatenate([[0.0], fix]))
    for j in range(n - 1, 0, -1):
        v = step(x, v, gaps[j])
    return float(step(np.atleast_1d(x0), v, gaps[0])[0])
