"""Batched torch port of the deterministic rainbow quadrature.

The roadmap heads toward a daily-sold rainbow *book* (hundreds of seasoned
options per as-of date) and cheap-VaR studies whose expensive baseline
revalues that book under thousands of scenarios. Both are the same
computation — many independent quadrature problems with identical
node/grid shapes — so this engine is batched-first: a `QuadBatch` holds B
problems (per-problem marginal tables, correlation cholesky, discount,
performance-to-date factors) and `price_quad_batch` prices all of them in
one call, on CPU or GPU, chunked so the big survival tensor never
outgrows `max_bytes`. A single option is a batch of one
(`price_rainbow_quad_torch`, mirroring vol_pca.rainbow.price_rainbow_quad
argument-for-argument); a book is a batch of ~250; a VaR sweep is a
stream of batches. Two engines share the batch: `price_quad_batch`, the
deterministic quadrature, and `price_sobol_batch`, the scrambled-Sobol
sampler with one resident point set per book slot (common random numbers
across days/components, independent scrambles across slots).
`MarginalFactory` builds the tables on the device too: every date's
pillar grids and curves upload once (~10MB for the whole dataset), and
because the bicubic interpolant has fixed knots, spline evaluation is
linear in the pillar values — the scipy not-a-knot solve is baked into
constant cardinal coefficient tensors at construction, so a marginal
build is matmuls, gathers and elementwise special functions, no CPU
round-trip. The factory's `dgrid` hook (additive pillar-vol shocks) is
the scenario-engine seam for the coming VaR work, and its two-tau split
(lookup TTM vs pricing TTM) is the var.py roll convention.
`rainbow_attribution` on top of both is the daily-sold book's P&L
attribution driver — the rainbow analog of attribution.py's
single-input revaluations, run with the CRN Sobol engine.

Numerics: node placement, quantile/score-table interpolation and the
final reduction run in float64 — those tensors are (B, nodes)-sized, so
precision there is free. The survival matrix and its trapezoid panels,
the (B, nodes, grid) tensors that dominate time and memory, run in
`survival_dtype` (float32 by default: consumer GPUs execute fp64 at 1/64
rate, and the eps-clipped normal scores keep every fp32 intermediate
O(10)); the cumulative integral is then accumulated in float64, which is
where fp32 would actually lose digits. The whole graph is differentiable:
autograd on `QuadBatch.g` yields exact per-problem spot/seasoning deltas
in one backward pass, while vol sensitivities go through rebuilt tables —
the project's bump-and-revalue philosophy. The numpy engine remains the
canonical oracle; tests pin this module against it.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import norm, qmc

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without the gpu group
    raise ImportError(
        "vol_pca.rainbow_torch needs the optional 'gpu' dependency group "
        "(uv sync --group gpu)") from exc

from vol_pca.rainbow import (K_HI, K_LO, NOTIONAL, WEIGHTS, historical_corr,
                             spot_panel)

_EPS = 1e-16                     # cdf clip for normal scores, as in rainbow.py


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class QuadProblem(NamedTuple):
    """One rainbow valuation: three marginals glued by one copula."""
    marginals: tuple             # 3 ImpliedMarginal (equal grid lengths)
    corr: object                 # (3, 3) correlation matrix
    df: float                    # discount factor to expiry
    perf_to_date: tuple = (1.0, 1.0, 1.0)


@dataclass
class QuadBatch:
    """B problems stacked into device tensors (all float64). Shared by both
    engines: the quadrature reads x/cdf/psi, the Sobol sampler x/cdf."""
    x: torch.Tensor              # (B, 3, n) performance grids
    cdf: torch.Tensor            # (B, 3, n) marginal CDFs on x
    psi: torch.Tensor | None     # (B, 3, n) normal scores ndtri(clip(cdf))
    chol: torch.Tensor           # (B, 3, 3) lower cholesky of corr
    g: torch.Tensor              # (B, 3) performance-to-date factors
    df: torch.Tensor             # (B,) discount factors

    def __len__(self):
        return self.x.shape[0]

    @classmethod
    def from_problems(cls, problems, device=None, scores=True):
        """scores=False skips the psi (normal-score) tables — a Sobol-only
        batch doesn't need them and the ppf is the assembly's main cost."""
        device = device or default_device()
        xs, cdfs, chols, gs, dfs = [], [], [], [], []
        for pr in problems:
            xs.append(np.stack([m.x for m in pr.marginals]))
            cdfs.append(np.stack([m.cdf for m in pr.marginals]))
            chols.append(np.linalg.cholesky(np.asarray(pr.corr, dtype=float)))
            gs.append(np.asarray(pr.perf_to_date, dtype=float))
            dfs.append(float(pr.df))
        cdf = np.stack(cdfs)
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        return cls(x=t(np.stack(xs)), cdf=t(cdf),
                   psi=t(norm.ppf(np.clip(cdf, _EPS, 1.0 - _EPS)))
                       if scores else None,
                   chol=t(np.stack(chols)), g=t(np.stack(gs)), df=t(dfs))


def _interp(q, xp, fp):
    """Batched np.interp: q (B, m) into per-row tables xp/fp (B, n).

    xp must be non-decreasing; out-of-range clamps to the end values.
    Exact plateaus in xp (clipped cdf tails) cannot be straddled by the
    right=True bracket, so this matches the numpy engine's deduplicated
    interp everywhere the quadrature weights are non-negligible.
    """
    n = xp.shape[-1]
    i = torch.searchsorted(xp.contiguous(), q.contiguous(),
                           right=True).clamp(1, n - 1)
    x_lo, x_hi = xp.gather(-1, i - 1), xp.gather(-1, i)
    f_lo, f_hi = fp.gather(-1, i - 1), fp.gather(-1, i)
    w = ((q - x_lo) / (x_hi - x_lo).clamp_min(1e-300)).clamp(0.0, 1.0)
    return f_lo + w * (f_hi - f_lo)


def _chunk_integral(xs, dxs, pss, muss, a_c, b_c):
    """One chunk of the conditional survival integral: build P(P3 > x | .)
    on the grid, trapezoid-accumulate in float64, evaluate at both strikes.
    muss/pss arrive pre-divided by s3 and cast to the survival dtype."""
    surv = torch.special.ndtr(muss[:, :, None] - pss)
    cum = torch.cumsum(0.5 * (surv[..., 1:] + surv[..., :-1]) * dxs,
                       -1, dtype=torch.float64)
    cum = torch.cat([cum.new_zeros(*cum.shape[:-1], 1), cum], -1)
    return _cum_at(cum, xs, b_c) - _cum_at(cum, xs, a_c)


def _cum_at(cum, x3, q):
    """Evaluate per-node piecewise-linear cumulatives cum (cB, m, n) on the
    per-problem grid x3 (cB, n) at q (cB, m); clamps outside the grid."""
    n = x3.shape[-1]
    qc = torch.clamp(q, x3[:, :1], x3[:, -1:])
    i = torch.searchsorted(x3.contiguous(), qc.contiguous()).clamp(1, n - 1)
    x_lo, x_hi = x3.gather(-1, i - 1), x3.gather(-1, i)
    t = ((qc - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)
    c0 = cum.gather(-1, (i - 1).unsqueeze(-1)).squeeze(-1)
    c1 = cum.gather(-1, i.unsqueeze(-1)).squeeze(-1)
    return c0 + t * (c1 - c0)


def price_quad_batch(batch, n_outer=48, n_inner=48, weights=WEIGHTS,
                     k_lo=K_LO, k_hi=K_HI, notional=NOTIONAL,
                     survival_dtype=torch.float32, max_bytes=2 << 30):
    """Price every problem in the batch; returns pv as a (B,) float64 tensor
    on the batch's device (differentiable w.r.t. the batch tensors).

    Algorithm is line-for-line vol_pca.rainbow.price_rainbow_quad: HSI
    integrated out analytically conditional on the two copula normals,
    Gauss-Hermite outer x Gauss-Legendre inner split at the rank kink.
    Only the survival/trapezoid stage runs in `survival_dtype`; the
    cumulative integral and everything (B, nodes)-sized stay float64.
    `max_bytes` caps the survival-stage working set per chunk.
    """
    if batch.psi is None:
        raise ValueError("batch was built with scores=False (no psi tables); "
                         "rebuild with scores=True for the quadrature engine")
    x, cdf, psi, L, g, df = (batch.x, batch.cdf, batch.psi, batch.chol,
                             batch.g, batch.df)
    dev = x.device
    B, _, n = x.shape
    w_b, w_w, w3 = weights

    z1_np, w1_np = np.polynomial.hermite_e.hermegauss(n_outer)
    gx_np, gw_np = np.polynomial.legendre.leggauss(n_inner)
    ten = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    z1, w1 = ten(z1_np), ten(w1_np / np.sqrt(2.0 * np.pi))
    gx, gw = ten(0.5 * (gx_np + 1.0)), ten(0.5 * gw_np)

    p1 = g[:, 0:1] * _interp(torch.special.ndtr(z1).expand(B, -1),
                             cdf[:, 0], x[:, 0])                    # (B, no)

    # inner split point: g2 * P2 crosses the (seasoned) p1
    t = _interp(p1 / g[:, 1:2], x[:, 1], psi[:, 1])
    v_star = torch.special.ndtr((t - L[:, 1, 0, None] * z1) / L[:, 1, 1, None])
    v = torch.cat([v_star[..., None] * gx,
                   v_star[..., None] + (1.0 - v_star[..., None]) * gx], -1)
    wv = torch.cat([v_star[..., None] * gw,
                    (1.0 - v_star[..., None]) * gw], -1)     # (B, no, 2ni)

    y2 = torch.special.ndtri(v.clamp(_EPS, 1.0 - _EPS))
    z2 = L[:, 1, 0, None, None] * z1[:, None] + L[:, 1, 1, None, None] * y2
    m = n_outer * 2 * n_inner
    p2 = g[:, 1:2] * _interp(torch.special.ndtr(z2).reshape(B, m),
                             cdf[:, 1], x[:, 1])                    # (B, m)

    p1f = p1.unsqueeze(-1).expand_as(y2).reshape(B, m)
    basket12 = (w_b * torch.maximum(p1f, p2) + w_w * torch.minimum(p1f, p2))
    mu = (L[:, 2, 0, None, None] * z1[:, None]
          + L[:, 2, 1, None, None] * y2).reshape(B, m)
    s3 = L[:, 2, 2].clamp_min(1e-12)

    w3g = w3 * g[:, 2]
    a = (k_lo - basket12) / w3g[:, None]     # call-spread strikes in X3
    b = (k_hi - basket12) / w3g[:, None]
    x3, psi3 = x[:, 2], psi[:, 2]

    # survival-curve integral int_a^b P(P3 > p | z1, y2) dp, chunked so the
    # (cB, cm, n) stage tensors stay under max_bytes. Under autograd each
    # chunk is checkpointed: without that, the graph would pin every chunk's
    # survival tensor for backward and peak memory would scale with B, not
    # with the chunk size.
    bytes_el = 24 if survival_dtype == torch.float32 else 40
    c_b = int(max(1, min(B, max_bytes // (m * n * bytes_el))))
    c_m = m if c_b > 1 else int(max(1, min(m, max_bytes // (n * bytes_el))))
    ckpt = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, cdf, psi, L, g))
    rows = []
    for i0 in range(0, B, c_b):
        sb = slice(i0, i0 + c_b)
        dx = torch.diff(x3[sb]).to(survival_dtype)[:, None, :]
        ps = (psi3[sb] / s3[sb, None]).to(survival_dtype)[:, None, :]
        mus = (mu[sb] / s3[sb, None]).to(survival_dtype)
        cols = []
        for j0 in range(0, m, c_m):
            sm = slice(j0, j0 + c_m)
            args = (x3[sb], dx, ps, mus[:, sm], a[sb, sm], b[sb, sm])
            cols.append(torch.utils.checkpoint.checkpoint(
                _chunk_integral, *args, use_reentrant=False)
                if ckpt else _chunk_integral(*args))
        rows.append(torch.cat(cols, -1))
    integ = torch.cat(rows, 0)

    below = (torch.minimum(b, x3[:, 0:1]) - a).clamp_min(0.0)
    cond = w3g[:, None] * (below + integ)
    node_w = (w1[None, :, None] * wv).reshape(B, m)
    return df * (node_w * cond).sum(-1) * notional


def sobol_normals(n_slots, n_paths, seed=0, device=None):
    """One scrambled Sobol point set per book slot, as standard normals
    (S, N, 3): generated once on the CPU with scipy's generator (the same
    one the numpy engine uses), moved to the device once, then resident.

    The slot design is the common-random-numbers discipline for the book:
    a vintage keeps its slot's points for every day and every component
    revaluation (so differences cancel the sampling error), while slot
    scrambles are independent (spawned seed sequences), so book-level
    errors still average down ~sqrt(slots) instead of adding coherently.
    """
    if n_paths & (n_paths - 1):
        raise ValueError("n_paths must be a power of two for balanced Sobol")
    device = device or default_device()
    k2 = int(np.log2(n_paths))
    u = np.stack([qmc.Sobol(d=3, scramble=True,
                            seed=np.random.default_rng(c)).random_base2(k2)
                  for c in np.random.SeedSequence(seed).spawn(n_slots)])
    return torch.as_tensor(norm.ppf(u), dtype=torch.float64, device=device)


def price_sobol_batch(batch, normals, slots=None, weights=WEIGHTS, k_lo=K_LO,
                      k_hi=K_HI, notional=NOTIONAL, max_bytes=2 << 30):
    """Scrambled-Sobol sampler on the device — the stochastic mirror of
    price_quad_batch over the same QuadBatch tables (psi not needed).
    Mirrors rainbow.price_rainbow/sample_performances step for step:
    correlate the slot's normals with each problem's cholesky, map to
    uniforms, pull each marginal's quantile, season by g, ranked basket,
    capped payoff, discounted mean. Returns a (B,) float64 pv tensor.
    Fully deterministic given `normals`; all-float64 (the path tensors are
    (B, N, 3) — small, so precision is free here).

    normals: (S, N, 3) from sobol_normals, or (N, 3) for one shared set.
    slots: (B,) mapping problem -> point set (default: b % S, which is the
    vintage->slot identity when the batch is a book in sale order).
    """
    x, cdf, L, g, df = batch.x, batch.cdf, batch.chol, batch.g, batch.df
    dev, B = x.device, x.shape[0]
    if normals.dim() == 2:
        normals = normals[None]
    n_sets, n_paths, _ = normals.shape
    slots = (torch.arange(B, device=dev) % n_sets if slots is None
             else torch.as_tensor(slots, dtype=torch.long, device=dev))
    w_b, w_w, w3 = weights
    c_b = int(max(1, min(B, max_bytes // (n_paths * 3 * 8 * 8))))
    pvs = []
    for i0 in range(0, B, c_b):
        sb = slice(i0, i0 + c_b)
        z = normals[slots[sb]]                            # (cB, N, 3)
        u = torch.special.ndtr(torch.einsum("bni,bji->bnj", z, L[sb]))
        perf = torch.stack(
            [g[sb, i, None] * _interp(u[..., i], cdf[sb, i], x[sb, i])
             for i in range(3)], -1)
        bkt = (w_b * torch.maximum(perf[..., 0], perf[..., 1])
               + w_w * torch.minimum(perf[..., 0], perf[..., 1])
               + w3 * perf[..., 2])
        pvs.append(df[sb] * (bkt - k_lo).clamp(0.0, k_hi - k_lo).mean(-1)
                   * notional)
    return torch.cat(pvs, 0)


def price_rainbow_quad_torch(marginals, corr, df, n_outer=48, n_inner=48,
                             weights=WEIGHTS, k_lo=K_LO, k_hi=K_HI,
                             notional=NOTIONAL, perf_to_date=(1.0, 1.0, 1.0),
                             device=None, survival_dtype=torch.float32):
    """Batch-of-one convenience mirroring rainbow.price_rainbow_quad."""
    batch = QuadBatch.from_problems(
        [QuadProblem(tuple(marginals), corr, df, tuple(perf_to_date))], device)
    pv = price_quad_batch(batch, n_outer, n_inner, weights, k_lo, k_hi,
                          notional, survival_dtype)
    return {"pv": float(pv[0]), "n_nodes": n_outer * 2 * n_inner,
            "device": str(pv.device)}


def _cardinal_consts(knots):
    """Fixed-knot not-a-knot cubic splines are linear in the node values, so
    the whole scipy solve collapses to constant data: the cardinal basis
    CubicSpline(knots, eye(k)) evaluated at q gives the (n_q, k) weight
    matrix with spline(q) = W(q) @ values. Export its per-interval
    polynomial coefficients once; evaluation on the device is then
    searchsorted + gather + Horner."""
    spl = CubicSpline(knots, np.eye(len(knots)))
    return spl.c, spl                       # (4, k-1, k) and the CPU oracle


def _cardinal_w(c, knots, q):
    """Cardinal weights W(q): c (4, k-1, k) constants, q (...,) already
    clipped to [knots[0], knots[-1]]. Returns (..., k), rows sum to 1."""
    i = (torch.searchsorted(knots, q.contiguous(), right=True) - 1
         ).clamp(0, len(knots) - 2)
    t = (q - knots[i]).unsqueeze(-1)
    w = c[0][i]
    for k in range(1, 4):
        w = w * t + c[k][i]
    return w


def _np_gradient(f, x):
    """torch replica of np.gradient(f, x) (2nd-order non-uniform interior,
    1st-order one-sided edges — numpy's default edge_order=1), same
    coefficient expressions so the factory matches the CPU oracle to fp."""
    hd = x[..., 1:-1] - x[..., :-2]
    hs = x[..., 2:] - x[..., 1:-1]
    a = -hs / (hd * (hd + hs))
    b = (hs - hd) / (hd * hs)
    c = hd / (hs * (hd + hs))
    mid = a * f[..., :-2] + b * f[..., 1:-1] + c * f[..., 2:]
    left = (f[..., 1:2] - f[..., 0:1]) / (x[..., 1:2] - x[..., 0:1])
    right = (f[..., -1:] - f[..., -2:-1]) / (x[..., -1:] - x[..., -2:-1])
    return torch.cat([left, mid, right], -1)


class MarginalFactory:
    """Device-resident marginal-table builder: rainbow.implied_marginal as
    batched tensor algebra, no CPU work per build.

    All dates' pillar grids and curve nodes for every underlying upload
    once at construction (the raw inputs are ~10MB for the whole dataset);
    the two cardinal-coefficient tensors (TTM and moneyness axes) plus the
    smile's edge-slope rows are constants. `tables()` then produces the
    x/cdf/psi tables for a whole book of taus in a few fused kernels —
    float64 throughout, which is affordable because every tensor is
    (B, n)-sized. Extras beyond the CPU oracle, both VaR seams:
    `dgrid` adds a pillar-vol shock ((8,13) or (B,8,13)) before the smile
    is read, and `tau_price` freezes the pricing TTM while `tau` decays
    the vol lookup (var.py's roll convention). `di_curve` decouples the
    forward/discount date from the grid date so attribution can move one
    input at a time. Requires tau > 0 (drivers settle expired options at
    intrinsic themselves) and does not need autograd through the tables
    (vol sensitivities are bump-and-revalue by project convention)."""

    def __init__(self, sds, device=None, n=2001, n_sig=10.0):
        self.device = device or default_device()
        self.n, self.n_sig = n, n_sig
        self.names = list(sds)
        first = sds[self.names[0]]
        t64 = lambda a: torch.as_tensor(np.asarray(a, dtype=float),
                                        dtype=torch.float64, device=self.device)
        self.ttm_knots, self.mon_knots = t64(first.ttm_pillars), t64(first.moneyness)
        c_t, self._spl_t = _cardinal_consts(first.ttm_pillars)
        c_m, self._spl_m = _cardinal_consts(first.moneyness)
        self.c_ttm = [t64(c_t[k]) for k in range(4)]
        self.c_mon = [t64(c_m[k]) for k in range(4)]
        lo, hi = float(first.moneyness[0]), float(first.moneyness[-1])
        self.mon_lo, self.mon_hi = lo, hi
        self.dw_lo, self.dw_hi = t64(self._spl_m(lo, 1)), t64(self._spl_m(hi, 1))
        self.u = t64(np.linspace(-1.0, 1.0, n))
        self.grids, self.curve_ttm, self.curve_lndf, self.curve_lnfr = {}, {}, {}, {}
        for name, sd in sds.items():
            assert np.array_equal(sd.ttm_pillars, first.ttm_pillars)
            assert np.array_equal(sd.moneyness, first.moneyness)
            self.grids[name] = t64(sd.grids)
            kmax = max(len(c) for c in sd.curve_ttms)
            pad = lambda cs: t64(np.stack(
                [np.concatenate([c, np.full(kmax - len(c), c[-1])]) for c in cs]))
            self.curve_ttm[name] = pad(sd.curve_ttms)
            self.curve_lndf[name] = pad(sd.curve_lndf)
            self.curve_lnfr[name] = pad(sd.curve_lnfr)

    def _curve(self, table, name, di, tau):
        row = table[name][di]
        return _interp(tau[None, :], self.curve_ttm[name][di][None, :],
                       row[None, :])[0]

    def discount(self, name, di, tau):
        """exp(np.interp) replica of SurfaceData.discount at one date."""
        tau = torch.as_tensor(tau, dtype=torch.float64, device=self.device)
        return torch.exp(self._curve(self.curve_lndf, name, di, tau))

    def forward_ratio(self, name, di, tau):
        """F(tau)/spot — the marginal's model-free mean."""
        tau = torch.as_tensor(tau, dtype=torch.float64, device=self.device)
        return torch.exp(self._curve(self.curve_lnfr, name, di, tau))

    def tables(self, name, di, tau, tau_price=None, di_curve=None, dgrid=None):
        """x, cdf, psi tables ((B, n) each) for one underlying at one date.

        di: grid date index; di_curve (default di): forward-curve date;
        tau (B,): vol-lookup TTMs; tau_price (default tau): TTM used for
        the forward, the sqrt-T in pricing and the grid half-width;
        dgrid: additive pillar-vol shock, (8, 13) or (B, 8, 13)."""
        dev = self.device
        tau = torch.atleast_1d(torch.as_tensor(tau, dtype=torch.float64, device=dev))
        tp = tau if tau_price is None else torch.atleast_1d(
            torch.as_tensor(tau_price, dtype=torch.float64, device=dev))
        tp = tp.expand_as(tau)
        grid = self.grids[name][di]
        if dgrid is not None:
            grid = grid + torch.as_tensor(dgrid, dtype=torch.float64, device=dev)
        wt = _cardinal_w(self.c_ttm, self.ttm_knots,
                         tau.clamp(self.ttm_knots[0], self.ttm_knots[-1]))
        row = (torch.einsum("bk,bkm->bm", wt, grid) if grid.dim() == 3
               else wt @ grid)                                    # (B, 13)
        f = self.forward_ratio(name, di if di_curve is None else di_curve, tp)
        half = (self.n_sig * row.max(-1).values * torch.sqrt(tp)).clamp_min(1.0)
        x = f[:, None] * torch.exp(half[:, None] * self.u[None, :])
        mon = 100.0 * x
        w = _cardinal_w(self.c_mon, self.mon_knots,
                        mon.clamp(self.mon_lo, self.mon_hi))
        vols = (w * row[:, None, :]).sum(-1)
        s_lo, s_hi = row @ self.dw_lo, row @ self.dw_hi
        lam = 50.0
        vols = torch.where(mon < self.mon_lo, row[:, 0:1] + s_lo[:, None] * lam
                           * torch.expm1(-(self.mon_lo - mon) / lam), vols)
        vols = torch.where(mon > self.mon_hi, row[:, -1:] - s_hi[:, None] * lam
                           * torch.expm1(-(mon - self.mon_hi) / lam), vols)
        vols = torch.maximum(vols, 0.25 * torch.minimum(row[:, 0:1], row[:, -1:]))
        st = vols * torch.sqrt(tp.clamp_min(1e-12))[:, None]
        d1 = (torch.log(f[:, None] / x) + 0.5 * st * st) / st
        call = f[:, None] * torch.special.ndtr(d1) - x * torch.special.ndtr(d1 - st)
        cdf = torch.cummax(1.0 + _np_gradient(call, x), -1).values.clamp(0.0, 1.0)
        psi = torch.special.ndtri(cdf.clamp(_EPS, 1.0 - _EPS))
        return x, cdf, psi

    def strike_shift_dgrid(self, name, di, eps):
        """Sticky-strike spot-bump surface for one underlying: under
        S -> S*(1+eps) with vols pinned per STRIKE, the moneyness-quoted
        surface must read sigma_new(m) = sigma_old(m*(1+eps)) — the
        vanilla sticky_strike_dsigma sampling (m * S0/S1) applied to the
        pillar rows before the table build, so the bump re-derives the
        whole implied distribution from the shifted smile. Returns the
        additive pillar shock (8, 13) for tables(dgrid=...); sampling
        clamps at the quoted 50/150 edges (vanilla convention). A flat
        smile shifts onto itself: the shock is exactly zero, which is
        what makes sticky-strike == sticky-moneyness in a flat world."""
        grid = self.grids[name][di]
        mon = (self.mon_knots * (1.0 + eps)).clamp(self.mon_lo, self.mon_hi)
        w = _cardinal_w(self.c_mon, self.mon_knots, mon)
        return grid @ w.T - grid

    def batch(self, tbls, g, df, chol):
        """Assemble a QuadBatch from three per-index `tables()` results.
        chol: one (3, 3) lower cholesky shared by every problem."""
        x = torch.stack([tb[0] for tb in tbls], 1)
        g = torch.as_tensor(g, dtype=torch.float64, device=self.device)
        df = torch.as_tensor(df, dtype=torch.float64, device=self.device)
        return QuadBatch(
            x=x, cdf=torch.stack([tb[1] for tb in tbls], 1),
            psi=torch.stack([tb[2] for tb in tbls], 1),
            chol=chol.expand(x.shape[0], 3, 3), g=g,
            df=df.expand(x.shape[0]) if df.dim() == 0 else df)


def _intrinsic(g, weights=WEIGHTS, k_lo=K_LO, k_hi=K_HI, notional=NOTIONAL):
    """Expiry settlement: the ranked-basket payoff at performances g (B, 3),
    undiscounted — book.py's settle-at-intrinsic convention."""
    w_b, w_w, w3 = weights
    bkt = (w_b * torch.maximum(g[:, 0], g[:, 1])
           + w_w * torch.minimum(g[:, 0], g[:, 1]) + w3 * g[:, 2])
    return (bkt - k_lo).clamp(0.0, k_hi - k_lo) * notional


def prepare_book_greeks(sds, asof=None, eps=0.01, mode="sticky_strike",
                        n_slots=256, device=None, corr_horizon=5, corr=None,
                        fac=None):
    """Assemble the 19 bumped-revaluation batches for the as-of book's
    spot delta vector and gamma/cross-gamma matrix: base, +-eps per
    underlying (6), and the four corners of each pair (12). Book = mark
    semantics: every vintage sold before `asof` (int index into the joint
    dates, a date, or None for the last date) and unexpired at `asof`.

    mode="sticky_strike": each index's spot bump scales its seasoning g
    AND rebuilds that index's tables from the moneyness-shifted smile
    (strike_shift_dgrid) — vols pinned per strike, a *different implied
    distribution* per bumped scenario. mode="sticky_moneyness": tables
    frozen, only g scales (the smile — the whole implied distribution —
    rides with spot; this is what the attribution driver's eq step and
    the autograd delta measure). The pricer/payoff is supplied at
    evaluation time (book_greeks), so one prepared state serves both the
    Sobol and quadrature engines and any path-count sweep.
    """
    if mode not in ("sticky_strike", "sticky_moneyness"):
        raise ValueError(f"unknown mode {mode!r}")
    dev = device or default_device()
    spots = spot_panel(sds)
    names = list(sds)
    d64 = spots.index.values.astype("datetime64[D]")
    if asof is None:
        t = len(d64) - 1
    elif isinstance(asof, (int, np.integer)):
        t = int(asof)
    else:
        t = int(np.searchsorted(d64, np.datetime64(asof, "D")))
        if t >= len(d64) or d64[t] != np.datetime64(asof, "D"):
            raise KeyError(f"{asof} not a joint quote date")
    di = {n: int(np.searchsorted(sds[n].dates, d64[t])) for n in names}
    if corr is None:
        corr = historical_corr(spots, corr_horizon).to_numpy()
    chol = torch.as_tensor(np.linalg.cholesky(np.asarray(corr, dtype=float)),
                           dtype=torch.float64, device=dev)
    fac = fac or MarginalFactory(sds, device=dev)
    expiry = d64 + np.timedelta64(365, "D")
    s_idx = np.nonzero(expiry[:t] > d64[t])[0]
    if not len(s_idx):
        raise ValueError(f"empty book at {spots.index[t].date()}")
    tau = (expiry[s_idx] - d64[t]).astype(float) / 365.0
    sp = spots.to_numpy()
    g0 = torch.as_tensor(sp[t] / sp[s_idx], dtype=torch.float64, device=dev)
    df = fac.discount(names[0], di[names[0]], tau)
    slots = torch.as_tensor(s_idx % n_slots, device=dev)

    tbl = {}
    for j, n in enumerate(names):
        tbl[(j, 0)] = fac.tables(n, di[n], tau)
        for s in (1, -1):
            tbl[(j, s)] = (fac.tables(n, di[n], tau,
                                      dgrid=fac.strike_shift_dgrid(n, di[n], s * eps))
                           if mode == "sticky_strike" else tbl[(j, 0)])

    def mk(shifts):
        tbls = [tbl[(j, shifts.get(j, 0))] for j in range(len(names))]
        g = g0.clone()
        for j, s in shifts.items():
            g[:, j] = g[:, j] * (1.0 + s * eps)
        return fac.batch(tbls, g, df, chol)

    batches = {"base": mk({})}
    for j in range(len(names)):
        for s in (1, -1):
            batches[("d", j, s)] = mk({j: s})
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            for sa in (1, -1):
                for sb in (1, -1):
                    batches[("x", a, sa, b, sb)] = mk({a: sa, b: sb})
    return {"batches": batches, "slots": slots, "names": names, "eps": eps,
            "mode": mode, "n_pos": len(s_idx), "date": spots.index[t],
            "chol": chol, "fac": fac}


def book_greeks(state, pricer, per_position=False):
    """Evaluate a prepared bump set with any batch pricer
    `(batch, slots) -> (B,) pv tensor`. Returns SOLD-book greeks w.r.t.
    RELATIVE spot moves (dS/S; divide by 100 for per-1% numbers):
    delta (n,), gamma (n, n) symmetric Hessian incl. cross terms, pv =
    sold-book mark, and the raw pv_combos. per_position=True adds
    delta_pos (B, n), the per-vintage delta vectors — free, since every
    pricing already returns the full (B,) pv vector. Under the Sobol
    engine one point-set family prices all 19 combos (CRN — the
    differences cancel sampling noise). Gamma MUST come from these
    bumps: per path the payoff is piecewise linear in g, so a
    pathwise/autograd second derivative through the sampler is zero
    almost everywhere."""
    pvv = {k: pricer(b, state["slots"]) for k, b in state["batches"].items()}
    pv = {k: float(v.sum()) for k, v in pvv.items()}
    e = state["eps"]
    n = len(state["names"])
    delta = np.zeros(n)
    gamma = np.zeros((n, n))
    for j in range(n):
        delta[j] = -(pv[("d", j, 1)] - pv[("d", j, -1)]) / (2 * e)
        gamma[j, j] = -(pv[("d", j, 1)] - 2 * pv["base"]
                        + pv[("d", j, -1)]) / e ** 2
    for a in range(n):
        for b in range(a + 1, n):
            gamma[a, b] = gamma[b, a] = -(
                pv[("x", a, 1, b, 1)] - pv[("x", a, 1, b, -1)]
                - pv[("x", a, -1, b, 1)] + pv[("x", a, -1, b, -1)]) / (4 * e ** 2)
    out = {"pv": -pv["base"], "delta": delta, "gamma": gamma, "pv_combos": pv}
    if per_position:
        out["delta_pos"] = torch.stack(
            [-(pvv[("d", j, 1)] - pvv[("d", j, -1)]) / (2 * e)
             for j in range(n)], -1).cpu().numpy()
    return out


def rainbow_book_greeks(sds, asof=None, eps=0.01, n_paths=512, n_slots=256,
                        seed=1, device=None, mode="sticky_strike",
                        engine="sobol", quad_nodes=(24, 24), normals=None,
                        weights=WEIGHTS, k_lo=K_LO, k_hi=K_HI,
                        notional=NOTIONAL, corr_horizon=5, corr=None,
                        fac=None):
    """One-call sold-book spot delta/gamma matrix by central 1%-default
    bumps (19 pricings + 6 shifted table builds). engine="sobol" prices
    every combo on one CRN point-set family (n_paths defaults to the
    512-path book standard); engine="quad" gives zero-variance finite
    differences (the noise-free cross-check)."""
    dev = device or default_device()
    state = prepare_book_greeks(sds, asof, eps, mode, n_slots, dev,
                                corr_horizon, corr, fac)
    if engine == "sobol":
        z = (sobol_normals(n_slots, n_paths, seed=seed, device=dev)
             if normals is None else normals)
        pricer = lambda b, sl: price_sobol_batch(
            b, z, slots=sl, weights=weights, k_lo=k_lo, k_hi=k_hi,
            notional=notional)
    else:
        pricer = lambda b, sl: price_quad_batch(
            b, *quad_nodes, weights=weights, k_lo=k_lo, k_hi=k_hi,
            notional=notional)
    out = book_greeks(state, pricer)
    out.update(date=state["date"], n_pos=state["n_pos"], mode=mode)
    return out


def rainbow_attribution(sds, n_paths=2048, n_slots=256, seed=1, device=None,
                        weights=WEIGHTS, k_lo=K_LO, k_hi=K_HI,
                        notional=NOTIONAL, corr_horizon=5, corr=None,
                        engine="sobol", quad_nodes=(24, 24), t_slice=None,
                        progress=None):
    """Daily P&L attribution of the daily-sold rainbow book — the exotic
    analog of the SPX call-spread study (attribution.py), on the GPU.

    Book: every joint-quote date sell one $1M 1Y 100/112 ranked-basket
    call spread (short), so the book ramps to ~250 vintages; a vintage
    whose expiry falls on or before the valuation date settles at
    intrinsic on that date's spots (book.py's convention). Every
    component is an independent single-input revaluation off the t-1
    close, valued by the CRN Sobol engine (one resident scramble per
    vintage slot, so component differences cancel the sampling noise):

    - eq: spots -> t with tables frozen (the smile rides with spot, the
      sequential sticky-moneyness view); eq_delta is the exact autograd
      d(pv)/dg linear term, eq_higher the remainder.
    - vol_<idx>: index idx's pillar grid -> t, everything else frozen
      (fixed tau, so term roll stays out of vol by construction);
      vol = all three grids -> t; vol_cross = vol - sum of singles.
    - time: all time inputs decay dt (vol lookup, sqrt-T, forward,
      discount) on t-1 surfaces/curves; time_roll is the var.py-style
      sub-piece (vol lookup decayed, pricing TTM frozen).
    - fwd: forward ratios re-read off the t curves (the r-q carry of all
      three indices); rate: USD discount-factor change — a pure
      multiplier on pv, so it costs no revaluation.
    - cross_sv: the spot x vol 4-corner cross on shared tables (the
      vanilla book's vanna line).
    - resid: pl minus everything above (genuine cross terms beyond
      cross_sv; CRN keeps sampling noise out of it).

    The correlation matrix is constant by construction (historical,
    horizon `corr_horizon`), so there is no correlation component.
    Returns a per-date DataFrame of book-level components (sold book:
    positive = the short book makes money), plus the fresh vintage's sale
    premium, mark of the surviving book, position count and spot returns.

    engine="quad" swaps the Sobol sampler for the deterministic quadrature
    at `quad_nodes` (the noise-free engine of record — ~15x the Sobol cost,
    so it is the cross-check on sampled dates, via `t_slice=(start, stop)`
    over date-pair indices, not the daily driver).
    """
    dev = device or default_device()
    spots = spot_panel(sds)
    names = list(sds)
    d64 = spots.index.values.astype("datetime64[D]")
    di = {n: np.searchsorted(sds[n].dates, d64) for n in names}
    if corr is None:
        corr = historical_corr(spots, corr_horizon).to_numpy()
    chol = torch.as_tensor(np.linalg.cholesky(np.asarray(corr, dtype=float)),
                           dtype=torch.float64, device=dev)
    fac = MarginalFactory(sds, device=dev)
    zsets = sobol_normals(n_slots, n_paths, seed=seed, device=dev)
    sp = spots.to_numpy()
    expiry = d64 + np.timedelta64(365, "D")
    t64 = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    rows = []
    t_lo, t_hi = t_slice if t_slice is not None else (1, len(d64))
    for t in range(max(t_lo, 1), min(t_hi, len(d64))):
        live = expiry[:t] > d64[t - 1]
        s_idx = np.nonzero(live)[0]
        tau0 = (expiry[s_idx] - d64[t - 1]).astype(float) / 365.0
        tau1 = (expiry[s_idx] - d64[t]).astype(float) / 365.0
        tau1f = np.maximum(tau1, 1e-6)
        settle = torch.as_tensor(tau1 <= 0, device=dev)
        g0, g1 = t64(sp[t - 1] / sp[s_idx]), t64(sp[t] / sp[s_idx])
        slots = torch.as_tensor(s_idx % n_slots, device=dev)

        T = {}
        for n in names:
            i0, i1 = di[n][t - 1], di[n][t]
            T[n] = {"base": fac.tables(n, i0, tau0),
                    "vol": fac.tables(n, i1, tau0, di_curve=i0),
                    "roll": fac.tables(n, i0, tau1f, tau_price=tau0),
                    "time": fac.tables(n, i0, tau1f),
                    "fwd": fac.tables(n, i0, tau0, di_curve=i1),
                    "full": fac.tables(n, i1, tau1f)}
        usd = names[0]
        df0 = fac.discount(usd, di[usd][t - 1], tau0)
        df_rate = fac.discount(usd, di[usd][t], tau0)
        df_time = fac.discount(usd, di[usd][t - 1], tau1f)
        df_full = fac.discount(usd, di[usd][t], tau1f)

        mk = lambda key3, g, df: fac.batch(
            [T[n][k] for n, k in zip(names, key3)], g, df, chol)
        if engine == "sobol":
            P = lambda b, sl=slots: price_sobol_batch(
                b, zsets, slots=sl, weights=weights, k_lo=k_lo, k_hi=k_hi,
                notional=notional)
        else:
            P = lambda b, sl=None: price_quad_batch(
                b, *quad_nodes, weights=weights, k_lo=k_lo, k_hi=k_hi,
                notional=notional)
        base3 = ["base"] * 3
        vol3 = ["vol"] * 3
        gg = g0.clone().requires_grad_(True)
        pv_base = P(mk(base3, gg, df0))
        pv_base.sum().backward()
        delta = gg.grad
        pv_base = pv_base.detach()
        pv_eq = P(mk(base3, g1, df0))
        pv_vol1 = {n: P(mk(["vol" if m == n else "base" for m in names], g0, df0))
                   for n in names}
        pv_vol = P(mk(vol3, g0, df0))
        pv_roll = P(mk(["roll"] * 3, g0, df0))
        pv_time = torch.where(settle, _intrinsic(g0, weights, k_lo, k_hi, notional),
                              P(mk(["time"] * 3, g0, df_time)))
        pv_fwd = P(mk(["fwd"] * 3, g0, df0))
        pv_x = P(mk(vol3, g1, df0))
        pv_full = torch.where(settle, _intrinsic(g1, weights, k_lo, k_hi, notional),
                              P(mk(["full"] * 3, g1, df_full)))

        # fresh vintage sold at t (enters tomorrow's book on slot t % n_slots)
        prem_tb = [fac.tables(n, di[n][t], np.array([1.0])) for n in names]
        prem = P(fac.batch(prem_tb, t64([[1.0, 1.0, 1.0]]),
                           fac.discount(usd, di[usd][t], np.array([1.0])), chol),
                 torch.as_tensor([t % n_slots], device=dev))

        eq = pv_eq - pv_base
        eq_delta = (delta * (g1 - g0)).sum(-1)
        vol = pv_vol - pv_base
        time = pv_time - pv_base
        fwd = pv_fwd - pv_base
        rate = (df_rate / df0 - 1.0) * pv_base
        cross = pv_x - pv_vol - pv_eq + pv_base
        pl = pv_full - pv_base
        resid = pl - eq - vol - time - fwd - rate - cross
        s = lambda v: -float(v.sum())          # sold book
        row = {"date": spots.index[t], "n_pos": len(s_idx),
               "n_settle": int(settle.sum()),
               "pl": s(pl), "eq": s(eq), "eq_delta": s(eq_delta),
               "eq_higher": s(eq - eq_delta), "vol": s(vol),
               "vol_cross": s(vol - sum(pv_vol1.values()) + 3 * pv_base),
               "time": s(time), "time_roll": s(pv_roll - pv_base),
               "time_other": s(time - (pv_roll - pv_base)),
               "fwd": s(fwd), "rate": s(rate), "cross_sv": s(cross),
               "resid": s(resid), "premium": float(prem[0]),
               "mark": -float(pv_full[~settle].sum()),
               "dt": float((d64[t] - d64[t - 1]).astype(float)) / 365.0}
        for j, n in enumerate(names):
            row[f"vol_{n.lower()}"] = s(pv_vol1[n] - pv_base)
            row[f"ret_{n.lower()}"] = sp[t, j] / sp[t - 1, j] - 1.0
        rows.append(row)
        if progress and t % progress == 0:
            print(f"  {t}/{len(d64) - 1} {spots.index[t].date()} "
                  f"n_pos={len(s_idx)}", flush=True)
    return pd.DataFrame(rows).set_index("date")


def rainbow_attribution_ss(sds, n_paths=512, n_slots=256, seed=1, eps=0.01,
                           device=None, weights=WEIGHTS, k_lo=K_LO, k_hi=K_HI,
                           notional=NOTIONAL, corr_horizon=5, corr=None,
                           t_slice=None, progress=None):
    """The sticky-strike re-cut of rainbow_attribution (user spec
    2026-08-06) — the attribution whose delta line is the sticky-strike
    hedge, so the delta-hedged series settles which framework hedges
    better. Same book, engine, CRN discipline and settle conventions as
    the sticky-moneyness driver; the pl/time/fwd/rate lines are the
    identical constructs (they match the old driver to fp at equal
    n_paths/seed) — only the equity/vol cut moves.

    Components (all independent single-input revals off the t-1 close,
    sold-book sign, fixed lookup tau so term roll stays out of eq/vol):

    - eq: ALL spots -> t the sticky-strike way — each index's tables are
      re-derived from its moneyness-shifted smile (strike_shift_dgrid at
      the realized ratio u_i = S1/S0: vols pinned per strike, a different
      implied distribution) AND g -> g1. Attributed by the 1%-bump greeks
      matrix measured on the same book/points (book_greeks): per-index
      delta and gamma terms, per-pair cross-gamma terms, eq_resid.
    - vol: ALL surfaces -> t at fixed strikes (day-t grids sampled at
      m/u_i, t-1 anchor and curves), g0; per-index singles + vol_resid
      (joint-move interaction). eq and vol compose exactly: applying both
      = plain day-t tables at g1, which is what cross_ev's 4-corner uses.
    - cross_ev: P(eq+vol) - P(eq) - P(vol) + P(base), the total
      equity x vol cross by bumped revaluations.
    - time, split in three: theta_value = discount-factor decay on the
      base pv (zero revaluations); theta_delta = per-vintage sticky-strike
      deltas x each vintage's forward-ratio decay over dt (the carry of
      the delta position, from the greek singles' pv vectors);
      theta_gamma = the rest (gamma theta + vol carry + drift
      nonlinearity, folded per spec).
    - fwd, rate, resid, premium, mark: as in the sticky-moneyness driver.

    Cost: ~30 table builds + 30 pricings/date at the 512-path standard.
    """
    dev = device or default_device()
    spots = spot_panel(sds)
    names = list(sds)
    d64 = spots.index.values.astype("datetime64[D]")
    di = {n: np.searchsorted(sds[n].dates, d64) for n in names}
    if corr is None:
        corr = historical_corr(spots, corr_horizon).to_numpy()
    chol = torch.as_tensor(np.linalg.cholesky(np.asarray(corr, dtype=float)),
                           dtype=torch.float64, device=dev)
    fac = MarginalFactory(sds, device=dev)
    zsets = sobol_normals(n_slots, n_paths, seed=seed, device=dev)
    sp = spots.to_numpy()
    expiry = d64 + np.timedelta64(365, "D")
    t64 = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    P = lambda b, sl: price_sobol_batch(b, zsets, slots=sl, weights=weights,
                                        k_lo=k_lo, k_hi=k_hi, notional=notional)
    lo = [n.lower() for n in names]
    rows = []
    t_lo, t_hi = t_slice if t_slice is not None else (1, len(d64))
    for t in range(max(t_lo, 1), min(t_hi, len(d64))):
        s_idx = np.nonzero(expiry[:t] > d64[t - 1])[0]
        tau0 = (expiry[s_idx] - d64[t - 1]).astype(float) / 365.0
        tau1 = (expiry[s_idx] - d64[t]).astype(float) / 365.0
        tau1f = np.maximum(tau1, 1e-6)
        settle = torch.as_tensor(tau1 <= 0, device=dev)
        g0, g1 = t64(sp[t - 1] / sp[s_idx]), t64(sp[t] / sp[s_idx])
        slots = torch.as_tensor(s_idx % n_slots, device=dev)
        u = sp[t] / sp[t - 1]                        # realized spot ratios
        ret = u - 1.0

        T = {}
        for j, n in enumerate(names):
            i0, i1 = di[n][t - 1], di[n][t]
            sh = lambda ii, e: fac.strike_shift_dgrid(n, ii, e)
            T[j] = {"base": fac.tables(n, i0, tau0),
                    1: fac.tables(n, i0, tau0, dgrid=sh(i0, eps)),
                    -1: fac.tables(n, i0, tau0, dgrid=sh(i0, -eps)),
                    "equ": fac.tables(n, i0, tau0, dgrid=sh(i0, u[j] - 1.0)),
                    "vol": fac.tables(n, i1, tau0, di_curve=i0,
                                      dgrid=sh(i1, 1.0 / u[j] - 1.0)),
                    "ev": fac.tables(n, i1, tau0, di_curve=i0),
                    "time": fac.tables(n, i0, tau1f),
                    "fwd": fac.tables(n, i0, tau0, di_curve=i1),
                    "full": fac.tables(n, i1, tau1f)}
        usd, iu0, iu1 = names[0], di[names[0]][t - 1], di[names[0]][t]
        df0 = fac.discount(usd, iu0, tau0)
        df_rate = fac.discount(usd, iu1, tau0)
        df_time = fac.discount(usd, iu0, tau1f)
        df_full = fac.discount(usd, iu1, tau1f)

        def mk(keys, g, df):
            return fac.batch([T[j][k] for j, k in enumerate(keys)], g, df, chol)

        def mkb(shifts):                 # greek combos off the base tables
            g = g0.clone()
            for j, sgn in shifts.items():
                g[:, j] = g[:, j] * (1.0 + sgn * eps)
            return mk([shifts.get(j, "base") for j in range(3)], g, df0)

        gbat = {"base": mkb({})}
        for j in range(3):
            for sgn in (1, -1):
                gbat[("d", j, sgn)] = mkb({j: sgn})
        for a in range(3):
            for b in range(a + 1, 3):
                for sa in (1, -1):
                    for sb in (1, -1):
                        gbat[("x", a, sa, b, sb)] = mkb({a: sa, b: sb})
        gk = book_greeks({"batches": gbat, "slots": slots, "names": names,
                          "eps": eps}, P, per_position=True)

        base3 = ["base"] * 3
        pv_base = P(mk(base3, g0, df0), slots)
        pv_eq = P(mk(["equ"] * 3, g1, df0), slots)
        pv_vol1 = [P(mk(["vol" if m == j else "base" for m in range(3)],
                        g0, df0), slots) for j in range(3)]
        pv_vol = P(mk(["vol"] * 3, g0, df0), slots)
        pv_ev = P(mk(["ev"] * 3, g1, df0), slots)
        pv_time = torch.where(settle, _intrinsic(g0, weights, k_lo, k_hi,
                                                 notional),
                              P(mk(["time"] * 3, g0, df_time), slots))
        pv_fwd = P(mk(["fwd"] * 3, g0, df0), slots)
        pv_full = torch.where(settle, _intrinsic(g1, weights, k_lo, k_hi,
                                                 notional),
                              P(mk(["full"] * 3, g1, df_full), slots))

        prem_tb = [fac.tables(n, di[n][t], np.array([1.0])) for n in names]
        prem = P(fac.batch(prem_tb, t64([[1.0, 1.0, 1.0]]),
                           fac.discount(usd, iu1, np.array([1.0])), chol),
                 torch.as_tensor([t % n_slots], device=dev))

        s = lambda v: -float(v.sum())               # sold book
        eq = s(pv_eq - pv_base)
        vol = s(pv_vol - pv_base)
        vol1 = [s(v - pv_base) for v in pv_vol1]
        cross_ev = s(pv_ev - pv_eq - pv_vol + pv_base)
        time = s(pv_time - pv_base)
        fwd = s(pv_fwd - pv_base)
        rate = s((df_rate / df0 - 1.0) * pv_base)
        pl = s(pv_full - pv_base)

        # eq Taylor pieces from the sold-signed 1%-bump matrix
        d_terms = gk["delta"] * ret
        g_terms = 0.5 * np.diag(gk["gamma"]) * ret ** 2
        pairs = [(0, 1), (0, 2), (1, 2)]
        x_terms = [gk["gamma"][a, b] * ret[a] * ret[b] for a, b in pairs]
        # theta pieces: discounting on the base pv, then the per-vintage
        # sticky-strike deltas carried by each vintage's forward decay
        theta_value = s((df_time / df0 - 1.0) * pv_base)
        drift = np.stack(
            [(fac.forward_ratio(n, di[n][t - 1], tau1f)
              / fac.forward_ratio(n, di[n][t - 1], tau0) - 1.0).cpu().numpy()
             for n in names], -1)
        theta_delta = float((gk["delta_pos"] * drift).sum())

        row = {"date": spots.index[t], "n_pos": len(s_idx),
               "n_settle": int(settle.sum()), "pl": pl,
               "eq": eq, "eq_delta": float(d_terms.sum()),
               **{f"eq_delta_{n}": float(d_terms[j]) for j, n in enumerate(lo)},
               "eq_gamma": float(g_terms.sum()),
               **{f"eq_gamma_{n}": float(g_terms[j]) for j, n in enumerate(lo)},
               "eq_xgamma": float(np.sum(x_terms)),
               **{f"eq_xgamma_{lo[a]}_{lo[b]}": float(x)
                  for (a, b), x in zip(pairs, x_terms)},
               "eq_resid": eq - float(d_terms.sum() + g_terms.sum()
                                      + np.sum(x_terms)),
               "vol": vol,
               **{f"vol_{n}": vol1[j] for j, n in enumerate(lo)},
               "vol_resid": vol - sum(vol1),
               "cross_ev": cross_ev, "time": time,
               "theta_value": theta_value, "theta_delta": theta_delta,
               "theta_gamma": time - theta_value - theta_delta,
               "fwd": fwd, "rate": rate,
               "resid": pl - eq - vol - cross_ev - time - fwd - rate,
               "premium": float(prem[0]),
               "mark": -float(pv_full[~settle].sum()),
               "dt": float((d64[t] - d64[t - 1]).astype(float)) / 365.0,
               **{f"ret_{n}": ret[j] for j, n in enumerate(lo)}}
        rows.append(row)
        if progress and t % progress == 0:
            print(f"  {t}/{len(d64) - 1} {spots.index[t].date()} "
                  f"n_pos={len(s_idx)}", flush=True)
    return pd.DataFrame(rows).set_index("date")
