"""GPU Monte Carlo rungs for the RILA reset package under local vol.

The second `*_torch.py` module in the repo. House rule unchanged — the numpy
engines in `vol_pca.local_vol` are canonical and these mirror them, with
tests pinning parity — only the "single torch module" wording generalises:
torch lives inside `*_torch.py` files and nowhere else.

Three rungs, each measured on error vs wall-clock (the honest axis):

(a) **pseudo on device** — the same log-Euler estimator as
    `local_vol.price_rila_mc`, throughput rung. Feed it the CPU's normals
    (`normals=`) and it reproduces the CPU price to ~1e-12 in float64.
(b) **scrambled Sobol + Brownian bridge** — dimension = number of steps
    (310 weekly / 1512 daily over the 6y). The bridge (QuantLib/Glasserman
    ordering) hands the terminal increment to Sobol coordinate 0 and then
    bisects, so the leading, best-equidistributed coordinates carry the
    coarse shape of the path and the effective dimension collapses. Error
    bars come from independent scrambles, the repo's standard QMC
    error-estimation pattern.
(c) **hybrid / conditional smoothing** — simulate only the six-month reset
    window (daily -> 126 dimensions) and integrate the 5.5y tail *exactly*:
    `make_tail_table` precomputes T(S_w, K) = the LV value at the window end
    of the European package with basis K, via one banded CN solve family
    over ~100 K nodes (`local_vol.pde_tail_table`). The payoff becomes a
    smooth 2-D table lookup instead of a kinked terminal function, which
    kills both the terminal variance and the QMC-unfriendly kinks. Seasoned
    mid-window states use the identical design with their own (shorter)
    window and table.

Sobol points come from `torch.quasirandom.SobolEngine` drawn straight into
a device buffer, not from rainbow_torch's `sobol_normals`: that pattern
generates one small resident point set per book slot (d = 3), while these
rungs need d = 84..1512 x N points, which neither fits "generate once, keep
resident" nor survives the host round-trip. Scrambling is per-seed, so a
scramble replicate is a fresh engine seed.

Numerics: path state is carried as q = ln(S/S0) and the basis as K/S0, both
O(1), so `dtype=torch.float32` is safe for the evolution (verified against
float64 in the tests and the notebook); payoff reduction always accumulates
in float64. Path chunks are sized under `max_bytes` — the (chunk, n_steps)
normal block, not the path state, is what fills an 8GB card.
"""

import time

import numpy as np
import torch

from vol_pca.local_vol import (K_CALL_HI, K_CALL_LO, K_PUT, mc_time_grid,
                               pde_tail_table)
from vol_pca.rainbow_torch import default_device


# ------------------------------------------------------------- device LV

class DeviceLV:
    """Local-vol slices and curve increments for one fixed time grid."""

    def __init__(self, lv, t_grid, device=None, dtype=torch.float32):
        self.device = device = device or default_device()
        self.dtype = dtype
        self.t_grid = np.asarray(t_grid, dtype=float)
        rows = np.stack([lv.sigma_row(t) for t in self.t_grid[:-1]])
        lnf = lv.ln_fwd(self.t_grid)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        self.rows = t(rows)                                   # (n_steps, ny)
        self.dlnf = t(np.diff(lnf))
        self.dl = t(lnf[:-1] - np.log(lv.spot))               # ln F_k - ln S0
        self.dt = t(np.diff(self.t_grid))
        self.sqdt = t(np.sqrt(np.diff(self.t_grid)))
        self.y0 = float(lv.ys[0])
        self.dy = float(lv.ys[1] - lv.ys[0])
        self.ny = len(lv.ys)
        self.spot = float(lv.spot)
        self.n_steps = len(self.t_grid) - 1

    def sigma(self, k, q):
        """sigma_loc(t_k, S) for path state q = ln(S/S0)."""
        u = ((q - self.dl[k] - self.y0) / self.dy).clamp(0.0, self.ny - 1.0001)
        i = u.long()
        f = u - i
        row = self.rows[k]
        return row[i] * (1.0 - f) + row[i + 1] * f


# ------------------------------------------------------- Brownian bridge

def bb_plan(times):
    """QuantLib/Glasserman bridge ordering for increments over `times`
    (t_1..t_n, with t_0 = 0). Returns the per-dimension gather plan."""
    t = np.asarray(times, dtype=float)
    n = len(t)
    filled = np.zeros(n, dtype=bool)
    bidx = np.zeros(n, dtype=np.int64)
    lidx = np.zeros(n, dtype=np.int64)
    ridx = np.zeros(n, dtype=np.int64)
    lw = np.zeros(n)
    rw = np.zeros(n)
    sd = np.zeros(n)
    bidx[0] = n - 1
    filled[n - 1] = True
    sd[0] = np.sqrt(t[n - 1])
    lidx[0] = ridx[0] = -1
    j = 0
    for i in range(1, n):
        while filled[j]:
            j += 1
        k = j
        while not filled[k]:
            k += 1
        m = j + ((k - 1 - j) >> 1)
        filled[m] = True
        bidx[i], lidx[i], ridx[i] = m, j - 1, k
        t_l = t[j - 1] if j else 0.0
        lw[i] = (t[k] - t[m]) / (t[k] - t_l)
        rw[i] = (t[m] - t_l) / (t[k] - t_l)
        sd[i] = np.sqrt((t[m] - t_l) * (t[k] - t[m]) / (t[k] - t_l))
        j = k + 1
        if j >= n:
            j = 0
    return {"bidx": bidx, "lidx": lidx, "ridx": ridx, "lw": lw, "rw": rw,
            "sd": sd}


def bb_increments(z, plan):
    """Turn standard normals (m, n) into Brownian increments (m, n) through
    the bridge: coordinate 0 sets W(t_n), the rest bisect."""
    m, n = z.shape
    w = torch.empty_like(z)
    bidx, lidx, ridx = plan["bidx"], plan["lidx"], plan["ridx"]
    lw, rw, sd = plan["lw"], plan["rw"], plan["sd"]
    w[:, int(bidx[0])] = float(sd[0]) * z[:, 0]
    for i in range(1, n):
        b, l, r = int(bidx[i]), int(lidx[i]), int(ridx[i])
        acc = float(rw[i]) * w[:, r] + float(sd[i]) * z[:, i]
        if l >= 0:
            acc = acc + float(lw[i]) * w[:, l]
        w[:, b] = acc
    return torch.diff(w, dim=1, prepend=torch.zeros(m, 1, dtype=z.dtype,
                                                    device=z.device))


# --------------------------------------------------------------- normals

class _NormalSource:
    """Chunked standard normals: pseudo (torch RNG) or scrambled Sobol."""

    def __init__(self, method, dim, seed, device, dtype):
        self.method, self.dim, self.device, self.dtype = method, dim, device, dtype
        if method == "pseudo":
            self.gen = torch.Generator(device=device)
            self.gen.manual_seed(int(seed))
        elif method == "sobol":
            self.eng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True,
                                                     seed=int(seed))
            self.buf = None
        else:
            raise ValueError(f"unknown method {method!r}")

    def draw(self, m):
        if self.method == "pseudo":
            return torch.randn(m, self.dim, generator=self.gen,
                               device=self.device, dtype=self.dtype)
        if self.buf is None or self.buf.shape[0] != m:
            self.buf = torch.empty(m, self.dim, dtype=torch.float64,
                                   device=self.device)
        self.eng.draw(m, out=self.buf, dtype=torch.float64)
        return torch.special.ndtri(self.buf.clamp(1e-15, 1.0 - 1e-15)).to(self.dtype)


# ------------------------------------------------------------ tail table

class TailTable:
    """Device copy of T(S_w, K) = LV value at the window end of the European
    package with basis K (already discounted to the window end)."""

    def __init__(self, tab, device=None, dtype=torch.float32):
        device = device or default_device()
        self.x = torch.as_tensor(tab["x"], dtype=dtype, device=device)
        self.k = torch.as_tensor(tab["k"], dtype=dtype, device=device)
        self.v = torch.as_tensor(tab["v"], dtype=dtype, device=device)
        self.x0, self.dx = float(tab["x"][0]), float(tab["x"][1] - tab["x"][0])
        self.nx = len(tab["x"])
        lnk = np.log(tab["k"])
        self.lk0, self.dlk = float(lnk[0]), float(lnk[1] - lnk[0])
        self.nk = len(tab["k"])
        self.t_window = float(tab["t_window"])

    def lookup(self, lns, k_abs):
        """Bilinear in (ln S, ln K); linear-in-K extrapolation above the top
        node (mirrors the PDE ratchet gather) and clamped in S."""
        ux = ((lns - self.x0) / self.dx).clamp(0.0, self.nx - 1.0001)
        ix = ux.long()
        fx = ux - ix
        uk = ((torch.log(k_abs) - self.lk0) / self.dlk).clamp(0.0, self.nk - 2.0)
        ik = uk.long()
        k0 = self.k[ik]
        fk = (k_abs - k0) / (self.k[ik + 1] - k0)
        v = self.v
        v00 = v[ix, ik]
        v10 = v[ix + 1, ik]
        v01 = v[ix, ik + 1]
        v11 = v[ix + 1, ik + 1]
        return ((1 - fx) * ((1 - fk) * v00 + fk * v01)
                + fx * ((1 - fk) * v10 + fk * v11))


def make_tail_table(lv, state, device=None, dtype=torch.float32, **pde_kw):
    return TailTable(pde_tail_table(lv, state, **pde_kw), device=device,
                     dtype=dtype)


# --------------------------------------------------------------- pricing

def rila_price_torch(lv, state, n_paths=1 << 16, per_year=252, method="sobol",
                     bridge=True, tail=None, dtype=torch.float32, seed=0,
                     device=None, chunk=None, max_bytes=1 << 31, normals=None,
                     reset=True, dev_lv=None, t_grid=None, fix_idx=None):
    """One Monte-Carlo valuation of the package on the device.

    `tail` (a TailTable) switches on the hybrid: the simulation stops at the
    window end and the terminal payoff is replaced by the exact conditional
    value T(S_w, K_w). Without it the paths run to maturity and hit the raw
    kinked payoff.

    `normals` (n_paths, n_steps) forces a given draw — the CRN parity hook
    against `local_vol.price_rila_mc`; `bridge` is then ignored (the CPU
    engine consumes increments in time order).
    """
    device = device or default_device()
    if t_grid is None:
        end = state.window_end if tail is not None else state.ttm
        sub = state.__class__(basis=state.basis, spot=state.spot, ttm=end,
                              fixings=state.fixings)
        t_grid, fix_idx = mc_time_grid(sub, per_year)
    dev_lv = dev_lv or DeviceLV(lv, t_grid, device=device, dtype=dtype)
    n_steps = dev_lv.n_steps
    plan = bb_plan(t_grid[1:]) if (bridge and normals is None) else None

    if chunk is None:
        chunk = max(1, min(n_paths, int(max_bytes // (n_steps * 16))))
        if method == "sobol" and normals is None:
            chunk = n_paths if chunk >= n_paths else 1 << int(np.log2(chunk))
    src = (_NormalSource(method, n_steps, seed, device, dtype)
           if normals is None else None)

    fix = set(int(i) for i in np.atleast_1d(fix_idx)) if reset else set()
    kr0 = state.basis / state.spot
    tot = torch.zeros((), dtype=torch.float64, device=device)
    tot2 = torch.zeros((), dtype=torch.float64, device=device)
    tot_s = torch.zeros((), dtype=torch.float64, device=device)
    for i0 in range(0, n_paths, chunk):
        m = min(chunk, n_paths - i0)
        if normals is None:
            z = src.draw(m)
            dw = bb_increments(z, plan) if plan is not None else z * dev_lv.sqdt
        else:
            z = torch.as_tensor(np.asarray(normals[i0:i0 + m]), dtype=dtype,
                                device=device)
            dw = z * dev_lv.sqdt
        q = torch.zeros(m, dtype=dtype, device=device)
        kr = torch.full((m,), kr0, dtype=dtype, device=device)
        for k in range(n_steps):
            sig = dev_lv.sigma(k, q)
            q = q + dev_lv.dlnf[k] - 0.5 * sig * sig * dev_lv.dt[k] + sig * dw[:, k]
            if (k + 1) in fix:
                kr = torch.maximum(kr, torch.exp(q))
        if tail is None:
            s, kk = torch.exp(q).double(), kr.double()
            pay = ((K_PUT * kk - s).clamp_min(0.0) - (s - K_CALL_LO * kk).clamp_min(0.0)
                   + (s - K_CALL_HI * kk).clamp_min(0.0))
        else:
            pay = tail.lookup(q + np.log(dev_lv.spot),
                              kr * dev_lv.spot).double() / dev_lv.spot
        tot += pay.sum()
        tot2 += (pay * pay).sum()
        tot_s += torch.exp(q).double().sum()
    mean = float(tot / n_paths)
    var = max(float(tot2 / n_paths) - mean**2, 0.0)
    t_end = dev_lv.t_grid[-1] if tail is None else tail.t_window
    d = float(lv.discount(t_end))
    scale = d * dev_lv.spot
    return {"pv": scale * mean, "se": scale * np.sqrt(var / max(n_paths - 1, 1)),
            "n_paths": n_paths, "n_steps": n_steps, "n_dim": n_steps,
            "method": method, "bridge": plan is not None,
            "hybrid": tail is not None, "dtype": str(dtype),
            "mean_s": dev_lv.spot * float(tot_s / n_paths),
            "fwd": float(lv.fwd(t_end))}


def prepare(lv, state, per_year=252, tail=None, device=None,
            dtype=torch.float32):
    """Time grid + device LV slices, built once and reused across
    replicates so the measured wall-clock is the estimator's, not the
    setup's (the setup is reported separately)."""
    end = state.window_end if tail is not None else state.ttm
    sub = state.__class__(basis=state.basis, spot=state.spot, ttm=end,
                          fixings=state.fixings)
    t_grid, fix_idx = mc_time_grid(sub, per_year)
    return {"t_grid": t_grid, "fix_idx": fix_idx,
            "dev_lv": DeviceLV(lv, t_grid, device=device, dtype=dtype),
            "tail": tail, "dtype": dtype, "device": device}


def rila_replicates(lv, state, n_paths, replicates=16, seed0=1000, prep=None,
                    per_year=252, tail=None, device=None,
                    dtype=torch.float32, **kw):
    """R independent scrambles (or seeds) of the same estimator.

    Returns the replicate mean, the standard error of that mean across
    replicates (the QMC error bar — a scrambled-Sobol run has no usable
    internal s.e.), the per-replicate RMSE about the mean, and the median
    single-replicate wall clock.
    """
    prep = prep or prepare(lv, state, per_year, tail, device, dtype)
    vals, secs = [], []
    for r in range(replicates):
        t0 = _now()
        out = rila_price_torch(lv, state, n_paths=n_paths, seed=seed0 + r,
                               per_year=per_year, **prep, **kw)
        secs.append(_now() - t0)
        vals.append(out["pv"])
    v = np.asarray(vals)
    return {"pv": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))),
            "rmse": float(v.std(ddof=1)), "vals": v,
            "secs": float(np.median(secs)), "secs_all": np.asarray(secs),
            "n_paths": n_paths, "replicates": replicates,
            "n_dim": out["n_steps"], "hybrid": out["hybrid"],
            "method": out["method"], "bridge": out["bridge"]}


def _now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def timed_price(fn, *a, **kw):
    """Wall-clock a device pricing call, synchronising around it."""
    t0 = _now()
    out = fn(*a, **kw)
    out["secs"] = _now() - t0
    return out
