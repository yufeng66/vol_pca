import dataclasses
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.stats import norm

from tests.test_book import _flat_sd
from vol_pca.data import load_surfaces
from vol_pca.rainbow import (date_index, historical_corr, implied_marginal,
                             price_rainbow, price_rainbow_quad, spot_panel)
from vol_pca.rainbow_torch import (MarginalFactory, QuadBatch, QuadProblem,
                                   price_quad_batch, price_rainbow_quad_torch,
                                   price_sobol_batch, rainbow_attribution,
                                   sobol_normals)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV3 = [ROOT / f"{n}_volSurface.csv" for n in ("SPX", "SX5E", "HSI")]
CORR = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])


@pytest.fixture(scope="module")
def flat_marg():
    return implied_marginal(_flat_sd(), 0, 1.0)


def test_fp64_matches_numpy_oracle(flat_marg):
    ref = price_rainbow_quad([flat_marg] * 3, CORR, df=0.96)
    q = price_rainbow_quad_torch([flat_marg] * 3, CORR, df=0.96, device="cpu",
                                 survival_dtype=torch.float64)
    assert abs(q["pv"] - ref["pv"]) < 1e-6          # same algorithm, fp64
    assert q["n_nodes"] == ref["n_nodes"]


def test_fp32_survival_stage_is_benign(flat_marg):
    # float32 only touches the survival/trapezoid stage; the float64
    # accumulation keeps the price within a fraction of a cent on $1M
    ref = price_rainbow_quad([flat_marg] * 3, CORR, df=0.96)["pv"]
    q = price_rainbow_quad_torch([flat_marg] * 3, CORR, df=0.96)
    assert abs(q["pv"] - ref) < 0.05                # dollars on $1M notional


def test_seasoned_batch_matches_numpy_loop(flat_marg):
    ms = implied_marginal(_flat_sd(), 0, 0.25)
    probs = [QuadProblem((flat_marg,) * 3, CORR, 0.96),
             QuadProblem((ms,) * 3, CORR, 0.99, (1.15, 0.92, 1.05)),
             QuadProblem((ms, flat_marg, ms), np.eye(3), 0.97, (0.9, 1.1, 1.0))]
    batch = QuadBatch.from_problems(probs, device="cpu")
    assert len(batch) == 3
    pv = price_quad_batch(batch).numpy()
    ref = [price_rainbow_quad(list(p.marginals), p.corr, p.df,
                              perf_to_date=p.perf_to_date)["pv"] for p in probs]
    assert np.abs(pv - np.array(ref)).max() < 0.05


def test_chunked_equals_one_shot(flat_marg):
    probs = [QuadProblem((flat_marg,) * 3, CORR, 0.96, (1.0, 1.0, g3))
             for g3 in (0.9, 1.0, 1.1, 1.2)]
    batch = QuadBatch.from_problems(probs, device="cpu")
    one = price_quad_batch(batch)
    tiny = price_quad_batch(batch, max_bytes=10 << 20)   # forces many chunks
    assert torch.allclose(one, tiny, atol=1e-9)


def test_analytic_anchors(flat_marg):
    # strikes far below the support pay the full width (below-grid survival
    # branch); strikes far above pay nothing
    deep = price_rainbow_quad_torch([flat_marg] * 3, np.eye(3), df=1.0,
                                    k_lo=0.01, k_hi=0.02)
    far = price_rainbow_quad_torch([flat_marg] * 3, np.eye(3), df=1.0,
                                   k_lo=50.0, k_hi=50.01)
    assert abs(deep["pv"] - 0.01 * 1e6) < 1e-2
    assert abs(far["pv"]) < 1e-6


def test_autograd_delta_matches_finite_difference(flat_marg):
    # one backward pass = exact d(pv)/dg for every problem and index; the
    # checkpointed survival stage must not perturb values or gradients
    g = (1.05, 0.97, 1.02)
    prob = QuadProblem((flat_marg,) * 3, CORR, 0.96, g)
    batch = QuadBatch.from_problems([prob], device="cpu")
    plain = price_quad_batch(batch, n_outer=32, n_inner=32)
    batch.g.requires_grad_(True)
    pv = price_quad_batch(batch, n_outer=32, n_inner=32)
    assert torch.allclose(pv, plain, atol=1e-9)      # ckpt path same values
    pv.sum().backward()
    grad = batch.g.grad.numpy().ravel()
    h = 1e-4
    for j in range(3):
        gp, gm = np.array(g), np.array(g)
        gp[j] += h
        gm[j] -= h
        fd = (price_rainbow_quad([flat_marg] * 3, CORR, 0.96, n_outer=32,
                                 n_inner=32, perf_to_date=tuple(gp))["pv"]
              - price_rainbow_quad([flat_marg] * 3, CORR, 0.96, n_outer=32,
                                   n_inner=32, perf_to_date=tuple(gm))["pv"]) / (2 * h)
        assert abs(grad[j] / fd - 1.0) < 1e-3


def _numpy_replay(z, marginals, corr, dfr, g):
    """The numpy sampling pipeline applied to the engine's own normals."""
    u = norm.cdf(z @ np.linalg.cholesky(np.asarray(corr, dtype=float)).T)
    perf = np.column_stack([m.quantile(u[:, j])
                            for j, m in enumerate(marginals)]) * np.asarray(g)
    b = (0.5 * np.maximum(perf[:, 0], perf[:, 1])
         + 0.3 * np.minimum(perf[:, 0], perf[:, 1]) + 0.2 * perf[:, 2])
    return dfr * np.clip(b - 1.0, 0.0, 0.12).mean() * 1e6


def test_sobol_engine_replays_numpy_math(flat_marg):
    # on identical normals the torch sampler must reproduce the numpy
    # pipeline (cholesky -> uniforms -> quantiles -> ranked payoff) exactly
    zs = sobol_normals(1, 2 ** 12, seed=3, device="cpu")
    batch = QuadBatch.from_problems([QuadProblem((flat_marg,) * 3, CORR, 0.96)],
                                    "cpu", scores=False)
    pv = float(price_sobol_batch(batch, zs)[0])
    ref = _numpy_replay(zs[0].numpy(), [flat_marg] * 3, CORR, 0.96, (1, 1, 1))
    assert abs(pv - ref) < 1e-6
    # and land within sampling noise of the numpy sobol oracle's own scramble
    oracle = price_rainbow([flat_marg] * 3, CORR, 0.96, n_paths=2 ** 12,
                           seed=0, method="sobol")["pv"]
    assert abs(pv - oracle) < 200.0


def test_sobol_slots_season_and_chunk(flat_marg):
    ms = implied_marginal(_flat_sd(), 0, 0.25)
    probs = [QuadProblem((flat_marg,) * 3, CORR, 0.96),
             QuadProblem((ms,) * 3, CORR, 0.99, (1.15, 0.92, 1.05)),
             QuadProblem((ms, flat_marg, ms), np.eye(3), 0.97, (0.9, 1.1, 1.0))]
    zs = sobol_normals(3, 2 ** 11, seed=5, device="cpu")
    batch = QuadBatch.from_problems(probs, "cpu", scores=False)
    pv = price_sobol_batch(batch, zs, slots=[0, 1, 2]).numpy()
    ref = [_numpy_replay(zs[k].numpy(), list(p.marginals), p.corr, p.df,
                         p.perf_to_date) for k, p in enumerate(probs)]
    assert np.abs(pv - np.array(ref)).max() < 1e-6
    tiny = price_sobol_batch(batch, zs, slots=[0, 1, 2],
                             max_bytes=1 << 17).numpy()
    assert np.abs(tiny - pv).max() < 1e-9


def test_sobol_anchors_and_guards(flat_marg):
    batch = QuadBatch.from_problems([QuadProblem((flat_marg,) * 3, np.eye(3), 1.0)],
                                    "cpu", scores=False)
    zs = sobol_normals(1, 2 ** 10, device="cpu")
    deep = float(price_sobol_batch(batch, zs, k_lo=0.01, k_hi=0.02)[0])
    far = float(price_sobol_batch(batch, zs, k_lo=50.0, k_hi=50.01)[0])
    assert abs(deep - 0.01 * 1e6) < 1e-3 and abs(far) < 1e-9
    with pytest.raises(ValueError, match="power of two"):
        sobol_normals(1, 1000)
    with pytest.raises(ValueError, match="scores=False"):
        price_quad_batch(batch)


def test_sobol_matches_quadrature_within_noise(flat_marg):
    # the two engines integrate the same model: at 2^13 paths the sampler
    # must sit within a few sampling-sigmas of the deterministic price
    batch = QuadBatch.from_problems([QuadProblem((flat_marg,) * 3, CORR, 0.96)],
                                    "cpu")
    pv_q = float(price_quad_batch(batch, n_outer=32, n_inner=32)[0])
    pv_s = float(price_sobol_batch(batch, sobol_normals(1, 2 ** 13, seed=9,
                                                        device="cpu"))[0])
    assert abs(pv_s - pv_q) < 150.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_sobol_gpu_matches_cpu(flat_marg):
    probs = [QuadProblem((flat_marg,) * 3, CORR, 0.96),
             QuadProblem((flat_marg,) * 3, CORR, 0.99, (1.15, 0.92, 1.05))]
    zc = sobol_normals(2, 2 ** 11, seed=4, device="cpu")
    cpu = price_sobol_batch(QuadBatch.from_problems(probs, "cpu", scores=False),
                            zc).numpy()
    gpu = price_sobol_batch(QuadBatch.from_problems(probs, "cuda", scores=False),
                            zc.to("cuda")).cpu().numpy()
    assert np.abs(cpu - gpu).max() < 1e-6


def test_factory_matches_cpu_oracle():
    # the device-resident table builder must reproduce implied_marginal:
    # same grids, same cardinal splines, same wing taper, same np.gradient
    sd = _flat_sd(n_days=3, vol=0.21)
    sd = dataclasses.replace(
        sd, grids=sd.grids + 0.04 * np.linspace(-1, 1, 13)[None, None, :]
        * np.linspace(1, 0.3, len(sd.ttm_pillars))[None, :, None],
        curve_ttms=[np.array([0.0, 2.0])] * 3,
        curve_lndf=[np.array([0.0, -0.06])] * 3,
        curve_lnfr=[np.array([0.0, 0.05])] * 3)
    fac = MarginalFactory({"A": sd}, device="cpu")
    for tau in (1.0, 0.25, 0.02, 1e-6):
        m = implied_marginal(sd, 1, tau)
        x, cdf, psi = fac.tables("A", 1, np.array([tau]))
        assert np.abs(x[0].numpy() / m.x - 1).max() < 1e-12
        assert np.abs(cdf[0].numpy() - m.cdf).max() < 1e-10
        mid = (m.cdf > 1e-6) & (m.cdf < 1 - 1e-6)
        ref = norm.ppf(np.clip(m.cdf, 1e-16, 1 - 1e-16))
        # cdf diffs ~1e-10 amplify by 1/phi(psi) near the window edge
        assert np.abs(psi[0].numpy() - ref)[mid].max() < 1e-6
    # curves replicate SurfaceData's np.interp of the log-linear nodes
    taus = np.array([0.03, 0.5, 1.0, 1.7, 2.5])
    assert np.allclose(fac.discount("A", 1, taus).numpy(),
                       [sd.discount(1, t) for t in taus], rtol=1e-14)
    assert np.allclose(fac.forward_ratio("A", 1, taus).numpy(),
                       [sd.forward(1, t) / sd.spot[1] for t in taus], rtol=1e-14)
    # recovered mean still matches the forward (the BL consistency check)
    x, cdf, _ = fac.tables("A", 1, np.array([1.0]))
    mean = float(torch.trapezoid(1.0 - cdf[0], x[0]) + x[0, 0])
    assert abs(mean / (sd.forward(1, 1.0) / sd.spot[1]) - 1) < 1e-3


def test_factory_two_tau_and_dgrid():
    # tau_price defaults to the lookup tau; a pillar shock must flow through
    # exactly like shocking the stored grid (the VaR scenario seam)
    sd = _flat_sd(n_days=2, vol=0.2)
    sd = dataclasses.replace(       # give the surface a term structure so a
        sd, grids=sd.grids          # decayed lookup actually moves the vols
        + 0.05 * np.linspace(0, 1, len(sd.ttm_pillars))[None, :, None])
    fac = MarginalFactory({"A": sd}, device="cpu")
    plain = fac.tables("A", 0, np.array([0.5]))
    same = fac.tables("A", 0, np.array([0.5]), tau_price=np.array([0.5]))
    assert torch.equal(plain[1], same[1])
    shock = 0.03 * np.linspace(-1, 1, 13)[None, :] * np.ones((len(sd.ttm_pillars), 1))
    sd_sh = dataclasses.replace(sd, grids=sd.grids + shock[None])
    m = implied_marginal(sd_sh, 0, 0.5)
    x, cdf, _ = fac.tables("A", 0, np.array([0.5]), dgrid=shock)
    assert np.abs(x[0].numpy() / m.x - 1).max() < 1e-12
    assert np.abs(cdf[0].numpy() - m.cdf).max() < 1e-10
    # frozen pricing tau with decayed lookup: vols move but the forward
    # (the distribution mean, set by tau_price) stays put
    roll = fac.tables("A", 0, np.array([0.4]), tau_price=np.array([0.5]))
    mean = lambda tb: float(torch.trapezoid(1.0 - tb[1][0], tb[0][0]) + tb[0][0, 0])
    assert abs(mean(roll) / mean(plain) - 1) < 1e-6
    assert not torch.allclose(roll[1], plain[1])      # different distribution


def test_attribution_flat_world_identities():
    # unchanged flat surfaces, constant spots: every component except time
    # must be *exactly* zero (CRN: identical tables -> identical prices),
    # and the short book earns pure decay
    corr = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    flat = {n: _flat_sd(n_days=5, vol=v, spot=s)
            for n, v, s in zip(("SPX", "SX5E", "HSI"), (0.2, 0.22, 0.25),
                               (100.0, 90.0, 80.0))}
    res = rainbow_attribution(flat, n_paths=256, n_slots=8, device="cpu",
                              corr=corr)
    assert list(res["n_pos"]) == [1, 2, 3, 4]
    zero = ["eq", "eq_delta", "eq_higher", "vol", "vol_spx", "vol_sx5e",
            "vol_hsi", "vol_cross", "fwd", "rate", "cross_sv", "resid"]
    assert np.abs(res[zero].to_numpy()).max() < 1e-9
    assert (res["pl"] > 0).all()
    assert np.allclose(res["pl"], res["time"], atol=1e-9)
    assert np.allclose(res["time"], res["time_roll"] + res["time_other"],
                       atol=1e-9)


def test_attribution_vol_bump_lands_in_vol():
    # a parallel vol bump on day 2 must be booked to vol on the 1->2 pair
    # only, negative for the short book, with eq/fwd untouched
    corr = np.eye(3)
    flat = {n: _flat_sd(n_days=4, vol=0.2, bump_day=2, bump=0.02)
            for n in ("SPX", "SX5E", "HSI")}
    res = rainbow_attribution(flat, n_paths=512, n_slots=8, device="cpu",
                              corr=corr)
    assert abs(res["vol"].iloc[0]) < 1e-9
    assert res["vol"].iloc[1] < -1e3            # short vega loses on the bump
    assert abs(res["vol"].iloc[2]) < 1e-9
    assert np.abs(res[["eq", "fwd", "rate"]].to_numpy()).max() < 1e-9
    singles = res[["vol_spx", "vol_sx5e", "vol_hsi"]].iloc[1]
    assert (singles < 0).all()                  # every surface contributed
    assert abs(res["resid"].iloc[1]) < 0.05 * abs(res["vol"].iloc[1])


@pytest.mark.skipif(not all(p.exists() for p in CSV3),
                    reason="vol surface data not present")
def test_factory_batch_prices_match_from_problems():
    # real surfaces: the factory-built QuadBatch must price like the batch
    # assembled from CPU implied_marginal objects, on both engines
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    names = list(sds)
    spots = spot_panel(sds)
    val = spots.index[-1]
    idx = {n: date_index(sds[n], np.datetime64(val, "D")) for n in names}
    taus = np.array([1.0, 0.4, 0.11])
    g = np.stack([(spots.loc[val] / spots.iloc[-1 - lag]).to_numpy()
                  for lag in (0, 150, 224)])
    probs = [QuadProblem(tuple(implied_marginal(sds[n], idx[n], float(t))
                               for n in names), CORR, 0.97, tuple(gr))
             for t, gr in zip(taus, g)]
    ref = QuadBatch.from_problems(probs, device="cpu")
    fac = MarginalFactory(sds, device="cpu")
    chol = torch.as_tensor(np.linalg.cholesky(CORR), dtype=torch.float64)
    fb = fac.batch([fac.tables(n, idx[n], taus) for n in names],
                   torch.as_tensor(g),
                   torch.full((3,), 0.97, dtype=torch.float64), chol)
    assert np.abs(price_quad_batch(fb).numpy()
                  - price_quad_batch(ref).numpy()).max() < 0.01
    zs = sobol_normals(3, 2 ** 11, seed=2, device="cpu")
    assert np.abs(price_sobol_batch(fb, zs).numpy()
                  - price_sobol_batch(ref, zs).numpy()).max() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_cpu(flat_marg):
    probs = [QuadProblem((flat_marg,) * 3, CORR, 0.96),
             QuadProblem((flat_marg,) * 3, CORR, 0.99, (1.15, 0.92, 1.05))]
    cpu = price_quad_batch(QuadBatch.from_problems(probs, device="cpu"))
    gpu = price_quad_batch(QuadBatch.from_problems(probs, device="cuda"))
    assert np.abs(cpu.numpy() - gpu.cpu().numpy()).max() < 0.01


@pytest.mark.skipif(not all(p.exists() for p in CSV3),
                    reason="vol surface data not present")
def test_real_surface_batch_matches_numpy():
    sds = {p.name.split("_")[0]: load_surfaces(p) for p in CSV3}
    spots = spot_panel(sds)
    val = spots.index[-1]
    corr = historical_corr(spots, 5)
    probs = []
    for lag, tau in ((0, 1.0), (63, 0.75)):
        sale = spots.index[-1 - lag]
        g = tuple((spots.loc[val] / spots.loc[sale]).to_numpy())
        mrg = tuple(implied_marginal(sd, date_index(sd, np.datetime64(val, "D")), tau)
                    for sd in sds.values())
        probs.append(QuadProblem(mrg, corr, 0.96, g))
    batch = QuadBatch.from_problems(probs)      # default device (GPU if there)
    pv = price_quad_batch(batch).cpu().numpy()
    ref = [price_rainbow_quad(list(p.marginals), p.corr, p.df,
                              perf_to_date=p.perf_to_date)["pv"] for p in probs]
    assert np.abs(pv - np.array(ref)).max() < 0.05
