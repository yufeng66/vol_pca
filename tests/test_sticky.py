import pathlib

import numpy as np
import pytest

from tests.test_book import _flat_sd
from vol_pca.attribution import independent_attribution
from vol_pca.book import simulate_book
from vol_pca.data import SurfaceData, load_surfaces
from vol_pca.factors import sticky_strike_dsigma
from vol_pca.surface import MONEYNESS, TTM_PILLARS

CSV = pathlib.Path(__file__).resolve().parents[1] / "SPX_volSurface.csv"


@pytest.mark.parametrize("interp", ["linear", "cubic", "pchip"])
def test_flat_smile_matches_fixed_moneyness(interp):
    # with no smile, rescaling the lookup moneyness changes nothing: the
    # sticky-strike changes equal the plain grid differences even as spot moves
    sd = _flat_sd(bump_day=2, bump=0.01)
    sd.spot[:] = [100.0, 103.0, 98.0, 101.0]
    ds = sticky_strike_dsigma(sd, interp=interp)
    flat = sd.grids.reshape(len(sd.dates), -1)
    assert np.allclose(ds, flat[1:] - flat[:-1])


@pytest.mark.parametrize("interp", ["linear", "cubic", "pchip"])
def test_linear_smile_gives_pure_slide(interp):
    sd = _flat_sd(n_days=2)
    slope = 0.001  # vol per moneyness point
    sd.grids[:] = 0.2 + slope * (MONEYNESS - 100.0)
    sd.spot[:] = [100.0, 101.0]
    ds = sticky_strike_dsigma(sd, interp=interp)[0].reshape(len(TTM_PILLARS),
                                                            len(MONEYNESS))
    c = 100.0 / 101.0
    # surface unchanged -> the sticky-strike change is the slide along the
    # smile; a linear smile is reproduced exactly by all three interpolants
    # on interior columns
    assert np.allclose(ds[:, 1:], slope * MONEYNESS[1:] * (c - 1.0))
    # the 50-column query (49.5) clamps at the grid edge: no change
    assert np.allclose(ds[:, 0], 0.0)


def test_unknown_interp_raises():
    sd = _flat_sd(n_days=2)
    with pytest.raises(ValueError, match="unknown interp"):
        sticky_strike_dsigma(sd, interp="quadratic")
    with pytest.raises(ValueError, match="include_roll"):
        sticky_strike_dsigma(sd, interp="pchip", include_roll=True)


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_include_roll_gives_pure_roll(interp):
    # static surface with a linear term structure, no spot move: the
    # fixed-option change is the pure roll -slope x dt; the shortest pillar
    # clamps at the 0.08 TTM grid edge so it sees no change
    sd = _flat_sd(n_days=2)
    slope = 0.05  # vol per year of TTM
    sd.grids[:] = 0.2 + slope * TTM_PILLARS[:, None] + 0.0 * MONEYNESS
    ds = sticky_strike_dsigma(sd, interp=interp, include_roll=True)
    ds = ds[0].reshape(len(TTM_PILLARS), len(MONEYNESS))
    dt = 1.0 / 365.0
    assert np.allclose(ds[1:], -slope * dt)
    assert np.allclose(ds[0], 0.0)
    # without the roll there is no change at all
    assert np.allclose(sticky_strike_dsigma(sd, interp=interp), 0.0)


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_include_roll_tracks_full_vol_component():
    sd = load_surfaces(CSV)
    m = 120
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    _, bucket_vega, _ = simulate_book(small)
    res, _ = independent_attribution(small)
    lin = (bucket_vega * sticky_strike_dsigma(small, include_roll=True)).sum(axis=1)
    assert np.corrcoef(lin, res["vol"].to_numpy())[0, 1] > 0.9


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_real_data_tracks_fixed_strike_vol_pl():
    sd = load_surfaces(CSV)
    m = 120
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    _, bucket_vega, dsigma = simulate_book(small)
    res, _ = independent_attribution(small)
    ds = sticky_strike_dsigma(small)
    # linear bucketed estimate tracks the attribution's fixed-strike vol
    # change ex term roll (identity is only approximate in these coordinates)
    lin = (bucket_vega * ds).sum(axis=1)
    target = (res["vol"] - res["vol_roll"]).to_numpy()
    assert np.corrcoef(lin, target)[0, 1] > 0.9
    # near the money, sticky-strike moves are roughly half the fixed-moneyness
    # ones (skew slide cancels most of the level move)
    atm = list(MONEYNESS).index(100.0)
    shape = (-1, len(TTM_PILLARS), len(MONEYNESS))
    assert (ds.reshape(shape)[:, :, atm].std()
            < 0.8 * dsigma.reshape(shape)[:, :, atm].std())
