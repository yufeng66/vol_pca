import pathlib

import numpy as np
import pytest

from tests.test_book import _flat_sd
from vol_pca.attribution import independent_attribution
from vol_pca.book import simulate_book
from vol_pca.data import SurfaceData, load_surfaces

CSV = pathlib.Path(__file__).resolve().parents[1] / "SPX_volSurface.csv"


def test_static_world_only_time():
    res, _ = independent_attribution(_flat_sd())
    for c in ["equity", "vol", "rate", "div", "vanna"]:
        assert np.allclose(res[c], 0), c
    assert np.allclose(res["pl"], res["time"] + res["resid"])
    assert (res["resid"].abs() < 1e-6 * res["time"].abs().max()).all()
    # r = q = 0 in the synthetic world -> no funding component of theta
    assert np.allclose(res["time_funding"], 0)


def test_vol_bump_lands_in_vol_component():
    res, _ = independent_attribution(_flat_sd(bump_day=2, bump=0.01))
    day = res.iloc[1]
    assert day["vol"] != 0
    assert abs(day["resid"]) < 0.05 * abs(day["vol"])
    other = res.drop(res.index[1])
    assert np.allclose(other["vol"], 0)


def test_components_sum_to_pl_by_construction():
    res, _ = independent_attribution(_flat_sd(n_days=5, bump_day=2, bump=0.004))
    parts = res[["equity", "vol", "rate", "div", "time", "vanna", "resid"]].sum(axis=1)
    assert np.allclose(parts, res["pl"])
    assert np.allclose(res["equity"],
                       res["eq_delta"] + res["eq_gamma"] + res["eq_higher"])


@pytest.mark.skipif(not CSV.exists(), reason="vol surface data not present")
def test_real_data_matches_book_pl_and_small_residual():
    sd = load_surfaces(CSV)
    m = 120
    small = SurfaceData(dates=sd.dates[:m], spot=sd.spot[:m], grids=sd.grids[:m],
                        curve_ttms=sd.curve_ttms[:m], curve_lndf=sd.curve_lndf[:m],
                        curve_lnfr=sd.curve_lnfr[:m])
    res, details = independent_attribution(small, detail_dates=[sd.dates[50]])
    book, _, _ = simulate_book(small)
    assert np.allclose(res["pl"].to_numpy(), book["pl"].to_numpy(), atol=1e-6)
    assert res["resid"].std() < 0.05 * res["pl"].std()
    # vol splits telescope exactly; fixed-point piece is book.py's vega target
    # (tiny tolerance: book.py includes settling legs, the split masks them)
    split = res[["vol_surface", "vol_roll", "vol_slide"]].sum(axis=1)
    assert np.allclose(split, res["vol"], atol=1e-8)
    assert np.allclose(res["vol_surface"], book["pl_vega_full"], atol=100.0)
    det = details[list(details)[0]]
    assert np.allclose(det["pl"], det[["equity", "vol", "rate", "div", "time",
                                       "vanna", "resid"]].sum(axis=1))
