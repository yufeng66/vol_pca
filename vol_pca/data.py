"""Load and clean the SPX vol surface CSV into per-date pillar surfaces."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vol_pca.surface import MONEYNESS, TTM_PILLARS, build_pillar_grid

_MON_COLS = [str(int(m)) for m in MONEYNESS]
_ATM_IDX = int(np.where(MONEYNESS == 100)[0][0])


@dataclass
class SurfaceData:
    dates: np.ndarray          # (n,) datetime64[D], sorted
    spot: np.ndarray           # (n,)
    grids: np.ndarray          # (n, n_ttm, n_mon) vols as decimals
    curve_ttms: list           # per date: (k,) TTMs incl. leading 0
    curve_lndf: list           # per date: ln DF at curve_ttms (0 at ttm=0)
    curve_lnfr: list           # per date: ln(F/spot) at curve_ttms (0 at ttm=0)
    n_dupes_dropped: int = 0
    n_wing_repaired: int = 0
    ttm_pillars: np.ndarray = field(default_factory=lambda: TTM_PILLARS.copy())
    moneyness: np.ndarray = field(default_factory=lambda: MONEYNESS.copy())

    def __len__(self):
        return len(self.dates)

    def discount(self, i, ttm):
        return np.exp(np.interp(ttm, self.curve_ttms[i], self.curve_lndf[i]))

    def forward(self, i, ttm):
        return self.spot[i] * np.exp(np.interp(ttm, self.curve_ttms[i], self.curve_lnfr[i]))


def _repair_upper_wing(vols):
    """Fix corrupted far-call-wing quotes (observed: 140 col exactly halved,
    150 col zeroed on a few 2016 dates). Walking outward from ATM, a vol
    dropping below 60% of its inner neighbour is replaced by that neighbour."""
    repaired = np.zeros(len(vols), dtype=bool)
    for j in range(_ATM_IDX + 1, vols.shape[1]):
        bad = vols[:, j] < 0.6 * vols[:, j - 1]
        vols[bad, j] = vols[bad, j - 1]
        repaired |= bad
    return int(repaired.sum())


def load_surfaces(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["IsEndOfDay"]].copy()
    df["asof"] = pd.to_datetime(df["AsOfTime"]).dt.tz_localize(None).dt.normalize()
    df["term"] = pd.to_datetime(df["Term"]).dt.tz_localize(None)
    n0 = len(df)
    df = df.drop_duplicates(["asof", "term"], keep="last")
    n_dupes = n0 - len(df)
    df["ttm"] = (df["term"] - df["asof"]).dt.days / 365.0
    df = df[df["ttm"] > 0].sort_values(["asof", "ttm"])

    vols = df[_MON_COLS].to_numpy(dtype=float).copy()
    n_repaired = _repair_upper_wing(vols)
    df[_MON_COLS] = vols

    dates, spot, grids = [], [], []
    curve_ttms, curve_lndf, curve_lnfr = [], [], []
    for asof, day in df.groupby("asof", sort=True):
        s = day["ImpliedSpot"].iloc[0]
        ttms = day["ttm"].to_numpy()
        dates.append(asof)
        spot.append(s)
        grids.append(build_pillar_grid(ttms, day[_MON_COLS].to_numpy() / 100.0))
        curve_ttms.append(np.concatenate([[0.0], ttms]))
        curve_lndf.append(np.concatenate([[0.0], np.log(day["DiscountFactor"].to_numpy())]))
        curve_lnfr.append(np.concatenate([[0.0], np.log(day["Forward"].to_numpy() / s)]))

    return SurfaceData(
        dates=np.array(dates, dtype="datetime64[D]"),
        spot=np.array(spot),
        grids=np.array(grids),
        curve_ttms=curve_ttms,
        curve_lndf=curve_lndf,
        curve_lnfr=curve_lnfr,
        n_dupes_dropped=n_dupes,
        n_wing_repaired=n_repaired,
    )
