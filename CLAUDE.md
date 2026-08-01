# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Setup

PCA analysis of SPX implied volatility surfaces. Python project managed with `uv` (Python 3.14, pinned in `.python-version`); the setup mirrors the `~/vix_refactor` project minus its PyTorch/trading dependencies.

Commands:
- `uv sync` — create/update `.venv` from `pyproject.toml` (dev group including pytest installs by default).
- `uv run python <script>` — run a script inside the venv.
- `uv run pytest` — run tests (configured for `tests/`, with repo root on `pythonpath` so in-tree packages import without installing).
- `uv run pytest tests/test_foo.py::test_bar` — run a single test.
- `uv run jupyter notebook` — notebooks.
- `uv add <pkg>` — add a dependency.

Main deps: pandas, scikit-learn, scipy, matplotlib, notebook.

Run the full pipeline: `uv run python scripts/run_analysis.py` (~10s; writes `data/daily_results.csv`, `data/attribution_daily.csv` and `data/factor_model.npz`, all gitignored).

## Project Goal & Status

Demonstrate PCA-factor-based vol risk on a test book: every day sell a $1M-notional 1Y SPX call spread (short 100%, long 110% strike), giving a steady-state book of ~250 spreads. Attribute daily P&L, compute the book's exposure to each PCA factor of surface moves, and estimate vol P&L as exposure × daily factor movement. **Done.** The headline framework is **sticky-strike** (user decision 2026-08-01): factors fit to sticky-strike pillar changes, targeting the book's fixed-strike vol P&L ex term-roll carry (attribution `vol − vol_roll`; total +$15.5M, daily std $70k), paired with the plain BS delta. Presentation rule from the user: do **not** quote the total of `vol_surface`/`pl_vega_full` as a headline number — it and the smile slide are large offsetting coordinate artifacts of the moneyness split; quote the fixed-strike vol P&L instead. Next step per the roadmap: historical VaR comparing the PCA-factor approach against full revaluation.

## Architecture (`vol_pca/` package)

The load-bearing design choice: all pricing looks vols up on a fixed per-date **pillar grid** (8 TTM pillars × the 13 quoted moneyness columns, `surface.py`) with **bicubic-spline** interpolation (cubic per axis, not-a-knot ends; `grid_lookup(..., interp="linear")` keeps the old bilinear lookup). Spline interpolation with fixed knots is still *linear in the pillar values*, so every queried vol is an exact linear combination of pillar vols — the weights are dense cardinal-basis products (global, can be negative) instead of bilinear's 4 local ones — and scattering each option's vega onto the pillars via those weights keeps the exact identity `pl_vega_lin == bucket_vega · Δσ` (asserted in tests to 1e-9). Caveat of the dense weights: bucketed vega gets oscillatory side-lobes (gross |bucket vega| ≈ +34% vs bilinear at unchanged net), so it is a factor-model input, not a per-pillar hedging report. The identity holds for sticky-moneyness Δσ (pillar-point differences); the headline sticky-strike Δσ samples between pillars, so its all-factor estimate carries a small extra reconstruction error on top of vega convexity (all-factor R² 0.985 vs the target).

- `data.py` — loads/cleans the CSV (EOD filter, dup drop, corrupted-upper-wing repair) into `SurfaceData`: per-date pillar grids plus log-linear discount/forward curves.
- `pricing.py` — Black-76 on the forward; spot greeks assume fixed F/S. Returns (price, delta, gamma, vega, vanna) with `greeks=True`.
- `book.py` — daily simulation and sequential attribution against the previous close. Delta is the **smile delta** (surface is moneyness-quoted, so a fixed strike slides along the smile as spot moves) — this sequential view stays sticky-moneyness; the headline factor framework instead pairs sticky-strike factors with the plain BS delta. Vanna is explicit because SPX spot-vol anticorrelation makes it a large systematic drag (−$32M over the sample).
- `attribution.py` — independent (non-waterfall) Black-Scholes-input attribution: each component is a single-input revaluation from the same previous-close base (equity split delta/gamma/higher, the option's own implied-vol change, zero rate r, implied dividend yield q, time, exact vanna cross). The time component is split by the BS PDE into `time_funding` (financing premium + carrying the delta hedge) vs pure gamma theta, so a delta-hedged view = `pl − eq_delta − time_funding`. The vol component splits exactly (telescoping full revals) into `vol_surface` (≡ book.py's `pl_vega_full`) + `vol_roll` (term roll; `vol_carry_ex` is its ex-ante version off the previous close — anticipatable carry, corr 0.98) + `vol_slide` (spot-driven smile slide). The headline PCA target is `vol − vol_roll` (fixed-strike vol ex anticipatable carry) = `vol_surface + vol_slide`. Daily totals match `simulate_book` exactly (tested); residual std is ~0.8% of P&L std. Inspected in `attribution.ipynb` (root; executed outputs committed); the factor model and this reconciliation are illustrated in `pca_factors.ipynb`. Note the q backed out of forward-vs-spot spikes on some stress dates (e.g. 2018-02-06) — a data snap inconsistency the `div` component absorbs and flags.
- `factors.py` — PCA on daily pillar Δσ with a per-pillar weight transform (`fit_pca(weights=...)`): `"cov"` (plain), `"corr"` (standardized), or a custom array. The **headline model** is vega-weighted (`w_p` = book's average |bucket vega|) PCA on **sticky-strike** pillar changes from `sticky_strike_dsigma(sd)` (new surface sampled at m·S₀/S₁ with the pricing interpolant, `interp="cubic"` default with `"linear"`/`"pchip"` variants: surface move + smile slide at fixed TTM, roll excluded; the 50/150 edge columns clamp). Target: fixed-strike vol P&L ex roll. k=3 R² by weighting: vega 0.811, corr 0.646, cov 0.474 (vega k=10: 0.934; all-factor ceiling 0.985). Exposures `E_k = (bucket_vega/w)·L_k`, scores `f_k = L_k·((Δσ−μ)∘w)`, estimate `= bucket_vega·μ + Σ E_k f_k`. Sticky-strike factors are far less spot-entangled than sticky-moneyness ones (PC1 |corr| with spot returns 0.34 vs 0.86) and pair with the plain BS delta. `include_roll=True` folds the term roll in too (pillar tracks a fixed option: strike and expiry, day-t lookup at τ−dt), making the basis complete against the full attribution `vol` (all-factor corr 0.994, cumulative within $0.5M with no carry line; loadings identical to ex-roll — roll just moves into the factor mean, ≈ −$5.3k/day; shortest pillar clamps at the 0.08 TTM edge); explored in `roll_in_pca.ipynb`, which also shows k=3 suffices for exposure reporting (R² 0.82) but stress days need k≈10 (big-move R² 0.66 → 0.83, p99 error $115k → $57k). The comparison that motivated the switch is `sticky_strike.ipynb`: with cubic sampling sticky-strike beats moneyness-PCA + slide at every k (k=3 R² 0.81 vs 0.77, ceiling 0.985 vs 0.955 — linearizing the net $70k move beats linearizing the two big offsetting pieces); under the old bilinear sampling the ranking was reversed — kinked bracket-crossing samples read as pillar noise and faked a ~+$9k/day carry drift in the factor mean (linear interpolation overestimates the convex smile between nodes). The factor machinery is illustrated in `pca_factors.ipynb`.

Conventions: vols as decimals internally (quoted % / 100), TTM in ACT/365, moneyness in %-of-spot units (50–150). Days after the two month-long data gaps (Oct 2014, Oct–Nov 2015; `gap_days > 7`) are excluded from PCA fitting and metrics.

## Dataset: `SPX_volSurface 2.csv`

SPX implied volatility surface snapshots, ~55k rows covering 3,141 trading days from 2013-12-30 through 2026-05-12 (after cleaning). Each row is one term (expiry) of one day's surface; a day's full surface spans multiple rows sharing the same `AsOfTime`. Known quirks handled by the loader: 17 non-EOD rows, 1,462 exact-duplicate (date, term) rows, a corrupted far-call wing on a few Feb 2016 short-term rows (140-col halved, 150-col zeroed), and the two month-long gaps noted above.

Columns:
- Unnamed first column: integer row index.
- `50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140, 150`: implied vols (in %) at moneyness levels expressed as percent of spot (100 = ATM).
- `AsOfTime`: snapshot timestamp (e.g. `2013-12-30T23:59:59Z`), the natural time-series key.
- `Term`: option expiry date; combined with `AsOfTime` gives time-to-maturity.
- `Forward`, `ImpliedSpot`, `DiscountFactor`: per-term forward, spot, and discount factor.
- `DataSourceId`, `IsEndOfDay`, `UnderlyingIndex` (always SPX), `VolSurfaceId`, `VolSurfaceTypeId`: metadata.

Note the filename contains a space — quote it in shell commands. The accompanying `:Zone.Identifier` file is a Windows download artifact and can be ignored.
