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

Run the full pipeline: `uv run python scripts/run_analysis.py` (~5s; writes `data/daily_results.csv` and `data/factor_model.npz`, both gitignored).

## Project Goal & Status

Demonstrate PCA-factor-based vol risk on a test book: every day sell a $1M-notional 1Y SPX call spread (short 100%, long 110% strike), giving a steady-state book of ~250 spreads. Attribute daily P&L (theta / smile delta / gamma / vega-by-full-revaluation / vanna / residual), compute the book's exposure to each PCA factor of surface moves, and estimate vega P&L as exposure × daily factor movement. **Done.** Next step per the roadmap: historical VaR comparing the PCA-factor approach against full revaluation.

## Architecture (`vol_pca/` package)

The load-bearing design choice: all pricing looks vols up on a fixed per-date **pillar grid** (8 TTM pillars × the 13 quoted moneyness columns, `surface.py`) with **linear** interpolation in both axes. Linearity makes every queried vol an exact linear combination of pillar vols, so scattering each option's vega onto the pillars via its interpolation weights gives a bucketed vega vector with the exact identity `pl_vega_lin == bucket_vega · Δσ` (asserted in tests to 1e-9). PCA truncation is then the *only* approximation between the factor estimate and the linear vol P&L; the remaining gap to full revaluation is vega convexity (material for spreads: net vega is a small difference of two large leg vegas).

- `data.py` — loads/cleans the CSV (EOD filter, dup drop, corrupted-upper-wing repair) into `SurfaceData`: per-date pillar grids plus log-linear discount/forward curves.
- `pricing.py` — Black-76 on the forward; spot greeks assume fixed F/S. Returns (price, delta, gamma, vega, vanna) with `greeks=True`.
- `book.py` — daily simulation and sequential attribution against the previous close. Delta is the **smile delta** (surface is moneyness-quoted, so a fixed strike slides along the smile as spot moves); vanna is explicit because SPX spot-vol anticorrelation makes it a large systematic drag (−$32M over the sample vs +$25M vega).
- `attribution.py` — independent (non-waterfall) Black-Scholes-input attribution: each component is a single-input revaluation from the same previous-close base (equity split delta/gamma/higher, the option's own implied-vol change, zero rate r, implied dividend yield q, time, exact vanna cross). The time component is split by the BS PDE into `time_funding` (financing premium + carrying the delta hedge) vs pure gamma theta, so a delta-hedged view = `pl − eq_delta − time_funding`. The vol component splits exactly (telescoping full revals) into `vol_surface` (≡ book.py's `pl_vega_full`, the PCA target) + `vol_roll` (term roll; `vol_carry_ex` is its ex-ante version off the previous close — anticipatable carry, corr 0.98) + `vol_slide` (spot-driven smile slide). Daily totals match `simulate_book` exactly (tested); residual std is ~0.8% of P&L std. Inspected in `attribution.ipynb` (root; executed outputs committed); the factor model and this reconciliation are illustrated in `pca_factors.ipynb`. Note the q backed out of forward-vs-spot spikes on some stress dates (e.g. 2018-02-06) — a data snap inconsistency the `div` component absorbs and flags.
- `factors.py` — PCA on daily pillar Δσ with a per-pillar weight transform (`fit_pca(weights=...)`): `"cov"` (plain), `"corr"` (standardized), or a custom array. The headline model is **vega-weighted** (`w_p` = book's average |bucket vega|): factor capacity goes to dollar-relevant movement instead of far-wing/short-TTM smile noise the book barely prices off. k=3 R² by weighting: vega 0.957, corr 0.909, cov 0.884. Exposures `E_k = (bucket_vega/w)·L_k`, scores `f_k = L_k·((Δσ−μ)∘w)`, estimate `= bucket_vega·μ + Σ E_k f_k`. `sticky_strike_dsigma(sd)` builds the alternative **sticky-strike** pillar changes (new surface sampled at m·S₀/S₁: surface move + smile slide, roll excluded; the exact bucketed-vega identity is lost and the 50/150 edge columns clamp). Compared in `sticky_strike.ipynb`: near-ATM moves are ~½ the fixed-moneyness ones and PC1 is far less spot-correlated (|corr| 0.35 vs 0.86, useful for parametric stress design), but as an estimator of the fixed-strike vol target (`vol − vol_roll`) it trails moneyness-PCA + slide at every k (k=3 R² 0.76 vs 0.80; all-factor ceiling 0.92 vs 0.96, the identity loss concentrating on big spot-move days) — so the headline model stays sticky-moneyness paired with the smile delta.

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
