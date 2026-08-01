# vol_pca

PCA-factor-based vol risk on SPX implied volatility surfaces, demonstrated on a
test book: every day sell a $1M-notional 1Y call spread (short 100% strike,
long 110%), ~250 spreads at steady state over 2013–2026.

Pipeline (`uv run python scripts/run_analysis.py`):

1. Clean the raw surface CSV into fixed pillar grids (8 TTMs × 13 moneyness).
2. Simulate the book daily; attribute P&L into independent Black-Scholes-input
   components (equity, vol, time, vanna, rate, div), and split the vol
   component exactly into the fixed-strike vol P&L (the risk target) and the
   anticipatable term-roll carry.
3. Bucket each option's vega onto the pillars (exact: the surface lookup —
   bicubic spline on the pillar grid — is linear in pillar vols), fit
   vega-weighted PCA on daily **sticky-strike** surface moves (each pillar
   tracks a fixed strike re-anchored at the previous close, sampled with the
   same bicubic interpolant; pillars weighted by the book's average absolute
   dollar vega), and estimate the fixed-strike vol P&L as factor exposure ×
   daily factor movement, paired with the plain BS delta.

Headline result: 3 vega-weighted sticky-strike factors explain the book's
fixed-strike vol P&L (daily std $70k) with R² 0.81 (k=10: 0.93; all-factor
linear ceiling 0.985). Unweighted (correlation) PCA manages only R² 0.65 at
k=3 — it spends factor capacity on far-wing short-term smile moves the book
barely prices off. The factor scores are nearly spot-orthogonal (PC1 |corr|
with spot returns 0.34 vs 0.86 for sticky-moneyness factors), which keeps
factor shocks meaningful as standalone scenarios.

Next: historical VaR comparing the PCA-factor approach to full revaluation.

The vol surface data file is deliberately gitignored — never commit it.

`attribution.ipynb` inspects daily P&L attribution with independent
Black-Scholes-input bumps (equity delta/gamma/higher, implied vol, rate,
dividend, time, vanna); residual daily std is ~0.8% of total P&L std.

`pca_factors.ipynb` illustrates the headline sticky-strike factor model:
factor shape heatmaps and tables, cumulative factor history, estimate quality
against the fixed-strike vol target, and the vol/roll split (roll is
anticipatable carry, forecastable from the previous close).

`roll_in_pca.ipynb` folds the term roll into the factor basis (each pillar
tracks a fixed option: fixed strike and expiry), so the factor estimates add
up to the attribution vol component with no separate carry line — the
loadings are identical to the ex-roll basis, the roll just moves into the
factor mean. It also answers "how many factors": k=3 is fine for exposure
reporting (R² 0.82), but stress days need k≈10 (big-move R² 0.66 → 0.83).

`sticky_strike.ipynb` is the comparison that motivated the framework choice:
sticky-strike vs sticky-moneyness coordinates, same vega-weighted fit.
Near-ATM daily moves halve in strike coordinates, PC1 is far less
spot-entangled, and — sampled with the smooth pricing interpolant — the
sticky-strike model beats moneyness-PCA + handed-over slide at every factor
count with a higher linear ceiling (under bilinear sampling it trailed at
every k: kinked bracket-crossing samples read as noise, and interpolation
bias faked ≈ +$9k/day of carry).

## Dev

- `uv sync` — set up the environment
- `uv run pytest` — run tests
