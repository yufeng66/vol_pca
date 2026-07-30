# vol_pca

PCA-factor-based vol risk on SPX implied volatility surfaces, demonstrated on a
test book: every day sell a $1M-notional 1Y call spread (short 100% strike,
long 110%), ~250 spreads at steady state over 2013–2026.

Pipeline (`uv run python scripts/run_analysis.py`):

1. Clean the raw surface CSV into fixed pillar grids (8 TTMs × 13 moneyness).
2. Simulate the book daily; attribute P&L into theta, smile delta, gamma,
   vega (full per-option revaluation), vanna, residual.
3. Bucket each option's vega onto the pillars (exact, by linear-interpolation
   weights), fit vega-weighted PCA on daily surface moves (pillars weighted by
   the book's average absolute dollar vega), and estimate vega P&L as factor
   exposure × daily factor movement.

Headline result: 3 vega-weighted PCA factors reproduce the full per-option
vega revaluation with correlation 0.98 (R² 0.96); the all-factor linear limit
is 0.996. Unweighted (correlation) PCA manages only R² 0.91 at k=3 — it spends
factor capacity on far-wing short-term smile moves the book barely prices off.

Next: historical VaR comparing the PCA-factor approach to full revaluation.

The vol surface data file is deliberately gitignored — never commit it.

`attribution.ipynb` inspects daily P&L attribution with independent
Black-Scholes-input bumps (equity delta/gamma/higher, implied vol, rate,
dividend, time, vanna); residual daily std is ~0.8% of total P&L std.

`pca_factors.ipynb` illustrates the factor model: factor shape heatmaps and
tables, cumulative factor history, estimate-vs-full-revaluation quality, and
an exact reconciliation of the attribution vol component with the PCA vega
target (vol = surface + roll + slide; roll is anticipatable carry, slide is
spot-driven).

## Dev

- `uv sync` — set up the environment
- `uv run pytest` — run tests
