# vol_pca

PCA-factor-based vol risk on SPX implied volatility surfaces, demonstrated on a
test book: every day sell a $1M-notional 1Y call spread (short 100% strike,
long 110%), ~250 spreads at steady state over 2013–2026.

Pipeline (`uv run python scripts/run_analysis.py`):

1. Clean the raw surface CSV into fixed pillar grids (8 TTMs × 13 moneyness).
2. Simulate the book daily; attribute P&L into theta, smile delta, gamma,
   vega (full per-option revaluation), vanna, residual.
3. Bucket each option's vega onto the pillars (exact, by linear-interpolation
   weights), fit correlation-PCA on daily surface moves, and estimate vega P&L
   as factor exposure × daily factor movement.

Headline result: 3 PCA factors reproduce the full per-option vega revaluation
with correlation 0.95 (R² 0.91); the all-factor linear limit is 0.996.

Next: historical VaR comparing the PCA-factor approach to full revaluation.

The vol surface data file is deliberately gitignored — never commit it.

## Dev

- `uv sync` — set up the environment
- `uv run pytest` — run tests
