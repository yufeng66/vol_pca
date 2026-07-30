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

## Dataset: `SPX_volSurface 2.csv`

SPX implied volatility surface snapshots, ~55k rows covering 3,142 trading days from 2013-12-30 through 2026-05-12. Each row is one term (expiry) of one day's surface; a day's full surface spans multiple rows sharing the same `AsOfTime`.

Columns:
- Unnamed first column: integer row index.
- `50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140, 150`: implied vols (in %) at moneyness levels expressed as percent of spot (100 = ATM).
- `AsOfTime`: snapshot timestamp (e.g. `2013-12-30T23:59:59Z`), the natural time-series key.
- `Term`: option expiry date; combined with `AsOfTime` gives time-to-maturity.
- `Forward`, `ImpliedSpot`, `DiscountFactor`: per-term forward, spot, and discount factor.
- `DataSourceId`, `IsEndOfDay`, `UnderlyingIndex` (always SPX), `VolSurfaceId`, `VolSurfaceTypeId`: metadata.

Note the filename contains a space — quote it in shell commands. The accompanying `:Zone.Identifier` file is a Windows download artifact and can be ignored.
