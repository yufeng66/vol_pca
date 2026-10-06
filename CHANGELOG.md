# Changelog

Research log of the cheap-VaR project — every idea tried, newest first.
Convention (user, 2026-08-06): each entry leads with a plain-language bullet
— the question asked, the method, the conclusion — for human readers; the
`Record:` bullets underneath are the dense trail (functions, numbers, caches,
pitfalls) for future AI sessions. Operational defaults and binding
conventions live in CLAUDE.md; this file holds the history and the evidence.
Add an entry here for every substantive experiment, including (especially)
negative results. Reconstructed 2026-08-06 from CLAUDE.md, session memory
and git history.

## 2026-10-05 — Regression (minimum-variance) delta on the ATM book, exact: lowest daily variance, worse drift, no β horizon dominates sticky strike

- Question (user): build the alternative delta — bump the equity and, when
  you do, co-bump every point of the implied-vol surface by its in-sample
  regression slope on that equity's return — and measure it against the
  current sticky-strike bump delta. Method: the three frameworks are one
  central 1% bump with three assumed surface responses β(τ, m) fed through
  the factory's additive `dgrid` seam — β = 0 (tables frozen, sticky
  moneyness), β = m·∂σ/∂m (the strike relabeling, sticky strike) and
  β = the per-pillar OLS slope of fixed-moneyness changes on the index's
  own return (`spot_vol_betas`, in sample on the joint-date axis, horizons
  1/5/20/60 days + a |ret| ≥ 1% variant). New driver
  `rainbow_regression_delta` (`rainbow_torch.py`), one GPU pass over the
  history for all seven deltas plus consistent eq/vol/cross recuts for
  three of them; notebook `rainbow_regression_delta.ipynb`. Conclusion:
  **the exact result lands where the cache-only proxy (previous entry)
  predicted.** (1) The data's β is only ~10% stronger than the sticky-strike
  β at the money (SPX 1Y ATM −0.42 vs −0.37 vol pts per 1%, R² 0.72), flat
  in moneyness where the relabeling's fades and turns positive in the call
  wing, and nearly horizon-independent (60d: −0.37 = the sticky-strike
  value); sticky moneyness (β = 0) is the outlier framework. (2) The
  regression delta sits on the far side of sticky strike from sticky
  moneyness (mean SPX sold-book delta −\$736k / −\$761k / −\$861k per 1%)
  and behaves like the minimum-variance delta it is: desk std \$182k →
  \$177k (−2.6%; ex COVID −3.3%), skew −4.0 → −3.5, COVID −\$10.8M →
  −\$10.0M at the same −\$2.2M worst day, corr(hedged, SPX) 0.33 → 0.15 —
  and the drift worsens, −\$11.3M vs −\$5.8M total, −\$1.3M vs +\$5.0M ex
  COVID, worse in every rally year, better only in 2022. (3) The horizon
  ladder is monotone toward sticky strike (std 177 → 179 → 179 → 188k,
  total −11.3 → −10.7 → −10.5 → −8.6M) and never reaches it: at 60 days
  the std is already worse than SS while the total is still \$2.8M behind,
  so **no regression horizon dominates sticky strike**; the frontier is
  daily-β (variance) — sticky strike — sticky moneyness (drift: +\$6.2M,
  \$311k std, corr 0.68; a market position). (4) The regression cut's vol
  line is the surface move orthogonal to spot and it trends *harder*
  (−\$31.0M; −\$25.0M ex COVID vs −\$15.9M at fixed strike) because the
  response removed — vol falls on up days — was a rally gain for the short
  book. (5) The big-move β is indistinguishable from the daily β (OLS
  already weights big days); the redistribution is explicit in the
  return-bucket table — moderate down days turn from −\$8.1M into +\$1.4M,
  (+1%, +2%] days from +\$1.9M into −\$12.2M. In-sample βs are the best
  case for every variance number.
- Record: `rainbow.spot_vol_betas(sd, dates, horizon, min_abs_ret)` →
  `SpotVolBetas(beta, alpha, r2, n, horizon)`; `rainbow.sticky_strike_beta`;
  `rainbow_torch.rainbow_regression_delta(sds, betas, recut, ...)` (labels
  `sm`/`ss` built in; ± pair = one build on a doubled tau with a (2B, 8, 13)
  dgrid; six bumps of a label = one `_cat_batches` pricing with repeated
  slots). Identities tested in `tests/test_rainbow_regdelta.py`: `ss` ≡
  `rainbow_attribution_ss` delta/eq/vol/cross to fp, `sm` recut ≡
  `rainbow_attribution` eq/vol/cross to fp, β = 0 ≡ frozen-table bump, flat
  world collapses every label, `sticky_strike_beta` through the seam
  recovers the SS delta to <5% of the SS–SM gap (planted-slope recovery for
  the regression). Caches `data/rainbow_regdelta.csv` +
  `data/rainbow_regdelta_betas.npz` via `scripts/run_rainbow_regdelta.py`
  (1,968 dates in 1,093 s). **Pitfall**: the mixed B / 2B / 6B batches
  fragment PyTorch's caching allocator — the first run crawled at 3.4 s/date
  with reserved memory pinned at the 8GB ceiling (7× the cost model);
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + chunked driver calls
  with `torch.cuda.empty_cache()` between them restored 0.55 s/date. The
  frozen-table bump reproduces the autograd SM delta's P&L to \$0.03M on
  \$6.2M. Hedged yearly \$M (SS / h1 / SM): 2021 1.37 / 0.61 / 5.99, 2022
  −0.14 / 0.71 / −5.57, 2024 0.26 / −0.86 / 3.38, 2025 −1.39 / −2.32 / 1.87.

## 2026-10-05 — Three deltas on the ATM book: sticky-strike vs sticky-moneyness vs regression (minimum-variance) delta, from the two attribution caches

- Question (user): would a bump delta that co-moves the vol surface by its
  empirical correlation with spot (the Hull–White minimum-variance delta in
  bump form) be a better hedge for the ATM book? Method: cache-only proxy —
  the SS desk-hedged series (`pl − eq_delta_SS − theta_value − theta_delta`),
  the same with the SM autograd `eq_delta` from `rainbow_attribution_sm_atm.csv`
  (pl identical to fp), and a regression delta approximated by removing the
  trailing-250d beta of the SS-hedged P&L to the three index returns
  (in-sample version = ceiling). Conclusion: **no single delta wins both
  objectives.** Regression delta: zero market beta, std −4% out of sample
  (−8% in-sample ceiling), better tail (COVID −8.5M vs −10.8M, skew −3.3 vs
  −3.8), but a *worse* drift — −11.1M vs −6.1M on the same dates (−12.7M
  in-sample), losing to SS in every rally year — because the daily beta is
  the mean-reverting level anticorrelation (hedged book +\$28k per 1% SPX),
  so the MV delta holds *less* stock than BS, the opposite direction from the
  slide. SM delta: fixes the drift (+6.3M total, +25.0M ex COVID, better in
  every rally year) but std \$311k = 1.7× SS, COVID −18.7M, worst day −4.3M,
  corr(hedged, ret_spx) +0.68 — a long-market tilt that paid in a bull
  market, not a hedge; in 2022 it lost −5.6M vs SS −0.1M. Ordering of the
  call delta: MV < BS(SS) < SM; the choice is a horizon/objective question
  (daily variance vs cumulative drift). Re-measures the spread-book
  SS-hedges-best verdict for the ATM book: SS still wins on daily std and
  tail; SM wins only on drift. Rolling betas are unstable (10–90%: −13k…+52k
  per 1% SPX), and the level piece is convex in returns, so a linear β
  under-hedges crashes.
- Record: approximations only — a real regression delta needs per-pillar
  β(m,τ) from rolling fixed-moneyness pillar Δσ regressions, co-bumped via
  the `dgrid` seam with tables rebuilt at the bumped spot (SM bump +
  correction; algebraically the same as the fixed-strike route). The
  SM-hedged theta pieces reuse the SS `theta_delta` (SS deltas × forward
  decay) — a small inconsistency, −0.06M scale. Yearly hedged \$M (SS/SM/MV):
  2021 1.37/5.99/−0.95, 2022 −0.14/−5.57/+1.52, 2025 −1.39/1.87/−3.33.

## 2026-10-05 — Why the ATM book's vol line trends: fixed-moneyness vol is range-bound (ex COVID), the drift is strikes sliding down the skew in the rally

- Question (user): the −\$22.0M `vol` line in the desk-hedged ATM chart
  trends down for seven years — vol goes up and down, so why isn't it
  range-bound? Method: run the sticky-moneyness driver on the same ATM book
  at the same 512 paths/seed (`rainbow_attribution(sds, n_paths=512, seed=1,
  k_hi=np.inf)`, 330 s) and subtract: `vol_SS − vol_SM` is exactly the
  fixed-strike-vs-fixed-moneyness difference, i.e. the P&L of each strike's
  *position on the smile* moving as spot moves (the pl and eq+vol+cross
  totals of the two drivers agree to fp, so the split is clean). Conclusion:
  the user's intuition is right for vol at fixed moneyness — that line is
  −\$16.3M, of which **−\$15.5M is the COVID window** (2020-02-20 → 04-03)
  and the other 7+ years net to **−\$0.85M**; it oscillates with the vol
  level (2022 −\$6.2M, 2023 +\$3.2M) and the level itself ended lower than
  it started (SPX 1Y ATM 20.3 → 18.1). The COVID spike was never earned back
  because a daily-sold ATM book carries maximum vega when the market sits at
  its strikes and little after any big move: the pre-crash vintages went 30%
  OTM, and the post-crash vintages ran deep ITM in the +70% rebound while vol
  decayed, so the book paid for the spike on full vega and collected the
  decay on a fraction of it. The **trend** is the other piece, the slide:
  −\$5.7M net, but +\$9.4M in COVID and **−\$15.1M outside it**, negative in
  every rally year (2019 −2.1, 2021 −3.8, 2023 −3.4, 2024 −2.6, 2025 −2.5,
  2026 −1.5) and positive only when strikes move back out of the money
  (2020 +4.2, 2022 +5.9). The book sells ATM every day and SPX tripled
  (2,507 → 7,490), so every strike rides down a ~3.6-pt/10-mon 1Y skew into
  the money, repricing at ever-higher fixed-strike vol — a short-vega loss
  that compounds with the spot trend instead of mean-reverting with the vol
  level. Almost all of it is SPX (−\$5.0M of the −\$5.7M; SX5E −\$0.6M,
  HSI 0). Day to day the two pieces offset (corr with ret_spx: fixed-moneyness
  +0.73, slide −0.83; stds \$292k vs \$187k vs \$146k for the SS line),
  which is why the sticky-strike cut looks calmer, but the level is bounded
  and the slide is not. Implication for the open question in the previous
  entry: the SPX beta the sticky-strike delta leaves in the ATM book *is*
  this slide — a sticky-moneyness delta would move it into `eq_delta` and
  have earned it on the hedge; the SS-vs-SM hedge verdict needs re-measuring
  on this book.
- Record: cache `data/rainbow_attribution_sm_atm.csv` (sticky-moneyness
  driver, ATM book, 512 paths, seed 1 — NOT in
  `scripts/run_rainbow_attribution.py`'s PASSES; regenerate with the call
  above). SM driver columns differ from SS: `cross_sv` (≡ SS `cross_ev` role),
  `vol` = per-index singles + `vol_cross`, `time`/`time_roll`/`time_other`.
  Identity check: `eq+vol+cross` and `pl` match the SS cache to <1e-4 \$.
  Per-vintage proxy (BS vega on each index's own smile, basket weights
  0.4/0.4/0.2, `grid_lookup` at m/u vs m) tracks the exact per-index vol
  lines at corr 0.95/0.98/0.86 and gives the same qualitative split, but
  understates SPX's level by ~\$6M — the uncapped basket call's vega is not a
  single smile point; use the engine split, not the proxy, for numbers.
  Regression follow-up (user): regressing the green line on daily (or
  monthly/quarterly) index returns does NOT remove the trend — the return
  betas are ≈0 (daily +\$7M fitted, quarterly R² 0.00 ex COVID) and the
  intercept carries −\$23M ex COVID (t≈−4). Cause: the two spot-driven pieces
  have mirror-image betas — slide ~ rets R² 0.88 with cumulative fitted
  −\$23M (spx −13.6, sx5e −9.0; t −21/−24), level ~ rets R² 0.67 with
  +\$30M fitted and a −\$31M intercept (the bounded level mean-reverting
  back against the trending regressor) — so they cancel in the total's beta
  and the level piece's mean-reversion intercept masquerades as an
  'unexplained' drift. Only the SS−SM structural split separates them.

## 2026-10-05 — Rainbow attribution on an ATM-call book: uncapped, the book becomes a short-variance position with vega as a first-class risk

- Question (user): show the `rainbow_attribution.ipynb` result for a book of
  **at-the-money rainbow calls** instead of the 100/112 call spread. Method:
  the same daily-sold \$1M 1Y ranked-basket book, copula, CRN Sobol engine
  and sticky-strike attribution, with the cap removed (`k_hi=np.inf`). Nothing
  else changes. Conclusion: the machinery transfers unchanged (top-level
  resid 0.39% of P&L std, equity Taylor resid 1.0% with no drift), but the
  economics don't. The raw book loses −\$115M against \$134M of premium
  (−\$109M of it unhedged delta into the rally). The desk's delta-hedged view
  is −\$5.8M at **\$182k daily std (2× the spread's \$94k)**: +\$71.3M of gamma
  theta vs −\$57.7M of realized gamma + cross-gamma (a +\$13.5M variance
  harvest the spread never had), eaten by **−\$22.0M of fixed-strike vol**
  (spread −\$0.3M). COVID's six weeks cost −\$10.8M; the rest of the history
  made +\$5.0M. The vol line is largely *move-driven*: day returns + squared
  returns explain R² 0.29 of it, the squared term alone −\$32.9M; quiet days
  (all |ret| < 0.5%) earn +\$8.3k/day; SPX < −2% days cost −\$17.5M. Two sign
  flips vs the spread: every cross-gamma pair is now a loss (−\$8.7M total;
  uncapped basket convexity is long every pair, and SPX–SX5E is the smallest
  despite the biggest weight × correlation, so the rank kink still partly
  offsets), and `cross_ev` is +\$6.2M (spread −\$6.6M). Open question this
  raises: the sticky-strike-delta-hedged ATM book keeps an SPX beta
  (corr(desk, ret_spx) 0.33 vs 0.14 on the spread, via fixed-strike vols
  falling on up days). The SS-hedges-best verdict was measured on the spread
  and should be re-measured on this book before it's assumed here.
- Record: `rainbow_attribution_ss(sds, k_hi=np.inf)` — payoff args already
  passed through `price_sobol_batch` and `_intrinsic` (torch `clamp(0, inf)`
  is a plain max), so no library change was needed. Third pass in
  `scripts/run_rainbow_attribution.py` (PASSES now carry driver kwargs) →
  `data/rainbow_attribution_ss_atm.csv` (1,968 date pairs, 532s GPU,
  gitignored). Notebook `rainbow_attribution_atm.ipynb` (22 cells, 3s,
  mirrors the spread notebook section by section + a vol-vs-returns
  regression cell + an ATM-vs-spread side-by-side). Pre-run 512-path check for
  the uncapped payoff: fresh ATM vintage \$75.1k (2026-07-31) / \$122.7k
  (2020-03-16), per-contract scramble std \$343 at 512 paths / \$123 at 2,048
  (spread ~\$100 at 2,048). A 30-day COVID slice (from 2020-02-20) at 512 vs
  2,048 paths matches every component total within 0.5%, and the gamma/eq_resid
  split moves ~\$25k on \$6M. Numbers: premium mean \$68.0k/vintage (range
  \$42.8k–\$127.5k), final mark −\$28.0M (spread −\$20.6M), eq_delta by index
  SPX −70.0 / SX5E −30.9 / HSI −8.2 \$M, eq_gamma −25.4 / −20.6 / −3.1, xgamma
  SPX–SX5E −2.13 / SPX–HSI −2.54 / SX5E–HSI −3.97, vol −14.7 / −6.6 / −1.0
  (negative every year 2019–2026, 2020 −8.4), theta_value −5.07 vs theta_delta
  +5.01 (nearly cancel on the larger delta position), theta_gamma +\$36.2k/day
  mean (\$26k std, positive every year). Desk hedged std \$182k → \$142k after the
  gamma matrix (spread \$94k → \$67k), skew −4.0 → −0.4, worst days 2020-03-09
  −\$2.2M and 2025-04-07 −\$2.0M. eq_resid worst \$434k on 2020-03-12; top
  resid worst \$129k on 2020-03-09.

## 2026-08-27 — PDE-vs-MC demonstration: full-path Monte Carlo converges to the RILA PDE reference on every date and every seasoning state

- Follow-up to the RILA build (user, same day): the PDE-is-the-value-of-record
  claim "could make the whole thing a lot faster and the GPU Sobol stuff
  unnecessary" — so demonstrate, don't assert, that plain full-path Monte
  Carlo converges to `pde_reset_price` on the five regime dates, for newly
  issued AND seasoned states. The demonstration is a **joint limit** shown on
  two separate axes: convergence funnels in N at fixed daily stepping (20
  panels: GPU pseudo full-path to 2^21 with scramble/seed replicates, the
  canonical CPU numpy engine inside the same funnel, Sobol+bridge approaching
  the same limit faster), and step-bias ladders at fixed N=2^17 (Sobol+bridge
  so noise ≪ bias). Conclusion: **MC ≡ PDE everywhere once the measured
  first-order Euler bias is accounted for.** Calm-regime funnels close onto
  the PDE line within ±2se (2026-07-31 newly issued: gap +0.0011 ± 0.0141 per
  100 at 2^21); the COVID surface converges *confidently to the scheme's
  biased value* — funnel gap +0.347 ± 0.014 vs independently measured daily-step
  bias +0.341 ± 0.011, pseudo and Sobol agreeing on the biased value to 3e-4 —
  and the bias halves per step doubling (0.341→0.167, ratio 2.04; per-state
  0.16–0.34 per 100 at daily steps on 2020-03-16). Richardson-extrapolating
  the two finest rungs lands back on the PDE within ±2se on 8/9 configs
  (worst intercept 0.027 on COVID mid-window). The PDE reference's own error:
  refining every axis at once (n_K 145→217, n_x 1201→1601, steps/yr 320→480)
  moves no state by more than 0.0036 per 100. Post-window seasoned states add
  the market-side anchor: PDE ≡ MC internally, both a calibration-repricing
  error from the direct surface read. Verdict stands: the PDE (~3s/state,
  no regime-sensitive step knob) is the production engine for this product;
  MC's role is independent verification and the template for products that
  don't compress into a small carried state.
- Record: driver `scripts/run_rila_pde_mc.py` (skip-if-cached, ~50 min cold on
  a shared GPU; stages refs/funnel/bias) → `data/rila_pdemc_ref.csv` (20
  states: PDE @ production config + refined-grid resolve + surface reads),
  `rila_pdemc_funnel.csv` (270 rows: gpu_pseudo N=2^12..2^21 all 20 states,
  reps 16/12/8 by N; cpu_plain anchors 2^14/2^16 ×3 seeds; gpu_sobol_bb
  2^12..2^17 on newly-issued), `rila_pdemc_bias.csv` (44 rows: py ∈
  {12,52,126,252,504} at N=2^17×8, newly-issued all dates + mid_r2 STUDY +
  COVID/current mid_r4 top-ups), `rila_pdemc_arb.csv` (fresh-seed arbitration
  of the one >2×2se residual: 2026-07-31 mid_r4 re-measures at +0.013 ± 0.021
  pseudo 2^19×32 / +0.003 ± 0.002 Sobol 2^17×16 — a tail draw among twenty
  seed-correlated ladders, not a disagreement; kept in the notebook as the
  honesty footnote). Notebook `rila_pde_vs_mc.ipynb` (13 cells, ~8s off
  cache). States: new (r=6), mid r=4 K/S=0.95, mid r=2 K/S=1.08, post-window
  3y K/S=0.90. Pitfalls: funnel replicate ladders share seeds across N rungs
  (rungs correlated — error bars per rung remain valid, but the worst-of-20
  top-rung draw can sit ~2×2se from truth, hence the arbitration file);
  seasoned mid-window bias must be measured per state (COVID r=4 bias +0.313
  explains its +0.299 funnel gap — without that row the verdict table shows
  an unexplained 13σ).

## 2026-08-27 — RILA reset package under Dupire local vol: what a 6y path-dependent insurance hedge costs, and how fast Monte Carlo gets there

- A new, **valuation-only** thread (user spec; explicitly no greeks, no VaR,
  no book simulation). The product is the insurer's hedge asset behind a
  registered index-linked annuity on SPX: 6y, terminal
  `(0.8K−S_T)+ − (S_T−K)+ + (S_T−2K)+` on a basis that **ratchets**
  `K ← max(K, S_{t_i})` at six monthly fixings, so the final basis is
  `max(S₀, S_{1/12..1/2})` and all three strikes move with it. Question:
  price it under Dupire local vol calibrated to the same five-date regime
  panel the rainbow thread uses, and measure how fast plain Monte Carlo
  converges versus advanced GPU methods. Four findings. **(1) The pillar
  grid cannot reach the product** — the repo's 8 TTM pillars stop at 1.0y,
  so the calibration input was rebuilt from the cleaned raw rows (~17 terms
  to ~9.4y) with total variance interpolated at fixed log-forward-moneyness
  (PCHIP in τ through exact per-term knots + the w(0,y)=0 anchor; a cubic
  overshoots between the anchor and the first ~1M term and invents calendar
  arb). **(2) The flat wing clamp is not a neutral baseline.** It kinks
  σ(y) at the 50/150 join; Dupire reads the kink as local butterfly
  arbitrage (denominator < 0 at 90–255 nodes per date, at maturities out to
  6y), the guard floors σ_loc in a band where the long-dated put wing still
  has vega, and the LV surface then fails to reprice its own input by ~0.31–0.42
  vol pts for τ≥0.5 (individual 5.4y/70-moneyness errors ≈ −2 pts), leaving
  the frozen-basis package 0.25–0.59 per 100 away from the direct surface
  read. The C1 edge-slope taper `rainbow.py` already uses fixes it
  (0.06–0.14 vol pts, package gap 0.0002–0.025) and is now the module
  default; the clamp is still calibrated and priced on every date as the
  spec's baseline/sensitivity. Wing sensitivity on the newly-issued package
  (2026-07-31): taper − clamp = **−0.978 per 100 of basis** (−$9,777 per
  $1M), of which the 2K call leg alone is −1.379 (2.620 vs 3.999; implied
  vol at the 200 strike 15.35% vs 17.51%). **(3) The ratchet is coupled on
  only six dates**, which buys two things: a deterministic PDE reference —
  n_K independent 1-D Crank–Nicolson solves sharing one tridiagonal
  operator, relabelled `V(t⁻,S,K)=V(t⁺,S,max(K,S))` at each fixing, ~3
  s/state — and a flagship MC engine that simulates only the six-month
  window (126 dims) and integrates the 5.5y tail *exactly* against a
  precomputed T(S_w,K) table. **(4) Sampling error stops being the problem
  long before the bias does.** The hybrid clears 1bp of spot sampling error
  at 4,096 paths in 0.035 s (vs 314 CPU s for plain MC, ~8,900×), but at
  daily window steps the log-Euler bias is still 0.009 per 100 on
  2026-07-31 and **0.332 on 2020-03-16**, where σ_loc reaches 324% — ~50×
  the sampling error at N=2^18, and first-order in Δt (halving the step
  halves it, measured over 21→1008 steps/yr on both dates). On crisis
  surfaces the budget belongs to steps, not paths — or to the PDE, which
  for this product is both exact and cheap.
- Record — **values** (per 100 of basis, signed, taper wings, PDE
  reference; newly issued K=S, 6y): 2020-03-16 **+9.127**, 2021-09-20
  −2.261, 2022-03-16 −6.859, 2024-10-08 −15.775, 2026-07-31 **−19.111**;
  frozen-basis European −3.230 / −7.582 / −12.619 / −20.349 / −23.056, so
  the **reset premium** is +12.358 / +5.321 / +5.760 / +4.574 / +3.945 —
  always positive (the ratchet can only shorten the short call and lengthen
  the put) and monotone in the vol regime. $1M-notional illustration on
  2026-07-31: value −$191,105, of which the ratchet is worth +$39,450. The
  sign flip is the put: on the COVID date a 6y 80%-strike put on a crashed
  spot is worth 15.74 points against 20.39 collected on the short 100 call,
  and the ratchet in that vol is worth 12.36. Seasoned mid-window (r=4/2 ×
  K/S 0.95/1.00/1.08) and post-window (r=0, ttm 5.5/3/1 × K/S
  0.75/0.90/1.00/1.10) grids in `data/rila_states.csv`; **post-window
  states are returned by direct surface reads, never simulated** (identity,
  tested), and the LV-PDE vs direct-read gap over all 60 of them (mean
  0.059, max 0.533 per 100 on the COVID date) is the calibration error
  measured end to end on the product itself.
- Record — **calibration**: `dupire_local_vol` on 120 geomspace(1/365,
  6.05) × 141 y∈[−3,3] = 16,920 nodes, floor 0.05², cap 4·σ_ATM(τ). Taper
  wings, τ≥0.5 vol-pt repricing error mean/max: 2020-03-16 0.859/4.19,
  2021-09-20 0.067/0.284, 2022-03-16 0.059/0.293, 2024-10-08 0.141/0.687,
  2026-07-31 0.095/0.618. Floor fires on 2.5–11.2% of nodes, cap on
  0.6–45.2% — the cap almost entirely at τ<0.2y, where Dupire is genuinely
  ill-conditioned (a 3-week smile running 85% vol at 50-moneyness to 12%
  ATM makes the denominator a small residual of large cancelling terms);
  the product's monthly fixings give that region ~no weight. 2020-03-16
  carries real arbitrage (1,691 calendar-arb + 130 butterfly-arb nodes) the
  floor absorbs — same crash-surface pathology the rainbow thread logged
  from the density side. **Refining the grid does not help**: n_y 141→601
  moves the τ≥0.5 mean by <0.05 vol pts, so the residual is data, not
  discretisation (141 kept, per spec).
- Record — **convergence** (all "per 100 of basis"; 1bp of spot = 0.01).
  CPU plain pseudo log-Euler is textbook N^(−1/2) with no surprises (the
  payoff is continuous and the fixings are exact grid nodes). 2026-07-31,
  daily (1512) steps: s.e. 0.143 at N=2^16 in 1.29 s ⇒ **10bp in 3.1 s,
  1bp in 314 s (13.4M paths)**; weekly (310 steps) same variance at
  one-fifth the cost (1bp in 60 s) but 5× the bias; monthly 14 s. COVID is
  ~11% noisier (1bp in 395 s). The **frozen-basis control variate** on the
  exact LV-PDE mean: corr 0.990, s.e. 0.143→0.020 (~50× variance), 1bp in
  6.4 CPU s — a pseudo-MC device only, and explicitly *not* a contradiction
  of the 2026-08-06 negative CV verdict, which was about scrambled Sobol
  (there the QMC rule has already integrated the smooth component the
  control shares, so the correction only adds the control's own noise; on
  COVID the CV is weaker, 3.2×, 1bp in 40 s). GPU rungs, RMSE across 16–24
  independent scrambles, 2026-07-31, RMSE @ N=2^16 / s / measured s to 1bp:
  device pseudo 0.126 / 0.269 / 21.6* ; Sobol full path no bridge 0.031 /
  1.005 / 4.6* ; **Sobol + Brownian bridge full path 0.0129 / 1.095 /
  2.12** ; hybrid pseudo 0.0350 / 0.021 / 0.051 ; **hybrid Sobol+bridge
  0.00126 / 0.097 / 0.035** (* = extrapolated along the fitted rate).
  Fitted RMSE ~ N^a: pseudo −0.544, Sobol −0.572, Sobol+BB −0.464, hybrid
  pseudo −0.564, **hybrid Sobol+BB −0.639** (COVID: −0.530 / −0.553 /
  −0.527 / −0.571 / −0.602). **The lesson on the rate**: on the raw
  1512-dimensional path scrambled Sobol buys a *constant*, not an exponent
  (4× at N=2^12, 13× with the bridge, but every fitted a stays near −0.5);
  only collapsing the dimension to 126 *and* smoothing the payoff moves the
  exponent. At N=2^16 the hybrid Sobol+BB is 100× more accurate than device
  pseudo **and 2.8× faster**; it is 28× more accurate than the pseudo
  hybrid at 4.6× the cost. Speedups vs plain CPU MC at 1bp: 15× (device
  pseudo), 69× (Sobol), 148× (Sobol+BB), 6,200× (hybrid pseudo), 8,900×
  (hybrid Sobol+BB) — the last two sit on a ~0.02–0.04 s kernel-launch
  floor, so they are launch-bound, not throughput-bound.
- Record — **bias ladder** (value − PDE reference, per 100): 2026-07-31
  hybrid window 21/63/126/252/504/1008 steps-per-year → 0.0140 / 0.0177 /
  0.0131 / 0.0092 / 0.0070 / 0.0054; 2020-03-16 → 1.775 / 1.011 / 0.599 /
  0.330 / 0.182 / 0.092 (clean first-order: each doubling ≈ halves it). The
  full-path engine at the same steps/yr agrees with the hybrid on the bias
  (2026 0.0098 vs 0.0092 at 252/yr, COVID 0.353 vs 0.332) — hybrid vs
  full-path parity holds to 0.7σ, so the hybrid is the same estimator, only
  cheaper. Extrapolating COVID, 1bp of bias needs ~9,300 steps/yr; the
  3-second PDE is the better answer there.
- Record — **modules**. `vol_pca/local_vol.py` (numpy only, tested to
  import in a torch-free env): `load_term_surfaces`/`TermSurface` (built on
  the new `data.load_clean_frame`, a behaviour-preserving extraction from
  `load_surfaces`), `dupire_local_vol` → `LocalVolSurface`,
  `european_package_surface`, `cn_rollback`, `pde_vanilla_calls`,
  `pde_reprice_table`, `pde_package`, `pde_tail_table`, `pde_reset_price`,
  `mc_time_grid`, `simulate_terminal`, `price_rila_mc`, `rila_value`,
  `flat_reset_oracle`, `RilaState`/`rila_state`/`per_100`.
  `vol_pca/rila_torch.py` (the **second** `*_torch.py`; CLAUDE.md's "single
  torch module" rule generalised to "torch only inside `*_torch.py`"):
  `DeviceLV`, `bb_plan`/`bb_increments`, `TailTable`/`make_tail_table`,
  `rila_price_torch`, `prepare`, `rila_replicates`. Driver
  `scripts/run_rila.py` (24.8 min cold, five skip-if-cached stages →
  `data/rila_lv_<date>_<wing>.npz`, `rila_calib.csv`, `rila_reprice.csv`,
  `rila_wing.csv`, `rila_states.csv`, `rila_conv_cpu.csv`,
  `rila_conv_gpu.csv`, `rila_bias.csv`). Notebook `rila_pricing.ipynb`
  (analysis off cache + one live validation cell, 12.5 s). Tests:
  `tests/test_local_vol.py` (19) + `tests/test_rila_torch.py` (11,
  importorskip); suite 125 passed.
- Record — **pitfalls hit, in the order they bit**. (a) *The PDE's top
  Dirichlet boundary.* The frozen-basis asymptote V → −K·D is **wrong**
  once a fixing can still fire (there V → −S_ratchet·D); hard-coding it put
  the reset reference ~1bp of spot off the flat-vol oracle. The fix is the
  **linearity condition** (V affine in S at both ends, imposed through the
  three outermost nodes so the system stays tridiagonal) — exact for a
  call, for the frozen package and for the ratcheting package alike, and no
  case analysis. (b) *The ratchet gather.* Interpolating V(t⁺,S,·) at
  K'=max(K,S) must be **linear in K with top extrapolation**, not clamped:
  clamping values a 10σ node at −k_max instead of ≈−S. The local end slope
  happens to be right in both regimes (−D with the basis frozen, ≈0 while a
  later fixing can still overwrite it). (c) *Rannacher on every leg.* The
  relabel leaves a delta kink at S=K, so each post-fixing roll needs its own
  fully-implicit start, not just the terminal payoff. (d) *The flat-vol
  oracle's quadrature.* Plain Gauss–Hermite over the un-split
  max(1,xg)·v(min(xg,1)) integrand was still drifting ~1bp at 128 nodes
  (kinked *and* unbounded); splitting at xg=1 — Gauss–Legendre in
  copula-uniform space below, closed-form lognormal partial expectation
  above — converges to 1e-6 in 0.4 s, and the PDE reference then lands on
  it to 5e-5 per 100. Oracle x-grid floor is 0.03, not the spec's 0.3:
  compounding down-months at the outer nodes reach ~0.05. (e) *Sobol
  generation.* `rainbow_torch.sobol_normals` (one small resident d=3 point
  set per book slot) does not transfer: here d=84–1512 × N neither fits
  "generate once, keep resident" nor survives the host round-trip, so the
  rungs use `torch.quasirandom.SobolEngine` drawn straight into a device
  buffer (one seed = one scramble). Generating N×1512 points is ~half the
  full-path rung's wall clock — which is exactly why the error-vs-seconds
  panel, not error-vs-N, is the honest chart. (f) *fp32.* Safe only because
  the path state is carried as q = ln(S/S₀) and the basis as K/S₀ (both
  O(1)); carrying ln S itself (≈8.9 for SPX) would accumulate ~4bp of
  relative error over 1512 steps. CRN parity CPU↔GPU: 8e-16 relative in
  fp64, 2.7e-7 in fp32.

## 2026-08-10 — Does the 100/112 cap shelter the rainbow price from copula/tail misspecification? Yes at-the-money; the shelter is regime-dependent, and an uncapped call would promote the dependence model to a first-order input

- Methodology discussion (user: pricing a call *spread* sidesteps tail-scenario
  correlation issues a plain call would force us to study — agree?). Measured
  answer on two regimes, same implied marginals and CRN scrambled-Sobol
  uniforms (2^17 paths, 2 seeds, seed-diff ≤$11), dependence swapped:
  Gaussian base vs Student-t ν=8/4 at the **same** correlation (pure
  tail-dependence stress, same Kendall τ) vs ±10pt average-corr blends.
  Fresh 1Y ATM unit (2026-07-31): the spread is essentially copula-flat
  (+0.3%/−0.1% per ±10 corr pts — the 100 and 112 kinks' dispersion vegas
  cancel; t4 −0.5%), while the uncapped 100 call moves ±2% per 10 corr pts
  and −1.5% under t4, all of it through the beyond-cap tail E[(B−1.12)+]
  (29% of the call's value, ±7% per 10 pts). But the shelter is
  **regime-dependent, not structural**: the displaced COVID vintage (sold
  2019-09-16, marked 2020-03-16, g≈0.75, τ≈0.5 — the band needs a joint
  ~+30% half-year rally) swings ±10% per 10 corr pts and −5.7% under t4
  at unchanged correlation, even as a spread — dollar-bounded by the cap
  ($6.2k value), but the copula's *shape* prices those vintages. Conclusion:
  agree — the cap localizes the price to the band CDF (layer-cake
  ∫P(B>k)dk over [1.00,1.12]) and bounds worst-case dependence error at
  12%·N·ΔP, and the near-corr-neutrality is a measured kink cancellation,
  not a free lunch; an uncapped call keeps the whole joint upper tail —
  copula family + corr level + marginal wing taper all become first-order,
  needing a model-risk band (t/asymmetric families), tail-dependence
  estimation on thin joint history, wing-taper sensitivity, path-count
  revalidation for the unbounded payoff, and Δcorr as a VaR bump/scenario
  dimension.
- Record: scratchpad-only study (recipe: `ImpliedMarginal.quantile` +
  hand copulas over one shared Sobol U(d=4) — z=Φ⁻¹(U₁..₃)·Lᵀ, t via
  χ²-mix of U₄, blends toward all-ones/identity for PSD ±10pt shifts;
  hand-Gaussian vs `price_rainbow` parity $3). Numbers, fresh ATM
  (spread $53.0k, call $75.1k = spread + $22.1k tail, P(B>hi)=0.28):
  t8/t4 spread −0.3/−0.5%, call −0.7/−1.5%, tail −1.8/−3.8%; ρ±10
  spread +0.3/−0.1%, call +2.2/−2.0%, tail +6.7/−6.6%. Displaced COVID
  vintage (spread $6.2k, call $7.9k, P(B>lo)=0.11): t8/t4 spread
  −3.0/−5.7% with tail +9.0/+14.8% (t moves dependence mass center→tails:
  near-band prices *fall* while the genuinely-far tail fattens — "tail
  dependence raises calls" is only true tail-by-tail); ρ±10 spread
  +9.2/−10.0%, call +12.0/−12.5% — displaced books are correlation-LONG
  (recovery needs co-movement) where the ATM book is corr-flat. Ranked
  weights keep a copula exposure even in-band: B = 0.4·(p₁+p₂) +
  0.1·|p₁−p₂| + 0.2·p₃, so the basket *forward* itself is
  copula-dependent (E|p₁−p₂|=0.107 ⇒ ~1.1 pts of E[B]=1.029) — unlike a
  fixed-weight basket. Risk-side twin for a call book: the historical
  scenario set already carries realized co-crash tails (joint sampling),
  but both VaR engines reprice at the frozen constant Σ̂ — a correlation
  seam like the excluded curve seam; second-order for the corr-flat
  spread book, a missing risk factor (bump greek + scenario dimension)
  for a call book. No implied-corr instruments exist in this dataset, so
  a call book's dependence exposure is reserve/limit territory, not
  hedgeable.

## 2026-08-06 — Rainbow cheap-VaR benchmark across five gamma regimes: the projection owns the bulk, a ~10% dual-screened partial reval makes VaR99 exact

- The rainbow thread's first VaR study (user: "pick one day with negative
  gamma — hedged vs unhedged, full bump vs greek based", then "pick several
  more days"). Phase 1, as-of **2022-03-16** (the short-gamma book: all
  nine 1%-bump gamma-matrix entries negative, 1'Γ1 −$640M; SS deltas
  −$27.3/−$19.9/−$10.9M per 100%): nested full reval of all 1,968
  joint-date scenarios (≈512k revaluations, **57 s** CRN Sobol 512 —
  beating the ~3 min planning figure) vs the formula-free bump projection
  (**102 CRN pricings, 1.9 s**); SS-delta hedge cuts full VaR99 $1.69M →
  $636k; projection gaps: unhedged −0.1%, hedged −2.8% (per-index k=4×3) /
  −4.2% (joint 3-surface fit, 4 factors). Phase 2 — the **five-date regime
  panel** (~1 GPU min/date makes hand-picked days the right scale; the
  ~33 h rolling history explicitly skipped): + 2020-03-16 (COVID peak — the
  displaced far-OTM book is LONG SPX gamma +$98M with tiny deltas),
  2021-09-20 (mixed signs), 2024-10-08 (rally book, long gamma),
  2026-07-31 (current book, deep ITM, 1'Γ1 +$691M). **The single-date
  hedged answer did not survive the panel**: the same k=4 projection that
  sits −2.8% on the bear book understates hedged VaR99 by −18…−49% on the
  other four regimes (hedged corr 0.32 on the COVID as-of, 0.92–0.99
  elsewhere); unhedged stays easy (within ~3% everywhere except the COVID
  as-of's −10%). Diagnosis: the hedged tail on every as-of is the same few
  historical shocks (March-2020, 2025-04), where (a) the spot Taylor is
  beyond its radius on |dS|≳5% — the COVID book's local long-gamma
  quadratic predicts +$2.3M on the −9.5% 2020-03-12 scenario where full
  reval says −$0.4M — and (b) linear vega understates short-vega losses
  ~20–40% under ±10–20-vol-pt shocks; factor truncation is NOT the driver
  (k=18 ≈ k=12; joint-4 tracks per-index-12 within a few points on every
  book, so 4 joint factors stay the projection's budget config). The fix is
  the vanilla crisis-tail owner, transferred: **dual-screened partial
  reval** — fully revalue worst-100 by unhedged ∪ worst-100 by hedged
  projection ∪ every any-index |dS|>4% scenario (~175–220 ≈ 10% of the
  sweep, greeks-side info only) — **hedged AND unhedged VaR99 land at 0.0%
  on all five books**; VaR95 within 0–27% at m=100, within ~1% at m=250
  (~20–25%). Budget ladder per as-of: 2 s projection → 8 s hybrid (VaR99
  exact) → 15 s wide hybrid (VaR95 too) → 57 s full benchmark.
- Record: VaR section at the end of `rainbow_torch.py`. `rainbow_scenarios`
  → RainbowScenarios (per-index spot ratios, fixed-moneyness Δgrids, dt
  over joint pairs); `rainbow_var_full` = the chunked nested sweep (chunk=8
  scenarios × book stacked per `tables()` call via (B,8,13) dgrids;
  scenario state = `tables(dgrid=Δ, tau=τ−dt, tau_price=τ)` at g0·u — the
  scenario dgrid is the RAW fixed-moneyness change, NO strike shift (the
  attribution's eq+vol composition identity is the proof; adding one
  double-counts the slide); curves frozen in both engines (fwd ~4% of
  daily P&L std, rate ~0.4%)); `rainbow_fit_dsigma`/`rainbow_scenario_dsigma`
  = roll-in sticky-strike pillar moves over joint pairs / re-anchored on
  the as-of grid (lookups only; vanilla re-anchoring rule);
  `rainbow_bump_greeks` → RainbowBumpGreeks (SS spot block ±1%/±2% singles
  → delta/gamma/speed + pair corners → cross-gammas; ±1σ dgrid factor
  pairs → exposures + free curvatures; 4-corner spot×factor crosses
  composing the strike shift ON the factor-shocked grid via
  `strike_shift_dgrid(extra=...)`; one factor-mean reval);
  `rainbow_bump_pnl` = the zero-valuation projection (cube on, quad off —
  quad="all" widens VaR gaps here too). Factor models fit ONCE on the full
  joint history, shared across as-of dates (frozen-weights convention):
  per-index corr-weighted PCA k≤6 (evr3 SPX 0.857/SX5E 0.861/HSI 0.873) +
  joint 312-dim fit k≤8 (evr3 0.705); μ shared per index ⇒ one mean reval
  serves both families. Caches per date via `scripts/run_rainbow_var.py`
  (skip-if-cached, shares factory/fits/scenarios): 
  `data/rainbow_var_scen_<asof>.csv` (per-scenario pnl_full + 26 scores) +
  `rainbow_var_greeks_<asof>.npz`; panel notebook `rainbow_var.ipynb`
  (off-cache, ~2 s; the dual screen is notebook-level analysis — the
  hybrid's replacement values come from the cached sweep, which is exactly
  what a real partial reval would compute since the screen uses only
  greeks-side ranks). Tests `tests/test_rainbow_var.py` (6): flat-world
  zero-P&L identity, pure-spot scenario ≡ the SS bump combo to 1e-9, bump
  spot block ≡ `book_greeks` to 1e-9, exposure+curvature telescoping ≡
  full reval to 1e-6. Pitfalls for the record: hedged-gap magnitude
  anti-correlates with hedged-VaR size (long-gamma books have small hedged
  VaR ⇒ big relative gaps, $40–80k absolute); the dual rank matters
  because Taylor overshoots big-dS scenarios conservatively while
  vol-driven true-tail scenarios can rank lower; per-date profiles:
  2020-03-16 mark −$1.6M u99 $515k h99 $205k, 2021-09-20 −$16.6M/$2.12M/
  $183k, 2022-03-16 −$4.8M/$1.69M/$636k, 2024-10-08 −$22.9M/$1.70M/$140k,
  2026-07-31 −$20.6M/$2.46M/$202k.

## 2026-08-06 — Sticky-strike attribution + the measured hedge verdict: SS delta cuts hedged risk 43%

- The follow-through on the greeks work: re-cut the rainbow book's daily
  attribution so the delta line IS the sticky-strike hedge, then let the two
  full-history hedged series settle which framework hedges better. Verdict —
  **the sticky-strike delta hedge wins decisively**: hedged-P&L std $166.7k
  → $94.2k (−43%), worst day −$3.12M → −$1.49M, and the mechanism is visible
  — the SM-hedged series still carries |corr| 0.52 with SPX returns (the SM
  delta is ~$1.4M more short via the smile-slide vega term, leaving
  +$1.4M·ret of market beta that *earned* over the rising sample, which is
  why SM's cumulative looks better: unhedged beta, not hedging skill).
  Holds in every subsample (2020: $389k→$213k; ex-2020: $94k→$57k; big SPX
  days: $530k→$274k; better on 62% of days). Adding the 1%-bump gamma +
  cross-gamma terms explains most of the remainder: std $67.3k, skew −8.4 →
  −1.0, worst −$0.85M — the crash tail is almost entirely second-order
  equity risk. The vanilla study's transferred verdict is now a measured one
  on the exotic book.
- Record: `rainbow_attribution_ss` in rainbow_torch.py + second pass in
  `scripts/run_rainbow_attribution.py` (cache
  `data/rainbow_attribution_ss.csv`; SM cache stays for the comparison).
  Components: eq = all spots → t sticky-strike (per-index tables re-derived
  via `strike_shift_dgrid` at the realized ratio u_i = S1/S0, g → g1),
  attributed by the 1%-bump matrix measured on the same book/CRN points via
  `book_greeks(per_position=True)` — columns eq_delta(_spx/sx5e/hsi),
  eq_gamma(_*), eq_xgamma(_pair), eq_resid; vol = day-t grids sampled at
  m/u_i (fixed strikes, t−1 anchor/curves, τ0 fixed ⇒ roll-free), singles +
  vol_resid; cross_ev = 4-corner eq×vol on totals — eq and vol compose
  EXACTLY (both applied = plain day-t tables at g1); theta 3-split per spec:
  theta_value = (df_time/df0−1)·pv_base (zero revals), theta_delta =
  Σ per-vintage SS delta × per-vintage forward-ratio decay
  (fac.forward_ratio(τ1f)/fr(τ0)−1; deltas free from the greek singles' pv
  vectors), theta_gamma = time − value − delta (rest folded per spec);
  fwd/rate/resid/premium/mark as the SM driver. ~30 table builds + 30
  pricings/date at the 512-path standard ≈ 260 ms/date → 1,968 pairs in
  515 s. Validation: flat-world test (vol/cross/fwd/rate vanish EXACTLY,
  splits are identities, resid = time×spot cross only); pl/time/fwd/rate
  match the SM driver < 1e-6 at equal paths/seed and the eq+vol+cross+resid
  regrouping preserves their sum (tested); COVID-window resid std $13k vs
  $412 calm (top-level resid absorbs eq_resid-adjacent third-order only on
  ±9% days). Honesty: the caches use 2,048 vs 512 paths — pl diff std
  $759/day, max $7.1k, two orders below the measured effects. Hedge
  comparison + verdict: rainbow_greeks.ipynb section 6 (cells 16–19).
  Notebook migration (user, same day): the canonical attribution study is
  now the sticky-strike `rainbow_attribution.ipynb` (16 cells, all SM
  calculation removed, analysis-only off the cache — 1.8s in-notebook
  runtime, 4.6s nbconvert wall; incl. the eq-Taylor-quality study:
  eq_resid totals +$0.03M over 7.6y, std $14k = 2% of pl std, worst $320k
  on 2020-03-17 — and the theta 3-split and per-pair xgamma cumulative
  views); the original SM study was renamed
  `rainbow_attribution_sm.ipynb` with a historical-reference banner
  (unmaintained; its eq_delta is the smile-delta reporting split).
- Record (desk-view headline, user, 2026-08-06): the canonical notebook's
  hedged headline is the **desk view** `pl − eq_delta − theta_value −
  theta_delta` = **−$11.51M** (std $94.1k — the carry is smooth, so the risk
  numbers don't move): delta P&L is hedged away, theta_value (−$3.52M) is
  treasury's interest on the option market value, theta_delta (+$1.44M) is
  the forward-roll gain that pays the delta-one desk's funding on the hedge;
  the desk keeps only theta_gamma (+$0.48M). Same convention as the vanilla
  thread's `hedged = pl − eq_delta − time_funding`. The SS-vs-SM 43% verdict
  (greeks nb §6) is carry-insensitive — both series there are
  carry-inclusive by identical construction. Same change mirrored in the
  vanilla `attribution.ipynb` (its headline was already the desk view —
  that's where the convention came from; chart upgraded to the decomposed
  form + daily histogram, md gets the treasury/delta-one mapping; vanilla
  `time_funding` nets to −$0.01M over 12.4y so the headline barely moves,
  verified off the long-history cache). Its hedged cell is committed
  **unexecuted**: the notebook loads `SPX_volSurface 2.csv`, deleted by
  mistake — re-execute after the user restores it.

## 2026-08-06 — Sticky-strike bump greeks: the cross-gamma matrix is the book's main convexity

- How should the rainbow book's hedging greeks be computed — delta vector and
  gamma matrix including cross terms, in which spot framework, and how many
  paths do they need? Built the sticky-strike bump route (user spec): each
  index's 1% spot bump **re-derives that index's implied distribution from
  the moneyness-shifted smile** (σ_new(m) = σ_old(m(1+ε))) and scales its
  seasoning; 19 CRN pricings per book. Findings: the **SPX–SX5E cross gamma
  ($207M/100%²) dwarfs every diagonal ($9–35M)** — the ranked weights carry
  0.1·|P₁−P₂| whose second derivative is a δ-ridge along P₁=P₂, so
  per-underlying gammas miss the book's dominant second-order risk; the
  sticky-strike vs sticky-moneyness delta gap is 1–3.5% ($1.4M of SPX hedge);
  **512 paths are comfortably enough for the whole matrix** (greeks converge
  better than the price under CRN); and on crash surfaces the trust order
  inverts — CRN-Sobol finite differences beat the quadrature's (COVID quad
  delta oscillates −4.8→−8.2→−5.9M across 24²/32²/48² nodes while ten Sobol
  families agree at −6.13±0.02M).
- Record: `MarginalFactory.strike_shift_dgrid` (cardinal-weight resampling of
  the pillar rows at m(1+ε), clamped 50/150; exactly zero on a flat smile ⇒
  SS ≡ SM there, tested), `prepare_book_greeks` (19 batches: base, ±ε per
  index, 4 corners per pair; mark-semantics book; mode switches SS/SM),
  `book_greeks` (any pricer; sold-book sign; delta/(2ε), diag (V₊−2V₀+V₋)/ε²,
  cross 4-corner/(4ε²)), `rainbow_book_greeks` wrapper (sobol default
  512/quad engines). Gamma cannot come from autograd through the sampler —
  payoff pathwise piecewise-linear ⇒ Hessian 0 a.e.; SM bump delta matches
  the pathwise autograd delta to ≤0.12% at ε=1%. ε-sensitivity (quad):
  delta flat to 0.2% over ε∈[0.5%,2%], big cross −3% (210→204), small SPX
  diagonal window-dependent (15.8→11.6) → quote gammas as ±1%-window
  convexities. Five-date sweep: 2019 ramp book short gamma (diag −70/−77M),
  COVID book delta collapsed to (−6.1,−4.2,−3.6)M with +98M SPX diag and
  negative crosses, 2024-08-05 asymmetric (−47.9 vs −30.3). Convergence (10
  scramble families × N∈{64…4096} × 5 dates): at 512 paths worst delta std
  $45–67k → ≤$673 P&L per 1% move (0.026bp of book notional); worst gamma
  element → ≤$313 on a joint 1% day (0.012bp); ~N^0.75–1.0 decay. Cost:
  19 pricings + 6 shifted table builds ≈ 0.1s/book at 512. Tests:
  `test_strike_shift_dgrid_matches_scipy` (1e-13),
  `test_book_greeks_flat_world`, `test_book_greeks_sobol_vs_quad_real`.
  Notebook: `rainbow_greeks.ipynb` (16 cells, executed). Standing decisions
  recorded same day: 512 Sobol paths is the book-level default, and
  sticky-strike bumps are the rainbow book's main greeks path (user; the
  autograd delta — sticky-moneyness by construction, torch-only — is demoted
  to SM test oracle + the attribution driver's eq_delta reporting split;
  "SS hedges better" remains a transferred verdict from the SPX study until
  the daily hedged-P&L comparison is run on this book).

## 2026-08-06 — Sobol path count for book marks: 256 is enough, 2,048 is 10× inside

- How many paths does one valuation of the ~258-option rainbow book need, if
  the standard is "independent Sobol runs agree within 1bp of book
  notional"? Ten independent scramble families on five as-of dates (2019
  ramp, COVID 2020-03-16, calm 2021, the 2024-08-05 vol spike, last date):
  64 paths fail (spread up to 2.1bp), 128 is borderline, **256 is the first
  comfortable level** (full 10-run spread 0.34–0.62bp), and the production
  2,048 sits 10–15× inside tolerance. Crisis dates are no worse than calm
  ones at book level, because the per-slot independent scrambles make the
  250 per-option errors average down √250 ≈ 16× — verified, not assumed.
- Record: worst single-run deviation at 256 paths 0.44bp of notional;
  book-level std shrinks ≈ N^(-1.0) (std $12.6k at 64 → $213 at 4,096
  paths); √-rule check passes at every N (book std ≈ √Σ per-option var,
  e.g. $434 measured vs $416 predicted, COVID book at 4,096). Per-option
  worst std at 2,048 is $69–126 — the $100/contract attribution target —
  vs $400–535 at 256, so: 2,048 for attribution, 256–512 for pure book
  marks and VaR sweeps. Book pricing 1.2–1.5 ms at ≤512 paths
  (launch-overhead-bound; 64 paths is no faster than 256) vs 4.5 ms at
  2,048. Prefix trick: the first 2^m points of a scrambled 2^12 Sobol set
  are a valid nested (t,m,3)-net, so one `sobol_normals(256, 4096, seed)`
  per family serves every N with paired comparisons. Script: scratchpad
  `paths_calibration.py` (not committed); results table in CLAUDE.md's
  rainbow_torch bullet at the time, then moved here.

## 2026-08-06 — Control variate (40:40:20 fixed geometric basket): measured negative

- Can a control variate cut the path count ("make it run faster")? Built the
  classic construction: price the call spread on the fixed-weight geometric
  basket P₁^0.4·P₂^0.4·P₃^0.2 on the same paths, replace its noisy MC mean
  with its exact expectation (computable because, conditional on two copula
  normals, the geometric basket is monotone in the third index — no rank
  kink). Verdict: **textbook under pseudo-random MC (30–70× variance
  reduction) but it does not compose with scrambled Sobol**, the production
  engine — QMC has already integrated the smooth payoff component the
  kink-free control shares, so the correction only adds the control's own
  independent noise. Plain Sobol dominates pseudo+CV at equal paths
  everywhere measured. Kept CPU-only as a documented negative result.
- Record: `geo_basket`, `price_geo_quad`, `price_rainbow(cv="geo",
  cv_beta=1.0|"fit")` in rainbow.py; no torch port. Numbers (24 scrambles,
  2,048 paths, calm 1Y): pseudo std $1,432 → $170 with CV; Sobol std $47 →
  $58 (worse); COVID surfaces: CV ratio ≥ 1 at every N ∈ {256…8,192} even
  with an oracle (dense-Sobol) anchor; small-N calm benefit (0.60× at 512)
  is beaten by plain Sobol at 2× paths, which costs less than the anchor.
  Second failure mode: the E[C] quadrature anchor inherits the crash-surface
  oscillatory tail (2020-03-16: −$513 at 24² nodes, +$756 at 96² vs dense
  Sobol) — the violence lives in the quantile tables, not the rank kink, so
  any anchor bias goes straight into the CV price. Bonus validation kept:
  `price_geo_quad` matches the closed-form Black price in the
  flat-lognormal world to ~2bp (`test_geo_quad_matches_black_flat`), an
  end-to-end pin of the copula+marginal machinery; `test_geo_cv_mechanics`
  pins the estimator identity and the pseudo-MC variance reduction. Lesson:
  a control variate is a pseudo-MC concept — against scrambled QMC, ask
  whether the control shares the *estimator-level* error, not the pathwise
  variance (pathwise corr was 0.99 and it still lost).

## 2026-08-06 — Marginal-factory speedup claim corrected: ~11×/core, not ~50×

- The user challenged the committed "24 ms per date vs ~1.2 s CPU" claim.
  Re-measured like-for-like (same 750 marginals, both warm): CPU
  `implied_marginal` loop 273 ms on one core vs GPU `MarginalFactory` 24 ms
  → **~11× one core**, roughly throughput parity against the 20-core fork
  pool. The stale ~1.2 s was the benchmark's problem-assembly loop (pandas
  lookups included) timed under GPU contention — never a like-for-like
  baseline. The factory's real value is latency and keeping the
  build→price loop on-device, not the multiplier.
- Record: corrected in CLAUDE.md, rainbow_attribution.ipynb cell 17, and
  memory; commit 760fb8f. Full attribution run measured at 387 s / 1,968
  pairs ≈ 197 ms/pair; per-stage sync timings (tables ~145 ms + 11 Sobol
  pricings ~47 ms + fwd+backward) sum higher than the real loop because the
  loop overlaps kernel launches. Process lesson recorded in memory: never
  quote a speedup whose baseline wasn't measured in the same pass.

## 2026-08-05→06 (overnight) — Rainbow book full-history P&L attribution shipped

- The rainbow analog of the vanilla book study: sell one $1M 1Y 100/112
  ranked-basket spread every joint date (~260 vintages at steady state) and
  attribute daily P&L by independent single-input revaluations on the CRN
  Sobol engine. Full 1,968-day history runs in 6.5 min on the GPU with
  residual std 0.6% of P&L std. Book economics: premium $96M, raw P&L
  −$65.7M (unhedged short delta into the rally, eq_delta −$54.5M);
  delta-hedged −$11.2M — median day +$582 but skew −8.3: crash gamma plus a
  steady −$6.9M spot×vol (vanna) drag outrun the carry. HSI is the largest
  cumulative single-surface vol drag (−$0.8M) at the smallest daily std;
  vol_cross +$1.0M is genuine joint-move convexity, not noise (CRN-clean).
- Record: `rainbow_attribution` driver in rainbow_torch.py (single torch
  module rule) + `scripts/run_rainbow_attribution.py` (cache
  `data/rainbow_attribution.csv`) + `rainbow_attribution.ipynb` (18 cells,
  executed outputs committed; commit a841720). Components: eq (autograd
  `eq_delta` + `eq_higher`; tables frozen ⇒ sequential sticky-moneyness
  view), per-index `vol_*` + `vol_cross` (fixed τ ⇒ roll-free by
  construction; seasoned strike-slide lives in eq/cross_sv — sticky-strike
  re-cut deferred to the factor stage), `time` (+`time_roll` via two-tau),
  `fwd` (r−q carry), `rate` (pure df multiplier, zero revaluations),
  `cross_sv` (4-corner spot×vol on shared tables), `resid`. Expiries settle
  at intrinsic on the next valuation date (book.py convention); slot =
  sale-ordinal % 256; ~21 table builds + 11 pricings + 1 backward ≈ 197
  ms/date. Residual max $94k (2020-03-09); identity corr 1.000000; splits
  (eq=delta+higher, time=roll+other, vol=Σsingles+cross) exact to $0.
  Validation: seed-flip moves book components ≤~$500 calm; quad(24×24)
  cross-check agrees ~$2k/day calm/recent, widening to $79k on a $1.7M-vol
  COVID day — the quadrature's own documented crash tail, so the Sobol
  marks stand. Driver takes `engine="quad"` and `t_slice` for window
  reruns. Industry deltas flagged: desks would use local-vol MC, implied
  correlation + risk premium, t-copula tails, SVI wings, quanto drift on
  SX5E (~±40bp fwd; HSI ~nil, HKD peg).

## 2026-08-05 — GPU engines: batched quadrature, CRN Sobol, device marginal factory

- Make book×scenario work routine on the 8GB laptop GPU (RTX 4070, WSL2),
  keeping numpy as the canonical oracle. Three engines landed in one module
  (rainbow_torch.py, commit b6519a5): the batched quadrature (74× the numpy
  loop), the CRN Sobol sampler (17–22 µs/price — the fast path), and the
  device-resident marginal factory (24 ms/date for 750 tables). Together
  they turn the nested-VaR baseline (~3,138 scenarios × 260 vintages) from
  a day-scale job into ~3 min (Sobol) / ~1h (quad) on device.
- Record: `QuadBatch.from_problems(scores=False skips ppf)`;
  `price_quad_batch` mirrors price_rainbow_quad line-for-line — fp32 only
  for the (B, nodes, grid) survival/trapezoid stage, fp64 elsewhere
  (`survival_dtype=torch.float64` for full parity), chunked under
  `max_bytes` (2GiB default), gradient-checkpointed per chunk (without it,
  backward pins every chunk's survival tensor and OOMs at book scale).
  Parity: fp64 ~1e-11 vs numpy, fp32 < $0.001/option. One autograd
  backward = exact d(pv)/dg for all vintages×indices (1.4e-5 rel vs bumps),
  8.5 s for 750 deltas. `sobol_normals(n_slots, n_paths)`: one scrambled
  scipy-Sobol set per book slot as normals, generated once on CPU, device
  resident; vintage keeps its slot's points across every day and component
  reval (common random numbers) while slot scrambles are independent
  (spawned SeedSequences) — book errors √-average (verified −$200 vs
  ±$705 √-rule). `price_sobol_batch` replays the numpy sampler on identical
  normals to ~1e-11; CUDA==CPU 7e-12; 4.2–5.5 ms per 250-book at 2,048
  paths (17–22 µs/price, 26 µs sustained over 2,000 problems); cross-check
  vs quad(1,152 nodes) per-contract std $45 ≈ the calibrated $50;
  `from_problems(scores=False)` assembles a 250-book in 0.06 s (the ppf is
  the assembly's main cost). `MarginalFactory`: all dates' pillar
  grids + curve nodes upload once (~10MB); fixed knots ⇒ spline evaluation
  linear in pillar values, so the scipy not-a-knot solves bake into
  constant cardinal-coefficient tensors (searchsorted + gather + Horner)
  with an exact torch replica of np.gradient's non-uniform stencil; parity
  vs `implied_marginal`: x 3e-15 rel / cdf 7e-12 / priced <$0.01 / curves
  exact. VaR seams built in and parity-tested: `dgrid` additive pillar
  shocks, two-tau roll convention (`tau` lookup vs `tau_price`), `di_curve`
  decoupling. Inherited CPU-identical quirk: BL mean-vs-forward gap ~43bp
  on COVID-crash surfaces (calm <1bp). Accuracy calibration at the user's
  $100/contract target: quad 24×24=1,152 nodes (calm ≤$40; COVID book worst
  −$122) ≈ Sobol 2,048 scrambled paths (RMSE ≤$99 crash, ≤$54 calm), near
  parity per valuation (1.14 ms/price GPU quad vs 1.0 ms/price/core CPU
  Sobol). Structural findings: crash-surface quad convergence is
  oscillatory with a ~$150 tail even at 8,192 nodes (2,048 nodes gave
  +$798 on a deep-OTM crash vintage; node placement vs the violent
  integrand — more nodes don't reliably help, marginal-grid n 2001→4001 is
  innocent), and single-input *differences* cancel the bias (time $1–7,
  spot ≤$32 at 1,152 nodes even on the COVID book) — attribution consumes
  differences, so level bias on crash quarters is acceptable. Roles: quad =
  reproducible engine of record (zero variance, batchable shape), Sobol =
  fast cross-check/production path. Pre-build sizing, superseded by the
  6.5-min GPU driver: full-history attribution = 477,084 position-days ×
  9–13 components ≈ 4.3–6.2M valuations → ~1.4–2h GPU quad or ~10–15 min
  Sobol-2048 on 20 CPU cores; CPU oracle table build 0.4 s per 250-book.
  Throughput (48×48 nodes, 250-vintage book): numpy 331 ms/price → torch
  CPU 45 ms → GPU 4.5 ms, 222 prices/s sustained;
  `scripts/bench_rainbow_gpu.py` reruns it; rainbow_option.ipynb section 6
  documents book pricing + the per-vintage autograd delta profile (old
  vintages rallied into the cap, delta decayed — per-underlying strike
  drift). Torch 2.12.0+cu130
  via the default `gpu` group; fp64 runs at 1/64 rate on consumer cards —
  fp32-first with fp64 reductions is mandatory.

## 2026-08-05 — Rainbow-option thread starts; dataset swapped to SPX/SX5E/HSI

- New thread to study the real book's exotics directly: the single-index
  SPX history was replaced by three co-quoted surfaces, and the test
  product became a 1Y call spread (100/112) on the ranked basket
  0.5·best(SPX, SX5E) + 0.3·worst + 0.2·HSI, priced by implied marginals
  glued with a Gaussian copula under constant historical correlation.
  First valuation (2026-07-31, 400k paths, se $79): **$53.1k per $1M
  (5.31%)** — ~14% under SPX-alone at the same strikes (basket
  diversification + surrendering SPX's rich +3.7% forward carry) ≈ the
  price of an SPX 100/110, i.e. "the cheaper structure funds the wider
  cap", quantified. The price
  is near correlation-neutral (±0.2 on any pair moves it <0.6% — basket-vol
  and rank-premium channels cancel), so the constant-ρ shortcut is benign
  for this payoff. Scrambled-Sobol QMC then made paths ~400× cheaper
  (1,024 Sobol ≈ 400k pseudo), and a deterministic quadrature replaced MC
  outright for this payoff class (≤3 European underlyings): 8,192 nodes
  reproduce a 4M-path reference to $0.13 with zero seed variance.
- Record: commits f4b916b (dataset repoint) + 3a4ffd3 (rainbow.py +
  rainbow_option.ipynb + tests). Datasets: identical schema per index,
  2018-12-27→2026-07-31, 1,969 joint dates, ~17 terms/date; loader handles
  non-EOD rows, exact dupes, 6 repaired SPX far-call rows. The old
  2013→2026 `SPX_volSurface 2.csv` is gone — committed vanilla-book/VaR
  outputs reflect the longer history and shift if rerun. `implied_marginal`
  (Breeden–Litzenberger on a log-spaced performance grid): wings beyond
  50–150 must neither flat-clamp (σ kink where the put wing has vega → ~2–3%
  Dirac of teleported mass, −30bp mean bias) nor extrapolate unbounded
  (violates Lee's bound on HSI's rising call wing) — edge-slope
  continuation tapered exponentially (scale 50 mon pts, floor 0.25× edge
  vol); recovered mean matches the forward <1bp (tested).
  `historical_corr` uses 5-day overlapping log returns — daily returns
  understate cross-region co-movement under async closes (SPX–HSI 0.17
  daily → 0.43 at 5d, flat by 10d). `price_rainbow_quad`: HSI integrated
  out analytically conditional on two copula normals (conditional payoff =
  shifted call spread priced off the survival curve on the marginal grid
  via ψ = Φ⁻¹(F(x)) tables), Gauss–Hermite outer × Gauss–Legendre inner
  split at the rank kink P₁=P₂; 2,048 nodes ~4bp; small-n convergence
  algebraic, not spectral (quantile tables carry micro-kinks).
  `marginal_call_spread` = 1-D single-index
  quadrature. Both pricers take `perf_to_date` (seasoning g = S_asof/S_sale;
  strikes in sale-date units). Seasoned evidence (six vintages, τ 1.0→0.11):
  one 5,000-node rule within $0.4–4.1 of 4M-Sobol everywhere (≤0.5bp),
  where pseudo-MC needs 10⁸–10¹⁰ paths and Sobol 33k–262k+; honest caveat
  — at matched
  single-price accuracy Sobol is a few× faster on CPU; quad's edge is zero
  variance + fixed batchable shape. Rank premium 0.1·E|P₁−P₂| ≈ 1.1 perf
  pts ≈ only ~$2.9k inside the cap. Validation: smiles re-implied from
  vanillas priced on simulated paths (`bs_implied_vol`) round-trip ≤0.2bp
  construction, 3–13bp MC noise, worst ~46bp at the dying-vega 60% wing
  (notebook section 5). No FX/quanto per spec; USD (SPX-curve)
  discounting.

## 2026-08-03→04 — Goal reframe + key-rate level/skew: first PCA alternative loses

- User reframe (2026-08-03), now the governing statement: **the goal is
  cheap VaR, not PCA** — find greeks-based VaR from a handful of bumped
  revaluations; PCA is one surface representation among several, and every
  alternative gets benchmarked on the same book, scenarios and rolling
  backtest. First alternative, per user spec: desk-style key-rate
  sensitivities — fixed level and skew shapes per quarterly knot, no
  fitting anywhere, 16 revaluations. Verdict: interpretable and lean (the
  call-spread book is structurally a *skew* book), but at this simplicity
  it loses to fitted factors — attribution R² 0.705 vs PCA k=3's 0.811,
  rolling hedged VaR |gap| 7.4% with cube (8.3 plain) vs 5.6% for PCA
  k=4+cube (4.7% at k=10+cube), and the **same crisis tail**
  (2020-03-16 −67% vs −61%): the displaced-book failure is realized-move
  curvature, basis-independent. Its single-date VaR win (−0.7%) was the
  third single-date mirage of the project.
- Record: commit b345daa; `key_rate.py` + `key_rate.ipynb` +
  `rolling_var_keyrate` (4th pass in run_var_rolling.py, cache
  `data/var_rolling_keyrate.csv`). Design: two shapes per knot (3/6/9/12M,
  tent weights in TTM so level sens sum exactly to net vega; TTM<3M clamps
  onto 3M), level = parallel bump at own maturity, skew = tilt (m−100)/5
  vol pts deliberately unclamped into the wings (user); realized moves =
  deliberately simple single-point reads (sticky-strike ATM change per
  knot, (Δσ₁₁₀−Δσ₉₀)/4). Sens grouping is per-leg shape *evaluation*
  (`key_rate_sens` on u·vega / u·vanna) — never a pillar-grid
  representation (9M tent unrepresentable on TTM pillars) and free of
  bucketed-vega side-lobes; `bumped_key_rate_sens` = 16 revals, matches
  closed form 6e-4. Numbers: k=8 PCA reaches 0.927; key-rate max error
  $1.4M on 2020-03-16 (violently non-monotonic smile move — curvature
  outside level+tilt span, 3-point reads scored −$1.26M vs +$140k truth);
  rolling better on only ~⅓ of days, >10% gap on 29%; raw VaR gap 3.3% vs
  0.9% (k=10c). Measured headroom for round two: butterfly reads
  (90+110−2·ATM)/2 lift ceiling 0.705→0.790; wider 80–120 skew read alone
  0.750; short-end read nil. Book structure: ≈ +$0.9M per unit tilt vs
  $160k/pt net level vega, concentrated 6–9M.

## 2026-08-01 (evening) — Formula-free bump path: all greeks via 32 revaluations

- User decision locking the methodology to the real-book constraint:
  assume greeks exist **only** via bumped revaluations (the exotics are
  simulation-priced), 1σ-sized bumps, ≤5 vol factors affordable;
  closed-form greeks stay only as test oracles. The bump path measures
  everything in 32 revaluations per book and, over the whole history, runs
  hedged mean |gap| 6.4% vs the formula path's 5.7% at the same k=5+cube —
  with the identical −65% crisis tail (k=5 truncation, owned by the
  gate/screen). Two whole-history reversals of single-date reads: the
  spot×factor crosses are essential (the "<2% vanna" single-date ablation
  was a mirage), and the PC1² quadratic is a daily-P&L tool but *hurts*
  VaR.
- Record: commits dcdc3f4, 81bd28e. `book_reval_fn` = single
  price(spot, dsigma, dr, dq) entry with vol lookups frozen at base
  moneyness — re-mapping moneyness under spot bumps double-counts the
  slide already in sticky-strike scores and distorts crosses up to 30×.
  `bump_greeks`: 10+2k+4·k_cross = 32 revals at defaults (spot stencil
  ±1/2%, mean-move reval, five ±1σ factor pairs → exposures + free
  diagonal curvatures, 4-corner PC1-3 spot×factor crosses, ±5bp parallel
  rate/div). `bump_pnl`: rate/div must hit a |units|·TTM-weighted mean of
  per-leg moves — a plain mean lets 1/T-amplified implied-q snap noise of
  near-zero-sensitivity short legs poison the term (p99 $90k / max $1.3M
  before the fix). Dropping crosses: 166+ books < −30%; PC4-5 crosses add
  nothing; PC1² off by default for VaR (widens k=5 gaps 0.2–0.5pp) while
  lifting daily vol R² 0.860→0.875. `rolling_var_bump` + third
  run_var_rolling.py pass, cache `data/var_rolling_bump.csv`. Earlier
  exploratory layer kept: `bumped_factor_exposures`/`book_price_fn`
  (central differences along L_k/w, 2k+1 revals + free PC1 curvature;
  matches analytic exposures to 6e-6 worst date; ½c·f₁² add-on: k=10 R²
  0.937→0.946, big-move RMSE −9%, VaR hedged 4.74→4.65% with days>10%
  11.4→9.8 and k=4 5.64→5.39, doesn't touch crisis books). Bump-vs-formula
  medians at k=5+cube: 5.3% vs 4.3%.

## 2026-08-01 — VaR method review: gate + dual-screened partial reval

- Open-minded review (user ask): get greeks VaR closer to full reval on bad
  days without losing normal days. Findings: the bulk gap is mostly
  *genuine nonlinearity*, not truncation (all-factor k=104 improves mean
  |gap| only 4.74%→4.00%), while the crisis tail IS truncation
  (−61%→−14% at k=104) — so no weighting or factor-count change beats ~4%
  bulk. Two fixes that work: **gate** the headline VaR with the
  strike-histogram number (swap when it exceeds by >15%: fires 24/2,889
  days, bulk unchanged at mean 4.62%, worst −61%→−15%, clean separation),
  and
  **dual-screened partial reval** (fully revalue the top-100 scenarios of
  each projection ∪ all |dS|>4%, ~5% of the budget → +0.0% on the seven
  hardest books).
- Record: scratchpad passes, conclusions folded into var.py docs/workflow
  (not a separate module). Cheap pure-greeks fixes fail: relative floors
  on vega weights are *worse* (worst −68/−75% — exposure noise via
  E=vega/w in barely-weighted wings; the strike-histogram works via kernel
  mass at the book's strikes, not its floor); stress-day-only fitting is a
  no-op. Gate separation: bulk divergence p95 +7% vs ≥+25% on every big
  failure. Screen detail: dual rank matters because cube/gamma Taylor
  terms overshoot on ±10% dS scenarios and rank true tail scenarios as
  wins; raw-VaR ranking near-perfect (p99 coverage 47 scenarios);
  ~120–160 revals ≈ 5% of budget, ≤0.38% mean. Recommended workflow:
  daily greeks + gate; screen on alarm days.

## 2026-08-01 — Rolling VaR backtest: the cube reversal and the truncation diagnosis

- The user rejected a single-as-of-date conclusion and asked for the whole
  history — which *reversed* it: over 2,889 as-of books the third-order
  equity term (cube) helps hedged VaR at small k (~0.5pp mean for two
  extra spot bumps; best small config k=4+cube), median hedged |gap| 3.0%
  at k=10. Crisis-peak dates understate ~60% at any k ≤ 10 — diagnosed as
  **factor truncation from frozen average-book vega weights** (the
  displaced book's vega sits in far-wing pillars the weights ignore), not
  vega convexity. The statics-only adaptive weighting built in response
  (strike-histogram, user-requested, no greeks) caps the failure (worst
  −61%→−15–22%, zero books beyond −30%) but is worse in the bulk (6.6% vs
  4.7% mean) → kept as a parallel drift alarm, not the headline.
- Record: commits bd196d5/a9bf143 era; `rolling_var_backtest` (every as-of
  date ≥ 252; fork-pool, ~7 min on 22 cores; `scripts/run_var_rolling.py`
  caches `data/var_rolling.csv`), k∈{3,4,5,10} ± cube (cube =
  `book_speed`/6·dS³ off a 5-point spot stencil): hedged mean |gap|
  7.1/6.3/6.1/5.0% plain → 6.7/5.6/5.7/4.7% with cube (helps ~60% of days
  at k=3–4; raw VaR neutral-to-worse at k≤5). Books span net vega
  −$376k..+$208k (short 43% of days); worst as-of 2020-03-16 −56..−66% at
  every k≤10; k=104 −14%; refitting with own-book |bucket vega| gets k=10
  to −19%. `strike_histogram_weights` + `rolling_var_adaptive` (per-date
  PCA refit, `s{k}*` columns, cache `data/var_rolling_strike.csv`, ~35
  min): Gaussian kernel (5 mon-pts) at every leg's strike, notional mass,
  floor 0.05 of peak — lessons: full histogram not per-bucket average
  strike (averages drift to vol-dead ITM legs), floor low (0.2+ re-admits
  noisy short-dated wings; floor 1 = cov PCA); better on only 35% of days,
  k=4 collapses to 20%; divergence = refresh the vega weights.

## 2026-08-01 — Historical VaR: greeks projection vs full revaluation (the central deliverable)

- The headline question made concrete: 1-day historical VaR of the
  last-date book over 3,138 joint scenarios (spot, fixed-moneyness pillar
  Δgrid, Δr/Δq, calendar gap), full revaluation vs greeks projection.
  Hedged VaR99 $183k by full reval; greeks k=3 −9.6%, k=5 +3.8%, **k=10
  +0.9%** at ~15 valuations per position vs 3,138; all-factor −0.0%. Bulk
  R² ~0.95 at every k — k buys the *tail* (worst-1% mean |err| $58k→$23k).
  Raw VaR99 $1.24M is delta-dominated. Scenario factor scores must be
  re-anchored on the as-of surface — reusing the fitting sample's
  historical scores drops R² 0.95→0.81 and understates hedged VaR99 by 12%.
- Record: commit bd196d5; var.py: `build_book` (close-of-day legs incl.
  that day's new spread), `full_reval_pnl` (shocked grid sampled at each
  leg's new moneyness and dt-decayed lookup TTM with the BS time input
  frozen — slide and roll-down realize on the as-of surface; scenario P&L
  = pure market shock, no theta/funding), `greeks_pnl` (delta/gamma,
  bucketed vega → factor exposures, bucketed vanna paired with k-factor Δσ
  reconstruction, 1bp rho/div), `scenario_dsigma` (per-scenario pillar
  move re-anchored as-of, grid lookups only). As-of 2026-05-12 book: net
  long vega ≈ $160k/vol pt from strike drift, net short delta −$48M/100%.
  Ablations (single-date; later partly reversed): drop gamma +31%,
  vanna/rate/div <2% at the quantile; worst hedged scenarios are vol-crush
  days (2020-03-19 −$1.5M, 2018-02-07); raw VaR is made by +7–10% rebound
  days; k buys tail overlap 25→28 of the 31 worst scenarios. Notebook:
  historical_var.ipynb.

## 2026-08-01 (morning) — Bicubic interpolation flips the verdict: sticky-strike becomes the headline

- Two commits four hours apart tell the story: under the original bilinear
  lookup, sticky-strike factors *lost* to moneyness-PCA + slide (741ca72
  "headline model stays sticky-moneyness"); after the user asked for
  cubic-spline interpolation everywhere, the ranking flipped at every k
  (acbdb51) — the earlier deficit was a bilinear artifact (kinked
  bracket-crossing samples read as pillar noise and faked a ~+$9k/day
  carry drift). **Headline framework since**: vega-weighted PCA on
  sticky-strike pillar changes, target = fixed-strike vol P&L ex roll
  (`vol − vol_roll`), paired with the plain BS delta. Sticky-strike
  factors are far less spot-entangled (PC1 |corr| with spot 0.34 vs 0.86).
- Record: surface.py bicubic (cubic per axis, not-a-knot; fixed knots ⇒
  linear in pillar values ⇒ exact vega-scatter identity `pl_vega_lin ==
  bucket_vega · Δσ` kept to 1e-9; caveat: dense cardinal weights give
  bucketed vega oscillatory side-lobes, gross +34% at unchanged net — a
  factor-model input, not a hedging report). `sticky_strike_dsigma`
  (samples at m·S₀/S₁, interp="cubic"/"linear"/"pchip", 50/150 edges
  clamp). k=3 R² by weighting: vega 0.811, corr 0.646, cov 0.474 (vega
  k=10: 0.934; all-factor ceiling 0.985 vs moneyness-path 0.955).
  `include_roll=True` folds term roll in (pillar tracks fixed
  strike+expiry, day-t lookup at τ−dt) → complete basis vs attribution
  `vol` (corr 0.994, loadings identical, roll → factor mean ≈ −$5.3k/day;
  shortest pillar clamps at 0.08 TTM edge); k=3 suffices for exposure
  reporting (R² 0.82), stress days need k≈10 (big-move R² 0.66→0.83).
  `fit_pca` stores `score_std` (natural 1σ bump sizes). Notebooks:
  sticky_strike.ipynb (the comparison), roll_in_pca.ipynb,
  pca_factors.ipynb; 406a0e6 (2026-08-05) added the sticky-strike vs
  sticky-moneyness framing contrast to attribution.ipynb.

## 2026-07-30 — Project start: vanilla call-spread book, exact attribution, first PCA

- Setup and the first full loop in one day: can a handful of surface
  factors explain a realistic option book's vol P&L? Test vehicle: sell a
  $1M-notional 1Y SPX 100/110 call spread daily (~250-spread steady
  state); simulate, attribute daily P&L by independent single-input
  Black-Scholes revaluations (reconciles to the simulation exactly,
  residual ~0.8% of P&L std), fit PCA to daily pillar moves, and estimate
  vol P&L as exposure × factor move. Vega-weighting the PCA by the book's
  |bucket vega| was the first big accuracy lever. Presentation rule
  (user): never headline the `vol_surface`/`pl_vega_full` total — it and
  the smile slide are large offsetting coordinate artifacts of the
  moneyness split; quote the fixed-strike vol P&L instead (+$15.5M over
  the sample, daily std $70k).
- Record: commits 6892184..951c75c. data.py (EOD filter, dup drop,
  corrupted-upper-wing repair; log-linear discount/forward curves);
  pricing.py (Black-76 on the forward, spot greeks at fixed F/S; returns
  price/delta/gamma/vega/vanna); book.py (sequential attribution vs
  previous close; delta is the smile delta — moneyness-quoted surface ⇒
  fixed strike slides along the smile — so the sequential view is
  sticky-moneyness; vanna explicit because SPX spot-vol anticorrelation
  is a large systematic drag, −$32M over the sample); attribution.py
  (independent non-waterfall single-input revals: eq delta/gamma/higher,
  own-vol, r, q, time split by the BS PDE into time_funding vs gamma
  theta; vol splits exactly by telescoping into vol_surface + vol_roll +
  vol_slide; `vol_carry_ex` = ex-ante roll, corr 0.98; daily totals match
  simulate_book exactly; q backed out of forward-vs-spot spikes on stress
  dates is a data-snap inconsistency the div component absorbs);
  factors.py (`fit_pca(weights="cov"|"corr"|custom)`, exposures
  E_k=(bucket_vega/w)·L_k, scores f_k=L_k·((Δσ−μ)∘w), estimate =
  bucket_vega·μ + ΣE_k f_k). Notebooks: attribution.ipynb,
  pca_factors.ipynb. Python 3.14 uv project mirroring ~/vix_refactor
  minus torch/trading deps.
