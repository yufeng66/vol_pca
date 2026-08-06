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
