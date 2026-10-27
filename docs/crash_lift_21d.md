# The 21-day crash lift: the case for it as the house correlation-skew model

*2026-10-27. Ranked-basket rainbow (SPX / SX5E / HSI, B = 0.5·best + 0.3·worst + 0.2·HSI,
1Y 100/112 call spread). Code: `sigmoid_profile`, `profile_uniforms`,
`sample_performances_profile` in `vol_pca/rainbow.py`; driver
`scripts/run_rainbow_corr_profile.py` (row `crash_lift`); notebook
`rainbow_corr_profile.ipynb`; tests `tests/test_rainbow_profile.py`. Evidence
trail: the two 2026-10-27 entries in `CHANGELOG.md`. The 5-, 10- and 63-day
calibration-horizon comparisons in §5 come from scratch reruns of the same
calibration, logged there but not cached.*

---

## 1. The claim

The crash lift is a one-parameter extension of the Gaussian copula the rainbow
book is priced on today. It keeps the implied marginals exact, keeps the 5-day
unconditional correlation matrix exact, and adds one number, λ, that raises
every pair's correlation when the common market factor is in a sell-off of
about one standard deviation or worse. Calibrated to the 21-day history it

- reproduces the one feature of the realized dependence that shows on all three
  pairs, at every horizon, in both halves of the sample and with the COVID
  window removed: the crash co-movement. It models nothing else;
- removes 40% of the Gaussian copula's misfit to the 21-day binned correlation
  profile with one parameter, where the twelve-parameter alternative removes
  85% with eleven more;
- moves the price the right way on both bank-quote legs, ATM up and cap down,
  by a stable +9 to +11 bps of notional on the package across four regimes,
  about a fifth of the gap;
- leaves the hedge where it is: package deltas move by about 3%;
- costs one extra Sobol dimension and one 1-D table per index, nests the
  current model at λ = 0, and has one knob to govern.

The case rests on three choices: a dependence model rather than a level or
wing adjustment, one parameter rather than twelve, and 21 days rather than 5 or
63. Each is argued below from numbers measured in this repository.

## 2. What the method is

**Scores.** One common market factor W and three idiosyncratic shocks ε_i, all
standard normal and independent:

    Z_i = a_i(W) · W + sqrt(1 − a_i(W)²) · ε_i
    ρ_ij(W) = a_i(W) · a_j(W)

**Profile.** The loading of every index is lifted by the same factor in the
down tail:

    a_i(W) = a_i · ( 1 + λ · s((c − W) / w) ),   s = logistic,  c = −1,  w = 0.3

clipped at 0.999. The centre c and width w are read off the shape of the
realized profile (§4) and held fixed. The plateau loadings a_i are not free
either: they are solved so the unconditional Pearson correlation of the scores
equals the 5-day historical matrix (0.751 / 0.429 / 0.478), a four-step fixed
point on the base loadings. **λ is the only fitted parameter.**

**Marginals stay exact.** The raw scores Z_i are no longer normal, so their
exact CDF F_i is tabulated by Gauss–Legendre over W (split at the profile's
centre) and the performance of index i is Q_i(F_i(Z_i)) with Q_i the
Breeden–Litzenberger implied quantile. Tests pin the marginals (KS < 0.01) and
the Gaussian limit (λ = 0 gives Φ to 1e-9 and the baseline price to floating
point). The model changes the copula and nothing else; every implied vol
surface is respected exactly as it is today.

**Calibration.** Normal scores of overlapping 21-day log returns; the market
move is the mean score; quintile bins of it; the pairwise correlation inside
each bin, with a 126-day block bootstrap for its sd; the model's binned profile
from 2^17 common-random-number Sobol paths; χ² over the fifteen bins against the
bootstrap sd, minimised over λ by a scalar search. Because conditioning on the
mean score distorts within-bin correlations (the HSI pairs read −0.6 in the
middle bins of a plain Gaussian), both the data and every model are compared
through the same binning, never against the unconditional number.

**Result at λ\* = 0.247.** Plateau loadings 0.770 / 0.873 / 0.482 (Gaussian
0.821 / 0.914 / 0.522); crash-state loadings (1 + λ)·a = 0.96 / 1.00 / 0.60.

| market factor W | P(W below) | SPX–SX5E | SPX–HSI | SX5E–HSI |
|---|---|---|---|---|
| −2.5 | 0.6% | 0.96 | 0.58 | 0.60 |
| −2.0 | 2.3% | 0.95 | 0.57 | 0.60 |
| −1.5 | 6.7% | 0.93 | 0.54 | 0.58 |
| −1.0 | 15.9% | 0.85 | 0.47 | 0.53 |
| −0.5 | 30.9% | 0.73 | 0.40 | 0.45 |
| 0.0 | 50% | 0.68 | 0.38 | 0.43 |
| +1.0 and above | 84% | 0.67 | 0.37 | 0.42 |
| unconditional | | 0.751 | 0.429 | 0.478 |

Read it as: in normal and rising markets the indices are a little less
correlated than the unconditional number says (0.68 against 0.75 for
SPX–SX5E), and in a sell-off beyond one sigma they are much more (0.85 at −1σ,
0.95 beyond −2σ). The unconditional average is unchanged.

## 3. Why a dependence model, and why this sign

The bank quotes the ATM basket call about +25 bps of notional above the model
and the 112% cap call about −25 bps below it; the package gap is about +50.
Measured on the quad engine on 2026-07-31 (2026-10-22 work, logged on
2026-10-27), every level knob moves both legs the same way: correlation +0.10
gives +15 / +14, index vols +1 pt give +36 / +31. The wing tilt that fits the
cap (−0.5 vol pt per 10 moneyness points: −8 / −20) hurts the ATM, and a
level-matched t-copula has the wrong sign on both (−10 / −8). **Asymmetric
correlation is the only mid-model mechanism with the right sign on both
legs**: raising the down-side correlation by 0.10 moves ATM +11 / cap +6,
lowering the up-side correlation by 0.10 moves ATM −5 / cap −9.

The reason is where each leg's convexity sits. The ATM call's kink is at 100%,
inside the region a sell-off reaches, so its value responds to the basket's
variance in down states, and with the 0.5 / 0.3 rank weights that variance
rises with correlation. The 112% call's kink sits in the rally region, so its
value responds to up-state variance, which falls when up-state correlation is
low. A copula whose correlation is high when the market is down and ordinary
when it is up therefore raises the ATM and, through the re-matched level,
lowers the cap. That is the quote's pattern, and it is the pattern the history
shows.

## 4. Why one parameter

### 4.1 The data contain one robust feature

Realized 21-day profile, excess correlation over a Gaussian copula at the same
level under the same binning, with bootstrap sd:

| quintile (mean W) | SPX–SX5E | SPX–HSI | SX5E–HSI |
|---|---|---|---|
| worst (−1.20) | **+0.27** (0.12) | **+0.32** (0.16) | **+0.25** (0.17) |
| 2nd (−0.39) | −0.03 (0.17) | +0.01 (0.08) | +0.01 (0.10) |
| 3rd (+0.04) | −0.12 (0.16) | +0.04 (0.07) | +0.02 (0.10) |
| 4th (+0.43) | −0.05 (0.15) | −0.04 (0.06) | −0.01 (0.09) |
| best (+1.13) | +0.00 (0.07) | −0.15 (0.07) | −0.04 (0.11) |

Three things stand out. The worst-quintile lift is present on all three pairs
and is the same size on all three, +0.25 to +0.32. The middle three quintiles
are Gaussian within noise. The best quintile is Gaussian on two pairs and shows
a 2-sd dip on one, SPX–HSI. The crash lift is the model of the first two
observations and a deliberate decision not to model the third.

A single shared λ on the common factor is not a simplification imposed on the
data; it is what a uniform lift across pairs *means*. If one market factor
drives the co-crash, every pair's correlation rises together through its
loadings, and that is the structure the worst quintile displays.

### 4.2 The feature survives every cut; the others do not

| sample (21-day bins) | λ |
|---|---|
| full sample | 0.25 |
| first half (to 2022-10) | 0.25 |
| second half (from 2022-10) | 0.12 |
| ex-COVID (2020-02..06 dropped) | 0.11 |

Without March 2020 the worst-quintile excess is +0.15 / +0.17 / +0.16 against
+0.27 / +0.32 / +0.25 with it: the lift halves but it does not disappear, and
it stays uniform across the three pairs. The χ² objective is flat in λ between
0.16 and 0.30 on the full sample (11.1 to 11.4 against 17.5 at λ = 0), so the
fitted value is 0.22–0.25 depending on the bootstrap weights; the package
adjustment moves about 3 bps either side of its +11 across that range.

The 21-day exceedance asymmetry (correlation when both indices are below
their 30th percentile minus when both are above the 70th) is +0.34 with a 90%
bootstrap band of [+0.10, +0.52]; the half-split asymmetry (down half minus up
half of the mean score) is +0.19 with sd 0.07. A Gaussian copula gives zero on
both statistics by construction. The feature is
statistically there and economically obvious; its *size* is the uncertain
part, and that uncertainty is exactly what a one-parameter model carries in
one number.

### 4.3 What the alternatives cost

| candidate | parameters | χ² on 15 bins | half-split SPX–SX5E (down \| up) | comment |
|---|---|---|---|---|
| Gaussian (5d) | 0 | 18.3 | 0.51 \| 0.51 | no asymmetry at all |
| step 0.90 / 0.60 | 2 | 12–15 | 0.68 \| 0.39 | discontinuous ρ(W); lowers the up side the data do not show for SPX–SX5E |
| tanh λ = 0.3 | 1 | 41.8 | 0.79 \| 0.25 | antisymmetric by construction; overshoots every horizon |
| **crash lift** | **1** | **10.8** | **0.63 \| 0.42** | realized 0.59 \| 0.45 |
| sigmoid, rank-fit | 12 | 2.7 | 0.59 \| 0.44 | per-index down lift, per-index up cut, fitted centres and width |

The twelve-parameter sigmoid is the best fit, and that is the problem with it.
Twelve parameters on fifteen bins leave three degrees of freedom; a χ² of 2.7
is what pure noise fitted by a flexible curve looks like. Its extra parameters
are spent on the HSI-specific residuals: the under-lift of the HSI pairs in the
worst quintile (−1.2 and −1.1 sd under the crash lift) and the SPX–HSI
best-quintile dip (+1.8 sd). Those are one-pair, 1–2 sd features, and the
up-tail cut among them is the one that matters for price (§8). The crash
lift's residual misfit is small, localised and named; the sigmoid's fit is a
description of this sample. It was called "a fit, not a model" at the time, and that is the right
label.

The step is the natural first idea and the crash lift is its repair: it keeps
the step's economics (a different correlation in a crash) and fixes its two
defects, the discontinuity in ρ(W) and the assumption that the up side differs
from the middle, which the SPX–SX5E data reject (best-quintile excess +0.00).

### 4.4 One parameter is a governance property

λ has a plain meaning ("in a crash beyond one sigma, the market-factor loadings
rise by 25%; SPX–SX5E correlation goes from 0.68 to 0.95") and a plain
stress ladder (§10). A twelve-parameter profile cannot be explained to a
committee, cannot be stressed one axis at a time, and will change shape every
time it is refitted. A hedge or reserve that rests on one number can be
monitored; one that rests on twelve cannot.

## 5. Why 21 days

The question is what co-movement the book actually experiences. Hedged P&L is,
to leading order, the gamma matrix times the realized co-movement over the
interval at which the hedge can respond. That interval is set by two things:
the closing-time structure of the three markets, and how long a joint sell-off
takes to play out.

### 5.1 Daily is wrong, and a week is the floor

The hedge ratio set after the New York close is stale through Hong Kong's
catch-up gap the next morning, so the lag-one cross-product is delivered to the
book as delta P&L on the gap. The hedge sees the covariance *including* the
lead-lag:

| pair | same-day daily corr | next-day catch-up | lag ±2 | 5-day corr | 21-day rank corr |
|---|---|---|---|---|---|
| SPX–SX5E | 0.57 | +0.13 | ≤0.15 | 0.75 | 0.69 |
| SPX–HSI | 0.17 | +0.23 | ≤0.04 | 0.43 | 0.39 |
| SX5E–HSI | 0.34 | +0.19 | ≤0.06 | 0.48 | 0.47 |

For SPX–HSI more than half of the two-day cross-covariance arrives the next
Hong Kong day. Lags beyond two days add nothing. Daily correlation would
misprice the terminal distribution; a week is the shortest window that
contains what the hedge experiences.

### 5.2 For the *level*, nothing changes beyond a week

Rank correlation of SPX–SX5E at 5 / 21 / 63 days: 0.717 / 0.743 / 0.687,
sampling sd 0.02 / 0.045 / 0.08 (393 / 93 / 31 independent windows). Flat
within noise. The 21-day *Pearson* figure of 0.823 is March 2020, not a horizon
effect; it is the tail dependence counted into the level. That is why the level
is held at the 5-day matrix (§6) and only the *shape* is taken from 21 days.

### 5.3 For the *shape*, the crash co-movement builds over weeks

Realized statistics by horizon (normal scores of overlapping h-day returns),
with the crash-lift λ fitted to each horizon's own quintile bins using that
horizon's bootstrap sd:

| horizon | independent windows | worst-quintile excess, SPX–SX5E / SPX–HSI / SX5E–HSI | half-split asymmetry | λ | χ²: Gaussian → crash lift |
|---|---|---|---|---|---|
| 5d | 393 | +0.16 / +0.22 / +0.20 (sd 0.08–0.10) | +0.08 | 0.08 | 29.6 → 25.7 |
| 10d | 196 | +0.21 / +0.11 / +0.20 (sd 0.12–0.20) | +0.11 | 0.08 | 19.7 → 18.0 |
| 21d | 93 | +0.27 / +0.32 / +0.25 (sd 0.12–0.17) | +0.19 | 0.22–0.25 | 17.6 → 11.0 |
| 63d | 31 | +0.46 / +0.27 / +0.07 (sd 0.16–0.25) | +0.30 | 0.50 | 23.6 → 12.5 |

The crash co-movement is visible at every horizon, but it is not the same
object at every horizon. At 5 and 10 days the lift is small and the
one-parameter shape barely improves on the Gaussian (χ² down 9–13%): the
week-scale co-movement is mostly the lead-lag catch-up of §5.1, which the
level already carries. At 21 days the lift has built to its full size,
uniformly across the three pairs, and the shape removes 38% of the Gaussian
misfit. At 63 days the mean lift is the same (+0.27 against +0.28) but it has
migrated into one pair. Contagion is not a one-day event: a joint sell-off
runs for weeks (February–March 2020 was four; the 2022 legs were multi-week),
and the same stale-hedge mechanism delivers the cross-gamma P&L to the book
across the whole run. Twenty-one days is the shortest horizon at which the
feature has fully formed and the longest at which it can still be measured
with ~90 independent windows.

### 5.4 Why not 63 days

The 63-day calibration roughly doubles the covered share of the quote gap
(crash lift +18 bps on the package against +10 at 21 days), and that is a
reason to look hard at it, not a reason to use it:

- **It is within noise of the 21-day read.** Thirty-one independent windows;
  bootstrap sd 0.14–0.24 in the tail bins; the 63-day asymmetry of +0.30 has a
  band that contains the 21-day +0.19.
- **The structure reshuffles, it does not scale.** The 63-day worst-quintile
  excess is +0.46 / +0.27 / +0.09 across the three pairs, against the even
  +0.27 / +0.32 / +0.25 at 21 days; without March 2020 the HSI pairs' lift is
  +0.04 and −0.09, gone. A true horizon effect would scale the pattern. A
  reshuffle is what sampling noise does, and a shared-λ model is the wrong
  model for a lift that lives in one pair.
- **The level itself is unstable at 63 days.** The rank correlation moves
  from 0.69 to 0.62 when one episode is removed.
- **The model saturates.** At the 63-day λ of 0.50 both SPX and SX5E loadings
  hit the cap below −1.5σ: correlation 1.00 in any sell-off worse than about one
  in fifteen. A fit sitting on the boundary of the parameter space is a fit
  the model cannot express, and the next refit cannot move it further.
- **Choosing the horizon by how much of the quote it covers is fitting to the
  quote by the back door.** If matching the dealer is the objective, calibrate
  λ to the quote directly and use the history as the plausibility band, with
  21 days as its centre and 63 days as its upper edge.

The fair point for 63 days, that the SPX–SX5E lift survives ex-COVID there
(+0.40) because the 2022 bear-market quarters count as joint sell-offs at that
horizon, is why the 63-day λ belongs in the stress ladder rather than in the
bin. The 21-day calibration is the centre of the band on every statistic
measured: horizon ladder of the lift (5d below, 63d above within noise),
sub-sample λ (0.11–0.25), and pair structure (uniform at 21d, lopsided at 63d).

## 6. Why the level stays at 5 days

Separating level from shape puts each at the horizon where it is best
measured: the level at 5 days (393 windows, sd 0.02, lead-lag absorbed), the
shape at 21 days (§5). It also avoids a double count. The 21-day Pearson level
of 0.823 is high *because* of the co-crash; a model that used it as the level
and then added a crash lift on top would count March 2020 twice. Holding the
unconditional matrix at the 5-day estimate and letting λ carry the tail is the
clean decomposition, and it is why the plateau sits below the Gaussian level
(0.68 against 0.75): the lift is paid for inside the same average.

The practical consequence is the one that matters for the desk: **the crash
lift is a pure shape overlay on the hedge model.** Every price difference
between the two rows is attributable to the tail shape alone, and the hedge
barely moves: package deltas by ±1% bump of each index's performance-to-date
rise by 2.8–3.2% per index (4% on the COVID date). The Gaussian copula at the
5-day level remains the hedge target, the quad engine of record applies to it,
and the crash lift sits above it as a valuation adjustment that can be
switched off by setting λ = 0.

## 7. Fit evidence

Model minus realized within-bin correlation, in units of the bootstrap sd of
that bin. These are the quantities the χ² sums; a correct model scatters within
about ±1:

| quintile | SPX–SX5E Gaussian | crash lift | SPX–HSI Gaussian | crash lift | SX5E–HSI Gaussian | crash lift |
|---|---|---|---|---|---|---|
| worst | **−2.1** | +0.3 | **−1.8** | −1.2 | **−1.7** | −1.1 |
| 2nd | +0.3 | +0.5 | +0.1 | +0.0 | −0.4 | −0.4 |
| 3rd | +0.9 | +0.6 | −0.4 | −0.3 | −0.5 | −0.4 |
| 4th | +0.4 | −0.2 | +0.9 | +1.2 | −0.2 | −0.1 |
| best | +0.2 | −1.5 | **+2.3** | +1.8 | +0.1 | −0.4 |
| χ² contribution | 5.5 | 2.9 | 9.7 | 6.2 | 3.2 | 1.7 |

The Gaussian's misfit is the worst quintile on all three pairs (−2.1, −1.8,
−1.7 sd) plus the SPX–HSI best quintile (+2.3). The crash lift closes the worst
quintile on the SPX–SX5E pair, the one that matters for price (SPX and SX5E
carry 13 and 11 bps per vol point of the cap's vega against 3.5 for HSI), and
reduces the HSI pairs' shortfall there by a third: one shared λ cannot lift a
low-loading index as much in correlation terms as a high-loading one, and the
remaining −1.2 / −1.1 sd falls where the book's sensitivity is smallest. The
residuals it introduces are the SPX–SX5E best quintile (−1.5 sd: the plateau
at 0.68 sits below the realized up-side 0.74 because the level re-match pays
for the lift inside the 5-day average) and the SPX–HSI upper bins it does not
model (+1.2, +1.8). Every residual of the crash lift is named and is under
2 sd; the Gaussian has three above 1.7 and one at 2.3.

Summary statistics against the realized 21-day set (half-splits on the sign of
the mean score, Gaussian under the same conditioning):

| statistic | realized | Gaussian | crash lift |
|---|---|---|---|
| χ² over 15 bins | | 18.3 | 10.8 |
| SPX–SX5E down half \| up half | 0.59 \| 0.45 | 0.51 \| 0.51 | 0.63 \| 0.42 |
| SPX–HSI | 0.15 \| −0.10 | 0.02 \| 0.02 | 0.07 \| −0.02 |
| SX5E–HSI | 0.23 \| 0.07 | 0.09 \| 0.09 | 0.15 \| 0.05 |
| mean asymmetry over pairs | +0.19 (sd 0.07) | 0.00 | +0.14 |

The mean asymmetry across pairs, the one number a committee would ask for,
moves from zero to +0.14 against +0.19 realized, inside one sd. The SPX–SX5E
pair's own asymmetry is +0.21 against +0.14; the shortfall on the mean is the
HSI pairs again.

## 8. Price impact and stability

Moves versus the Gaussian 5-day baseline, bps of notional, 2^20 CRN Sobol
paths (the two rows share their Sobol points, so each difference is a
low-noise number):

| date | book regime | ATM | cap | package | share of +50 |
|---|---|---|---|---|---|
| 2026-07-31 | current, deep ITM | +5.6 | −5.3 | +10.9 | 22% |
| 2022-03-16 | bear, short gamma | +2.6 | −6.6 | +9.2 | 18% |
| 2024-10-08 | rally | +5.2 | −5.1 | +10.3 | 21% |
| 2020-03-16 | COVID peak | +2.4 | −8.6 | +11.0 | 22% |
| 2026-07-31 at the ex-COVID λ = 0.11 | | +2.8 | −2.6 | +5.4 | 11% |

Four properties worth stating:

- **Right sign on both legs on every date**, including the COVID surface,
  where the Gaussian-21d row (level only) moves both legs up and the package
  by nothing. No level change of any kind produces the ATM-up / cap-down
  wedge.
- **The package effect is flat across regimes**, +9 to +11, while the split
  between legs moves with the book's moneyness. That is what a dependence
  effect should look like: the dispersion exposure is a property of the
  rank-weighted payoff, not of the date.
- **It is bounded.** The shape saturates: SX5E's crash-state loading is already
  at the cap at λ = 0.25, SPX's reaches it at about 0.35, and from there only
  HSI's loading responds. No λ reaches the +50 package gap, and the 63-day λ of 0.50
  stops at +18. That is a finding about the quote, not a weakness of the
  model: the gap is not a correlation-skew story at any horizon the data
  support, and the model says so instead of being bent until it fits.
- **The residual is the size of a bid/offer.** After the crash lift the
  unexplained gap is about 19–20 bps per leg; at 38 / 34 bps of vega per
  basket-vol point that is 0.5–0.6 vol points on each leg, the same number on
  both, which is what a margin that scales with vega looks like. Compare the
  legs in vol points and ask for a package quote; treat the rest as an overlay
  calibrated across banks and dates.

The difference between the one-parameter model and the twelve-parameter one on
the package is 1 bp (+10.9 against +11.9 on the quote date). Eleven parameters
buy a different leg split (the sigmoid moves the cap −10 and the ATM +2 through
its HSI up-tail cut) and nothing on the package.

## 9. Cost and footprint

- **Sampling.** One extra Sobol dimension (W, ε₁, ε₂, ε₃ instead of three
  correlated normals) and, per index, one 1-D CDF table built by Gauss–Legendre
  (2 × 600 nodes × 3,001 z points, milliseconds with `scipy.special.ndtr`)
  applied as a single `np.interp` before the implied quantile. The notebook
  runs in about 10 s; a 2^20-path price takes seconds on one CPU core; the
  book's 512-path marks are unchanged in cost.
- **Common random numbers.** Gaussian and crash-lift prices share Sobol points,
  so the adjustment and its λ sensitivity are smooth, low-noise differences —
  the same CRN property the attribution and greeks drivers rely on.
- **Engines.** The quad engine of record conditions HSI out through the
  Gaussian structure and does not apply; the production path for the book is
  already CRN Sobol (standing default 512 paths). The torch mirror
  (`rainbow_torch.py`) needs the per-index table lookup added to its sampler, a
  searchsorted-and-gather on the raw scores in the same pattern the marginal
  factory already uses; not built, small. The one-factor structure also makes
  the three indices conditionally independent given W, which is the same route
  the quadrature engine takes, so a deterministic version is possible if ever
  needed.
- **Calibration.** The full driver (two sigmoid fits, the crash-lift fit with
  three sub-sample λ's, five rows on four dates at 2^20 paths) takes about
  6 minutes of CPU; the crash-lift fit alone is a scalar search over 2^17-path
  profiles, well under a minute.
- **Tests.** Loadings reproduce any positive 3×3 pairwise target exactly; the
  constant profile recovers Φ to 1e-9 and the Gaussian Pearson within 0.01;
  every marginal passes KS < 0.01 under the profile; the per-index strengths
  match the closed-form formula.

## 10. Robustness as a hedge target, and governance

- **Nests the current model.** λ = 0 is the Gaussian copula to floating point.
  Nothing downstream (greeks drivers, attribution, VaR) has to change to carry
  it; the adjustment is a second pricing with the same points.
- **The hedge does not move.** Deltas within 3% (§6). A model whose
  recalibration shifts the hedge target is a model that churns the hedge
  every refit; this one does not, which is precisely why it can be carried as
  a reserve while the Gaussian stays the hedge model.
- **One knob, one ladder.** Package adjustment on the quote date at λ = 0 /
  0.08 / 0.11 / 0.25 / 0.50: +0 / +4 / +5 / +11 / +18 bps. Those five points
  are "off", "week-scale", "ex-COVID", "full sample" and "63-day edge" (on
  2022-03-16: +0 / +3 / +5 / +9 / +16). The ladder is the reserve band,
  the plausibility band for any quote-implied λ, and the stress test, all from
  one parameter.
- **Interpretable and monitorable.** λ answers one question: how much more do
  the indices load on the market factor when it is in a crash? It can be
  refitted on a rolling 21-day profile and is expected to live between about
  0.1 and 0.3 depending on whether a major drawdown is in the window. A move
  outside that range is a signal, not noise.
- **Exact marginals.** The market-implied vol surfaces are respected exactly,
  so vega, the strike-shift greeks and the sticky-strike machinery all carry
  over unchanged; only the historical input (the copula) is touched, and that
  is already the one historical input the current model has.
- **Fails safe.** Because the shape saturates (§8) the model cannot be pushed
  to an absurd price by an extreme λ; the worst it can say is "perfect
  correlation in a crash", which is a defensible stress in its own right.

## 11. What the case does not claim

- **It does not close the quote gap.** It explains about a fifth of the
  package and a fifth to a quarter of each leg. The remainder, 0.5–0.6 vol
  points per leg, is the same size on both legs and is best read as the
  dealer's margin. No correlation-skew model the data support closes it: even
  the 63-day twelve-parameter fit stops at 47%.
- **λ depends on whether March 2020 is in the window.** 0.25 with it, 0.11
  without; the first half of the sample gives 0.25, the second 0.12. The
  feature itself survives every cut; its size halves. This is the argument for
  carrying it as a reserve with a ladder rather than as a recalibrated hedge
  target, and the ladder brackets it.
- **It has no up-tail cut.** The SPX–HSI best-quintile dip of −0.15 (2 sd, one
  pair) is left out. Modelling it is worth about 5 bps on the cap and 1 bp on
  the package, and costs eleven parameters in the current form. If it survives
  more data it can be added as a second parameter, a shared up-tail cut, in
  the same functional form; the code already takes it.
- **Monthly dependence is applied to a one-year payoff.** That is true of every
  candidate including the Gaussian copula; the copula is a terminal-dependence
  assumption and no horizon of daily data measures one-year dependence with
  usable precision (about eight independent windows). Twenty-one days is the
  longest horizon at which the shape can be measured, which is a different
  statement from the right horizon for the product.
- **The binned correlations are conditional statistics.** Conditioning on the
  mean score distorts within-bin correlation, so the realized profile is only
  meaningful against the Gaussian benchmark under the same binning, which is
  how every χ² and excess above is computed.

## 12. Recommendation

1. **Hedge target:** the Gaussian copula at the 5-day level, unchanged.
2. **Valuation / reserve:** the 21-day crash lift at λ = 0.25, carried as a
   correlation-skew adjustment line, with the λ ladder 0 / 0.08 / 0.11 / 0.25 /
   0.50 reported alongside it as the band.
3. **Quote comparison:** state each leg in vol points; after the crash-lift
   adjustment the two legs sit about 0.5–0.6 vol points from the dealer, the
   same on both, which is the number to negotiate on or to reserve as margin.
4. **Refit** λ on the rolling 21-day profile at each annual model review, or
   when a new joint drawdown adds windows; a λ outside 0.1–0.3 triggers a look
   at the pair structure (uniform lift, or one pair?) before it is adopted.
5. **Do not** adopt the twelve-parameter profile or the 63-day calibration:
   the first fits this sample, the second fits one episode and sits on the
   model's boundary. Both are retained in the repository as the comparison
   set.

---

### Appendix A. Parameters of record

| item | value |
|---|---|
| functional form | `sigmoid_profile(a, lam_dn=λ, lam_up=0, c_dn=−1.0, width=0.3)` |
| λ (21-day bins, full sample) | 0.2466 |
| λ sub-samples | ex-COVID 0.113; first half 0.253; second half 0.118 |
| λ by calibration horizon (scratch) | 5d 0.08; 10d 0.08; 21d 0.22–0.25 (flat objective 0.16–0.30); 63d 0.50 |
| level target (5-day Pearson) | 0.7507 / 0.4290 / 0.4776 |
| plateau loadings | 0.7704 / 0.8729 / 0.4821 |
| Gaussian loadings | 0.8212 / 0.9142 / 0.5224 |
| χ² (15 bins) | 10.77 (Gaussian 18.3) |
| half-split down \| up | 0.631 / 0.072 / 0.147 \| 0.419 / −0.023 / 0.045 |
| cache | `data/rainbow_corr_profile_fit.json["crash_lift"]`, `data/rainbow_corr_profile.csv` row `crash_lift` |

### Appendix B. Reproduce

    uv run pytest tests/test_rainbow_profile.py          # ~10 s
    uv run python scripts/run_rainbow_corr_profile.py    # ~6 min CPU, both caches
    uv run jupyter nbconvert --to notebook --execute rainbow_corr_profile.ipynb   # ~10 s

The notebook's section B puts the crash lift beside the six other candidates
(ρ(W) curves, binned fit, χ² table); section C has the price table, the
(ΔATM, Δcap) scatter against the bank's (+25, −25), and the by-date package
plot.
