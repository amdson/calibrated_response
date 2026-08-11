# Solver / constraint-language additions

Running notes on possible additions to the equation + proposition language and the
maxent flow solver. Status: **shipped** / **proposed** (designed, not built) /
**open** (identified, undecided).

---

## 1. Inequalities in the equation language — *shipped*

`y > 1.5*x ~ N(0, 0.15)` alongside `y = 1.5*x ~ N(0, 0.15)`.

Different in kind from an equality: an equality **moment-matches** the residual
`r = lhs - rhs` (mean 0, var σ²) so the cloud hugs a line; an inequality is a
**support restriction**, `E[relu(-r)²] → 0` hinged per sample, so maxent spreads
over the whole admissible region. The `~ N(0, σ)` tail therefore changes meaning:
for `>` / `<` it sets how *softly* the boundary is enforced, not a residual
spread. No tail = as sharp as a deterministic identity (`eqn_rel_sd`).

Reuses `relation` on the `Estimate` base (previously ignored for equations), so no
new field. New constraint kind `eqn_ineq`, reusing the existing `_hinge_pen`.
Demo: `toy_inequality_flow.ipynb` (half-plane vs. constant-width band; the band
has an analytic answer — `x` uniform, residual uniform on `[0, w]`, entropy
`log w` — so `b.entropy()` is directly checkable).

## 2. Binary variables — *proposed*

Two classes, very different, and only one is pathological:

- **Derived** (`record = ind(anomaly > 1.60)`): a deterministic threshold function
  of a continuous coordinate. Pathological *as a flow coordinate* — a smooth
  invertible map can't track a step, so it evacuates mass from the threshold band
  (carving the upper tail), and the two-atom target fights the exact differential
  entropy term (H → −∞).
- **Exogenous** (a real yes/no event with an elicited `P`): the current
  `ContinuousVar(name, 0, 1)` site read out at `x > 0.5` is *awkward* (wasted
  capacity, soft-readout leakage) but **not degenerate** — nothing forces the
  density to concentrate at the atoms.

Proposal: **lower derived binaries out of the sampler.** No flow coordinate;
register the definition as an alias and substitute it into every statement
(`P(A > 10 | record = true)` → `P(A > 10 | anomaly_2026 > 1.60)`; at query time
it's just a sample filter). Keep exogenous binaries as-is for now. Held in
reserve: a **conditional-Bernoulli head** (continuous vars in the flow, each
exogenous binary a Bernoulli whose logit is a smooth function of the flow sample
— correlations for free, closed-form entropy, no coordinate forced bimodal).

General rule this crystallises: *a binary as a soft feature inside an aggregate
constraint (probability, conditional, correlation) is fine; a binary as a
materialized coordinate pinned by a per-sample identity is pathological.*

## 3. Joint (conjunctive) propositions — *proposed*

`P(x > 5 & y > 5) = 0.2`. Conjunction already works on the **conditioning** side
(`P(z > 1 | x > 5, y > 5)` — comma-split, joined by `_and_all`); only the *event*
side is limited to a single proposition.

Do **not** add a joint `ind()` form to the equation language: `ind(x>5)*ind(y>5)`
already is the probabilistic t-norm, and it has better gradients than a `min`-based
alternative (both factors get gradient everywhere). The gap is that the product
only reaches a *probability target* via a materialized binary — i.e. through
addition #2's pathological path.

Benefits over the workaround: the log-odds penalty (joint targets are small,
exactly where an absolute-scale residual penalty leaks); an exact hard-indicator
readout in `constraint_report`; no rare-conditioning variance (a chain-rule
`P(B)·P(A|B)` has effective sample size `N·E[cond]`, a direct joint is a plain
mean); no extra dimension. Implementation is small — widen
`ProbabilityEstimate.proposition` to a conjunction, point `_probability_loss` at
the existing `_and_all`, add `&` to the parser (`|` is taken by conditioning).

Boundary: conjunction yes, negation free (flip the inequality), **OR not
reachable** (needs negation of a conjunction). Ship conjunction, stop there.

## 4. Compound LHS in equations — *open*

The RHS is fully general arithmetic (quadratics, products of many variables,
variable exponents, rational functions, `abs`/`min`/`max`, indicators on compound
expressions, indicators as regime gates). The **LHS must be a bare variable
name** — enforced twice, by the parser regex and by `_equation_loss`'s
`span(lhs)` lookup for the tolerance.

So there is no way to state a constraint *between two compound expressions*:
`price*units > fixed_cost + 3*labor` is inexpressible. Bites hardest on the new
inequalities. The auxiliary-variable workaround (`prod = x*y`, then constrain
`prod`) walks back into #2.

Cost to lift: modest. `compile_residual` already parses `(lhs) - (rhs)`
symmetrically and handles a compound LHS fine; needs the regex plus a scale for
the tolerance (estimate the residual span from a sample batch, or require an
explicit `~ N(0, s)`).

## 5. Missing math functions — *open*

No `log`, `exp`, `sqrt`, `sum`, or conditional expressions. Two have clean
workarounds — `x**0.5` is the square root, `2.718281828**x` the exponential
(constant base compiles fine). **`log` has no workaround**, which matters for
additive-in-log-space multiplicative models. Cheap to add to the `ast` whitelist.

## 6. Indexed families / time-series structure — *proposed*

Declare an index set over a family of variables (`X[t]` for `t` in a time list),
then quantify a constraint over it. Instances: variogram
`Var(X[t] - X[s]) = γ(t-s)`, monotonicity `X[t+1] > X[t]` (newly expressible via
#1, and well-behaved — a support restriction over the ordered cone), AR(1) mean
reversion, bounded volatility `|X[t+1] - X[t]| < c`, trend `E[X[t]] = f(t)`.

**Tractable, cheaply.** A pairwise increment variance is just `eqn_dist` (mean +
second moment of a residual; low-variance sample means, no rare-event
conditioning). The whole `O(T²)` family collapses into one second-moment matrix —
`Var(X_i - X_j) = M_ii + M_jj - 2 M_ij` — so it's a single matmul per step,
`O(N·T²)`. Worth one *fused* constraint kind rather than `T(T-1)/2` closures.

**But the all-pairs form is usually unnecessary.** Constrain only *consecutive*
increments (`O(T)` statements, writable today) and maxent's completion makes them
independent — same mechanism as `eqn_dist` giving noise independent of the RHS —
which *generates* the full `Var(X_t - X_s) = α²|t-s|` law across all pairs for
free. Explicit pairwise constraints earn their keep only for **non-Markovian**
structure (long memory, mean reversion, `H ≠ ½`), where independent increments is
the wrong completion.

**Gotcha — the degenerate law.** `sd = α·Δt` (variance `∝ Δt²`) is *not* diffusive
growth: it is uniquely `X_t = A + t·B`, a line with random level and slope.
Increments perfectly correlated, joint of **rank 2 in T dimensions**, so no
density and `H = −∞` for `T > 2` — the flow fights it exactly like the derived
binary in #2, settling on a thin pancake and quietly violating the constraint. In
the fBm family `sd = α·Δt^H`, that spec is `H = 1`, the degenerate endpoint;
`H = ½` is the random walk, `H ∈ (0,1)` full-rank and well-conditioned.

**Free feasibility check.** Since the spec *is* a target second-moment matrix,
validate before fitting: the implied covariance must be PSD (variogram
conditionally negative definite). One eigenvalue check catches contradictory or
degenerate elicitations — including the `H = 1` rank-2 case — instead of burning a
fit and reading a confusing partial violation.

## 7. Expression-valued noise sd — *proposed*

Allow `~ N(0, <expr>)` where the sd is an arithmetic expression over the
variables, not just a constant. `eqn_dist` then standardises per sample:
`z = r/s(x)`, pinning `mean(z) = 0`, `mean(z²) = 1`.

Small change — `compile_residual`'s `_compile` already handles arbitrary
arithmetic, so `sigma` goes from a float to a closure; needs a positivity floor
on `s(x)`, a widened `NOISE_PATTERN` (currently numeric-only), and
`noise_sd: float -> Union[float, str]`.

Unlocks two things at once:

- **Proportional / relative error** — `y = f(x) ~ N(0, 0.1*abs(f(x)))`, i.e.
  "accurate to ±10%". Independently the most common Fermi noise model, and
  currently inexpressible.
- **Heteroscedastic-in-time** — the single-time-variable representation below.

## 8. Single time *variable* vs. site-per-time — *open (representation choice)*

Wanting `X|T=t1 - X|T=t2 ~ N(0, α√(t1-t2))` with **one** `(T, X)` pair rather
than a site per time. Three distinct representations, and the choice is upstream
of any syntax:

1. **Site-per-time** (`X_t1..X_tT`) — full joint process, path queries,
   cross-time conditioning. Costs `T` dimensions, answers only on the grid.
2. **Single `(T, X)`** — 2 dimensions regardless of resolution, continuous in
   `t`. But pins only the *marginal family* `p(x|t)`; **no path structure**.
3. **Basis coefficients** (`β_1..β_K` as sites, `X(t) = Σ β_k φ_k(t)`
   reconstructed in post) — full joint + path queries + continuous `t` in `K`
   dimensions. Same "don't materialize derived quantities as coordinates, read
   them off in post" principle as #2. Restricted to the span of the basis, which
   can't represent genuine Brownian roughness — usually irrelevant for fan charts.

**The statement is under-determined in representation 2.** `p(x,t)` fixes each
conditional `p(x|t1)`, `p(x|t2)` but says nothing about their **coupling**, and
the difference's law depends entirely on that. The default independent-draw
coupling *is* estimable from samples — but it is demonstrably the wrong one:
it gives `Var(X_t1 - X_t2) = Var(X|t1) + Var(X|t2)`, which stays large as
`t2 -> t1` where the truth goes to 0.

Counterexample worth remembering: `X_t = W_t` (Brownian) and `X_t = sqrt(t)·Z`
(one draw `Z ~ N(0,1)`, fixed for all `t`) have **identical** marginals `N(0,t)`
at every `t`, hence identical `p(x,t)`. One has rough independent increments; the
other is rank 1, every path a deterministic `sqrt(t)` shape scaled by one number.
Representation 2 cannot tell them apart.

A process is fixed by its finite-dimensional distributions; one-dimensional
marginals are the weakest slice. For a Gaussian process, mean + covariance
function suffices — and representation 2 supplies the mean function and only the
**diagonal** of the covariance. The off-diagonal is exactly what generates paths.
So anything path-dependent (`P(X_t2 > X_t1)`, crossing times, updating one time
from an observation at another) is permanently out of reach, and "sampling a
path" from it yields independent scatter, not a trajectory.

**Representation 3 is the fix, and it closes the loop with #6.** With
`X(t) = Σ β_k φ_k(t)`, the `β` joint induces a full covariance function
`C(s,t) = Σ_jk Cov(β_j, β_k) φ_j(s) φ_k(t)` (the KL-expansion view), so paths are
sampled by drawing `β` once and evaluating. Every time-series constraint then
becomes a smooth function of the `β` sample batch — e.g.
`Var(X(t) - X(s)) = Σ_jk Cov(β_j,β_k)(φ_j(t)-φ_j(s))(φ_k(t)-φ_k(s))` — i.e. the
same cheap Gram-matrix moment constraint from #6, in `K` dimensions and
continuous in `t`, with no site per time.

**But that marginal law is very tractable** — and it needs no conditioning at
all. Standardise: `z = (X - μ)/sqrt(σ₀² + α²·T)`, pin `mean(z) = 0`,
`var(z) = 1`. Two unconditional sample means over all `N` samples: no binning, no
soft window on a continuous `t`, no `N·E[cond]` effective-sample-size problem.
Requires #7 (expression-valued sd) to state. Maxent's completion then makes
`z ⊥ T`, i.e. exactly the heteroscedastic Gaussian family — the same independence
completion that `eqn_dist` relies on.

Well-matched to the flow: affine coupling *is* a location-scale transform
conditioned on the other coordinates, so `X = μ(T) + exp(s(T))·u` is native — the
opposite of the step-function binary case.

**Inherits the `eqn_dist` failure mode.** Pooled `var(z) = 1` does *not* force
conditional homoscedasticity: a capacity-limited flow can trade variance across
`T` (too wide early, too narrow late) while satisfying the pooled moment, exactly
the mechanism behind the derived-tail collapse. Mitigation: a few coarse binned
checks `E[z² | T ∈ bin] = 1` as diagnostics, promoted to constraints if they fail.

Also watch: `T`'s own marginal is a modelling artifact, not a belief — fix it
uniform and interpret only `T`-conditional quantities, since every unconditional
constraint silently averages over it. And `X`'s box must accommodate the widest
`t`, or the late-time band is truncated.

## 9. Dyadic increment variables (`delta_1yr`, `delta_2yr`, `delta_4yr`, ...) — *proposed*

Sites are **increments at a lag** rather than levels at a time, exploiting time
equivariance (= stationary increments). This is the random-midpoint-displacement
/ wavelet construction used to synthesise fBm, not a hack.

**The reparameterisation is the main win, independent of the dyadic spacing.**
In increment coordinates the maxent *default* — independent sites — **is** the
random walk. In level coordinates you have to constrain your way to independent
increments and hope a capacity-limited flow finds it (the exact mechanism behind
the derived-tail collapse). General principle: *choose coordinates in which the
maxent default is your prior.*

**It restores path sampling at `O(log T)` dimensions.** One draw gives `X` at the
anchor offsets directly. Finer paths come from the conditional split, which the
joint over sites already supplies: for a random walk
`delta_1 | delta_2 = d  ~  N(d/2, α²/2)` — literally midpoint displacement.
Recursing gives arbitrary resolution in post, with no extra sites. Non-self-similar
beliefs are fine too, using the scale-specific split from each consecutive dyadic
pair (`delta_1|delta_2`, `delta_2|delta_4`, ...).

**The deltas are not independent free variables.** `delta_k - delta_j` is itself a
lag-`(k-j)` increment, so equivariance forces `Var(delta_k - delta_j) =
Var(delta_{k-j})`, i.e. by polarisation

    Cov(delta_j, delta_k) = [Var(delta_j) + Var(delta_k) - Var(delta_{k-j})] / 2

(random walk check: `j=1, k=2` gives `Cov = α²` ✓). These are covariance targets
among the sites — the same Gram-matrix moment machinery as #6. Non-dyadic gaps
(`delta_4 - delta_1` needs `γ(3)`) come from interpolating the elicited variogram.

**Cheap feasibility check.** `sd(delta_k)` must be **subadditive** —
`sd(delta_{j+k}) <= sd(delta_j) + sd(delta_k)`, just the triangle inequality in
L². Catches nonsense like `sd(1yr)=0.1, sd(2yr)=0.5` instantly. The exact
criterion is conditional negative definiteness of the variogram = PSD of the
implied covariance, the same eigenvalue pre-flight as #6/#8.

**Where equivariance breaks.** Two regimes worth separating: increments from a
*fixed* anchor ("now") need no stationarity at all and are just the site-per-time
model reparameterised; reusing increments at *other* anchors needs stationarity
and is what buys path synthesis. Equivariance fails outright when calendar
structure matters (a scheduled election, a policy deadline, a known event date).

## 10. Scoped / confidence-weighted equations — *proposed*

Two forms of the same idea — restrict where a structural equation is trusted.
**Prefer (B).**

### (A) Conditional equations — `lhs = rhs` given `cond`

Tractable: every equation loss becomes the `cond_expect` ratio it already has a
pattern for (`E[r·c]/E[c] = 0`, `E[r²·c]/E[c] = σ²`, `E[relu(-r)²·c]/E[c] -> 0`).

Genuine use: **scoped beliefs** — "I only know this relation holds in normal
conditions, outside I have no idea." Note a regime-dependent *RHS* is already
expressible via indicator gates (`y = a*x*ind(x>c) + b*x*ind(x<c)`); what is new
is *declining to constrain* outside the region.

Two warnings:

- **Require noise; forbid the deterministic conditional.** An exact identity
  holding on a positive-probability region asks for a mixture of a singular and
  an absolutely-continuous component — a *discontinuous change in effective
  dimension*, which a diffeomorphism cannot represent. Strictly worse than an
  unconditional identity (already the #2 pathology, but at least uniform). In
  practice it degrades rather than crashes: the band never gets as thin as asked,
  worse the larger `P(cond)`.
- **The vacuity escape.** A conditional constraint is trivially satisfied by
  making the condition rare — the optimiser can shrink `P(cond)` instead of
  fixing the residual. Exactly the mechanism that made the conditional
  `P(record | anomaly > 1.6)` vacuous once the tail collapsed. **Always pair a
  conditional equation with a constraint on `P(cond)`.** Watch
  `conditioning_report` (ESS = `N·E[c]`).

Related precedent already in the codebase: the `onoff` kind gates a belief on a
*learnable* Bernoulli credence ("this constraint may be wrong"). A conditional
equation is the same weighted-constraint shape with the gate a fixed function of
`x` instead of a free parameter.

### (B) Metric-scaled noise — `Var(eps) = f(dist(x, x_base))`

Needs **nothing beyond #7** (expression-valued sd). Soft scoping: the equation is
tight near the anchor and loosens smoothly away from it. Better behaved than (A)
— always active, so no vacuity escape and no discontinuous dimension change.

**Defaults from precedent:**

- **Relative / proportional error**, `σ = κ·|rhs|` — Fermi estimation, log-normal
  error models. The single most useful default; `κ` reads directly as "accurate
  to ±κ".
- **Saturating (kriging / GP)**, `σ²(d) = σ₀² + (σ_∞² - σ₀²)(1 - exp(-(d/ℓ)²))`.
  GP predictive variance returns to the *prior* variance away from data — so set
  **`σ_∞` = the prior spread** (e.g. `span/4`). The equation then degrades to
  "no information" far from the anchor, never to negative information.
- **Taylor remainder**, `σ ∝ d²` when the RHS is a first-order linearisation
  (`σ ∝ d` for a zeroth-order one). Principled when the equation literally *is*
  an expansion about `x_base`.
- **Parameterise as `σ = exp(a + b·d)`** for automatic positivity — the standard
  heteroscedastic-regression form, and natively what an affine coupling layer
  computes.

**Never let `σ` grow unbounded.** Far from the anchor the constraint becomes
vacuous, and unconstrained volume is free entropy the optimiser will happily fill
— maxent pushes mass exactly where the model claims ignorance. Saturation caps
the damage; tying `σ_∞` to the prior spread makes it semantically correct.

Inherits #8's caveat: a *pooled* `var(z)=1` does not force the error profile to
be right in each region. Spot-check `E[z² | d ∈ bin]` across a few distance bins.

## 11. Synthetic-likelihood (NLL) constraint kinds — *shipped (model layer)*

Motivation: the squared kinds silently *average* conflicting estimates —
`E[x]=10` and `E[x]=45` fit tight at 27.5, with width identical to the agreeing
case (verified: sd 23.25 vs 23.25 in the demo). We want disagreement to become
predictive width, and agreement to tighten, including *indirect* conflicts
(direct vs conditional vs threshold-probability estimates) that pre-pooling at
the target level cannot see.

Mechanism (Bayesian synthetic likelihood, Wood 2010): score each target as the
mean of `k` draws from the fitted distribution, `t ~ N(E[f], Var[f]/k + τ²)`,
and minimise the NLL **with gradient through the model's own `Var[f]`** — the
`(m−t)²/s²` term makes unreachable targets cheapest to explain by inflating
variance; the `½log s²` term makes agreeing targets pay for width. `k` is the
unitless strength ("effective observations"), replacing unit-bearing sd widths
and the span-scaling heuristics. Probability targets use exact binomial
pseudo-counts `k·KL(t ‖ p)` — no CLT, no variance denominator, conflicts
resolve toward max-variance `p`.

Stability (the naive heteroscedastic NLL death-spiral is real):

- **β-NLL** (Seitzer et al. 2022): scale each NLL by `sg(s²)^β`, `β=0.5`
  default. `β=1` restores a constant-weight mean gradient; every β keeps the
  width-seeking gradient path through `Var`.
- **τ floor** on the implied noise sd: bounds the `1/s²` gradient and caps the
  `√k` conflict amplification (a disagreement of Δ implies population sd
  `~√k·Δ/2` under the mean-of-k story — τ and moderate k keep this sane).
- **Debias**: subtract `Var/N` from the squared batch residual.
- **Conditionals**: `k_eff = k·E[cond]` with `E[cond]` stop-gradiented (no
  starving the condition to mute the constraint) — but the marginal of the
  conditioning variable can still *drift* to relieve conflict via correlation,
  so the #10 rule applies here too: **pair conditional NLL constraints with a
  constraint on `P(cond)`**.

New kinds in `SamplerModel._prepare` (inherited by `FlowSamplerModel`):
`expect_nll`, `cond_expect_nll` (`f, [cond,] target, k[, tau[, beta]]`),
`prob_nll`, `cond_prob_nll` (`f, [cond,] target, k`).

Demo `examples/nll_conflict_demo.py` (x, y on [0,100], K=8, τ=3, β=0.5,
`entropy_reg=1`): sd(agree, 3 mixed-type estimates) 3.7 < sd(single) 5.3 ≪
sd(mixed conflict) 20.0 < sd(direct conflict E[x]=10 vs 45) 40.7 (bimodal;
uniform ceiling 28.9), squared baseline 23.25 either way. Note the NLL kinds
shrink harder than the squared kinds under agreement (`½log s²` pressure):
single-constraint sd ≈ `√(entropy_reg·k·τ)`-ish, so k and τ set the tightness
scale — raise τ / lower k if a lone mean estimate should not imply sd ~5 on a
span of 100.

Not yet wired into `DistributionBuilder` / `sample_losses.py` — open questions:
where `k` comes from (per-estimate field vs credence-derived), default τ as a
fraction of span, and whether the NLL kinds replace or sit beside the
squared/logit kinds behind a builder flag for benchmark A/B.

## 12. Mixture base distribution for the flow — *shipped (model layer)*

Multimodal maxent solutions (e.g. the conflict2 bimodal posterior from §11)
force an invertible map to tear a unimodal `N(0, I)` base apart — extreme
Jacobians, ugly optimization. `FlowSamplerModel(..., n_components=K)` swaps
the base for a uniform-weight Gaussian mixture with learnable per-component
means and diagonal scales (params leaf `"base"`; `base_spread` sets the init
mean sd). Modes then come from component placement, not map contortion.

Everything stays exact: `H(x) = −E[log q_mix(z)] + E[log|det J|] + Σ log
span` with the mixture density in closed form; component choice is uniform
(no weight params), so the reparameterized entropy gradient is unbiased.
`log_prob` and `entropy()` updated to match; `n_components=1` is bit-identical
to the old standard-base path. Caveat: with `flow_type="spline"`, component
means past `tail_bound` land in the identity tails — keep `base_spread` well
inside `B`. Learned mixture *weights* deferred (categorical reparam needed);
uniform weights lose nothing when K modestly exceeds the true mode count.

Demo: `examples/mixture_base_demo.py` — {agree, conflict2, interior2} ×
{K=1, K=4}, all checks pass. Findings worth keeping: (a) the §11 conflict2
optimum is **edge**-bimodal (sd 41 > uniform ceiling 28.9 → mass at 0/100,
mean-matched 73/27), and edge modes are cheap for a plain flow — the sigmoid
squash saturates — so K=1 and K=4 tie there; (b) on the interior-bimodal
stress (`E[sin(3πx/100)] ≈ 0.92`, peaks 16.7/83.3, valley 50) both bases
solve it too at 2 dims / 6 layers (K=4 loss −8.10 vs K=1 −8.06, both with a
~20-nat valley dip). Verdict: correct, exact, and free at small scale, but a
2-d toy does not expose the capacity gap — A/B it where the pathology
actually bites (higher-dim, more modes, or the spline capacity sweep) before
concluding it's needed.

## 13. Scaling to ~1000 dims: constraint-graph factorization — *proposed*

Target scale is ~1000 sites with most pairs uncorrelated. Key fact: for a
maxent objective with a factorized reference (uniform box or per-site
Gaussian prior), variable groups sharing **no constraint** have an exactly
product-form optimum — coupling them costs entropy and helps nothing. So:
build the constraint hypergraph (each constraint touches few sites),
union-find its connected components, fit one small model per component.
Entropy is the sum of per-component entropies (no 1000-dim log-det, no MC
variance growing with D); each flow stays in the few-dim regime where it
works; fitting is embarrassingly parallel and incremental (new constraints
refit only their component). Singleton components can use a 1-D spline or
closed form. Compose with §12: mixture-base only the components carrying
conflicts.

Risks / fallbacks: equation chains can merge everything into one giant
component — within a big component, the TN engine's chain/tree exploitation
applies (this gives the TN-vs-flow crossover a concrete criterion: component
size/treewidth, not total problem size), or use a coupling flow whose masks
follow the graph sparsity. For a large dense component, a CNF with
Hutchinson trace is the safe fallback (unbiased entropy, unrestricted vector
field) — O(1) backprops per sample. Ruled out: diffusion + ELBO — the ELBO
lower-bounds `log q`, hence *upper*-bounds entropy, so the maxent optimizer
can fake width by inflating the bound gap; this silently breaks the §11
conflict→width mechanism. kNN/non-parametric entropy also dies above ~10–20
effective dims (fine per-block, pointless there since flow entropy is exact).

## Known sharp edges (not additions — things to fix or document)

- `ind(x > 0.5)` is span-normalised, but `ind(x > y)` silently falls back to raw
  units (no single natural scale for var-vs-var), so at the default sharpness 80
  it is near-vertical for wide-span variables and nearly flat for narrow ones.
- Constraint stiffness scales with the **LHS variable's** span — well-calibrated
  for `z = a*x + b*y`, potentially far off when the RHS is strongly nonlinear.
  Set `eqn_rel_sd` / `noise_sd` explicitly there.
- No domain guards: `y/x` where `x`'s box straddles zero, or `x**0.5` where `x`
  can go negative, blows up or NaNs. Variable bounds are the only protection.
- **Soft-indicator leakage floor vs near-zero targets** (found via
  `benchmarks/prereq_experiment.py`): a (near-)impossibility constraint like
  `P(e ∧ ¬a) ≈ 0.001` scored with sigmoid indicators has a *floor* — even the
  exact maxent joint registers soft violation ~0.01 from sigmoid-width mass at
  the thresholds. If the target sits below that floor, a strong-k constraint
  penalizes the true solution ~k·(floor − target) per implication, which can
  exceed the entropy reward for the structure the constraint is *about* — the
  objective then genuinely prefers the vacuous (event-suppressed) joint; no
  optimizer or architecture fixes that. Direct-scoring check: the prereq-event
  problem preferred vacuous at every m until fixed. Fix: **sharpen the
  violation feature's indicators until the floor drops below the target**
  (floor ~ density·ln2/sharpness per side; sharpness 800 → floor ~0.0007 for
  a 1e-3 target; direct scoring then prefers the true joint at every m).
  Threshold *margins* (`e > t+δ`, `a < t−δ`) are the tempting WRONG fix: they
  open a dead band the loss cannot see but hard-threshold semantics still
  count, and entropy fills it (measured: P(e|¬C) jumped to ~0.1). Builder
  rule of thumb: any elicited `P(...) < ε` needs indicator sharpness ≳
  (density·ln2·factor-means)/ε on that feature, not a smaller ε. Better
  still: **straight-through indicators** (forward hard, backward soft — see
  `st_ind` in `benchmarks/prereq_experiment.py`) make the forward violation
  exact at any backward sharpness, and in feature *products* each factor's
  gradient is gated by the other's hard value (corner samples feel zero
  implication pressure). Neither fixes the init-slam dynamics — a strong-k
  impossibility crushes the event globally in the first ~100 steps and the
  vacuous basin is an attractor (longer training makes it worse) — so pair
  with a weak-k warmup phase (`warmup_steps` there). Open observation at
  m=6: with implications enforced exactly, the failure mode flips to corner
  *inflation* — over-correlating the prerequisites enlarges P(all met)
  beyond maxent (corr 0.28 vs true 0.04) because marginal constraints don't
  price correlation; same family as the §10/§11 correlation escape.
