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

### 11b. `eqn_nll` — k-sample equation beliefs — *shipped (model layer)*

Extension of the same machinery to equation-language constraints. The belief
"an informant saw `lhs = rhs` hold on k data points with residual scatter
≤ σ" is the sufficient statistics of k Gaussian residual draws (mean 0,
variance σ²), whose NLL under the model's actual residual moments is a single
closed form:

    ("eqn_nll", r, sigma, k[, beta])   →   k · KL( N(0, σ²) ‖ N(μ_r, v_r) )

**One-sided in variance**: believing an equation means "residual *at most*
σ", so `v_eff = max(v_r, σ²)` — a tighter-than-claimed residual is free, and
σ² doubles as the τ floor (bounded 1/v gradients). Below the floor the KL
reduces to the plain mean term `μ_r²/(2σ²)`; above it, `k·log(v_eff/σ²)`
prices loosening the equation. β-scale is `sg(v_eff/k)^β` — the k-sample
*mean's* variance, matching `expect_nll`'s `s²` so the kinds price their
escapes on the same footing (scaling by raw `v_r` overprices loosening by
√k — found the hard way). Same debias (`μ̂² − v_r/N`). `eqn_det` (σ=0) stays
as-is: an exact identity has no finite-k story. `eqn_ineq` doesn't fit the
frame either (support restriction, not a statistic).

Demo results (σ=5, K=8, same file): **which belief yields under conflict is
decided by k and by entropy, not hardcoded.** `eqn_conflict` (equation +
`E[x]=10`, `E[y]=45`, all k=8): the fit *keeps* the equation (sd(r)=5.5,
corr .99) and pools the level with sd(x)→38.9 — maxent-correct, since
widening marginals buys entropy while loosening the equation buys nothing,
and the bounded domain lets Var(x) inflate almost freely (raising level-k
mostly doesn't flip this: the model inflates Var ∝ k to keep `Var/k`
constant until the domain-variance ceiling binds). The clean test of the
loosening channel is `eqn_vs_eqn` (`y = x + N(0,5)` vs `y = x + 20 +
N(0,5)`): marginal widening can't reconcile a clash that lives in the
residual itself, so both equations loosen — E[r]=10.5 (pooled midpoint),
sd(r)=11.5 = 2.3σ, corr .89 (structure retained, weakened). `eqn_agree`
(equation + consistent levels) lands both means on target with sd(r)≈σ. All
10 demo checks pass.

Not captured (deliberately): residual–rhs *independence* — the k-sample story
constrains the residual's marginal moments only; independence still comes
(weakly) from maxent, as in `eqn_dist`. If the correlation escape shows up in
practice, a `corr`-style NLL term on `Corr(r, rhs)` composes cleanly.

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

## 14. Estimator + annealing consolidation — *shipped to the builder*

Outcome of the prereq-benchmark campaign (see the sharp-edges bullets for
the full diagnosis chain), now wired as `DistributionBuilder` defaults:

- **`prob_penalty="nll"` (new default)**: equality probability targets emit
  `prob_nll` / `cond_prob_nll` (binomial pseudo-counts, `k·KL(t‖p̂)`) with
  `k = 1/(sd_logit²·t(1−t))` Fisher-matched to the log-odds belief width
  (t=0.5, sd=0.3 → k≈53; capped at 1e4 for tail targets).  Inequality
  relations and `robust` gates fall back to the logit machinery unchanged.
- **`st_indicators=True` (default)**: loss-side proposition indicators are
  straight-through (`straight_through(soft, hard)` now in `model.py`):
  forward exact at any backward sharpness — kills the leakage floor;
  conjunctions inherit hard gating.
- **`fit(anneal_phases=8)` (default)**: likelihood-tempering staircase β:
  1→0 via `linspace` over equal step blocks inside ONE optimizer run
  (`beta_schedule` on `constraint_loss`, step-aware loss, single jit
  compile, no Adam resets).  No-op when no NLL-family constraints.
- Smoke-tested end-to-end: a 3-var prereq scenario through the production
  builder gives P(e|C)=0.39 at 800 steps with violations at 2e-4 —
  the configuration that used to collapse to ~0.02.

Sweep evidence (estimator_ab.jsonl): β-staircase b4/b8 beats hand-tuned
k-warmup at m=4 (0.41±0.02 vs 0.27±0.04, with a striking variance
collapse); m=6 needs budget, not scheme (b8@3000: 0.256 → b8@6000: 0.429,
seed0 = 0.507 on target).  **Holds are load-bearing**: per-step linear
ramp *underperformed* the b8 staircase at m=6 (0.07–0.12 vs 0.256) and
b16/b32 regressed — proper continuation equilibrates at each homotopy
point.  Rule of thumb: holds ≥ ~500–750 steps, total budget scales with
the stiffest constraint.

**Explicitly NOT closed** (deliberately parked to move on):

- **Correlation valley**: the remaining m=6 seed spread (0.32–0.51) is
  drift along a flat, unpriced direction — prerequisite over-correlation
  inflating P(C) (ratios 1.07–1.26, corr 0.08–0.12 vs true 0.04).  Candidate
  fix: weak isotropic pairwise `corr → 0` penalties (a factorization prior;
  NOT corr_true — that's the analytic answer).  Untested.
- Builder staircase = holds *without* optimizer resets — strictly untested
  vs the benchmark's phase-refit variant (expected same-or-better).
- Expectation / equation kinds not annealed (only the prob_nll family has
  `takes_beta`); `expect_nll`'s baked β=0.5 is a *permanent* β-NLL setting —
  if annealed later, ramp 1→baked, not 1→0.
- Per-step ramp needs a minimum-hold guard if revisited (br0.85's corr
  inflation = hold squeezed to 450 steps).
- Variance/ESS ideas parked: Polyak-EMA eval weights (cheap, do first),
  loss-statistic EMA for rare-event p̂ (stateful loss refactor),
  `b2=0.9999`.
- Mixture base (K>1) × β-anneal combination never tested — all K>1 rows
  predate the anneal machinery.

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
- **Hybrid pathwise+score estimator for hard indicators** — *shipped as an
  option*. Two channels can move probability across a sharp boundary:
  transport (move samples; needs a soft/ST surrogate, zero gradient for hard
  `f`) and reweighting (change `log q` on either side; the score-function
  identity `∇E[f] = E[f·∇log q]`, needs no smoothness). Until now every
  constraint gradient was pure transport; `log q` only appeared in the
  entropy term. The unified estimator uses the soft surrogate as a control
  variate inside the score estimator:
  `∇E[h] = ∇E[s] + E[(h−s−b̄)·∇log q]` — unbiased for the HARD expectation,
  with REINFORCE variance confined to the ~1/sharpness boundary band where
  `h≠s` (this is what makes it usable on 1e-3-rare events; REBAR/RELAX is
  the discrete-variable analogue). Equivalent view: the true gradient of a
  discontinuous `E[f]` = interior pathwise term + boundary flux (Reynolds);
  a sharp sigmoid's pathwise gradient is a kernel estimate of the flux with
  bandwidth 1/sharpness — so "sharpness" was always a bias/variance
  bandwidth knob, and the score residual restores what the kernel misses.
  Plumbing: `FlowSamplerModel.constraint_loss(..., with_logq=True)` appends
  a per-sample `log q_θ(stop_grad(x))` column to the feature matrix (one
  extra inverse pass; the forward-path `log N(z) − logdet` is WRONG here —
  its θ-gradient includes the transport term). `hybrid_ind_prod` +
  `viol_mode="hybrid[λ]"` in `benchmarks/prereq_experiment.py`; forward
  value bit-identical to ST, `score_scale=0` reproduces ST exactly (ST-gated
  pathwise part).
  **Verdict after the GPU estimator A/B + λ-sweep: UNSTABLE at every tested
  scale (λ ∈ {0.03, 0.1, 0.3, 1.0}) — retired in favor of ST+warmup.** Full
  strength violated even the k=32 marginals (marg_err up to 0.7, NaNs); the
  λ-sweep showed the instability is structural, not gain: failures are
  non-monotone in λ because the per-sample score weight scales with |log q|,
  which is unbounded and heavy-tailed under a sharpening flow — no fixed λ
  bounds a single sample carrying hundreds of nats. Tantalizingly, the two
  runs that stayed stable produced the BEST numbers ever seen on the prereq
  benchmark (P(e|C) = 0.502, 0.478 at m=4), confirming the Fisher–Rao
  reweighting theory when the loop doesn't blow up. If revisited: clip or
  rank-normalize the per-sample score contribution, or prefer the two
  INHERENTLY bounded reweighting channels — learnable mixture weights
  (birth-death over K components; score is a softmax over K logits, nothing
  scales with a deep flow's log q) and Bernoulli/categorical heads (exact
  Rao-Blackwellized reweighting for discrete events, no estimator at all).
  Meanwhile the A/B also showed sharp-soft (800) is DOMINATED: it fixed the
  objective but killed the gradient (support band ~1/800), collapsing at
  m=6 even with warmup — ST's exact-forward/wide-backward decoupling is
  what wins. ST *without* warmup at m=6 inflates the corner on every seed
  (corr up to 0.64, P(C) 2–3× maxent): warmup also damps the correlation
  escape, and `max_corr` is now a headline metric.
- **β-annealing (likelihood tempering) as the general warmup — shipped for
  testing.** The k-warmup question ("how to pick the multi-term schedule in
  general?") has a one-parameter answer inside the k-sample NLL family:
  every constraint is a likelihood, so anneal the *temper* rather than
  per-term weights. `prob_nll` gained an optional `beta` (scaled by
  `sg(p(1−p)/k)^β`, the k-sample binomial mean's variance — same footing as
  `expect_nll`/`eqn_nll`'s β): β=0 is the plain k·KL, β=1 cancels k
  entirely. Measured at the prereq init: constraint loss mass 146.7 nats at
  β=0 vs 0.145 at β=1 (entropy scale ~1–3) — no slam is *possible* at β=1;
  and per the loss-scale table (scratchpad `loss_scale_check.py`), β=0.5
  under-enforces at convergence (equilibrium violation ~1e-2 vs target
  1e-3), which is precisely why the anneal must END at β=0. Annealing β
  1→0 is power-posterior / homotopy continuation: each constraint's
  strength ramps over a range set by its own k and variance — stiff
  impossibilities get a long ramp, mild marginals barely move, nothing is
  hand-tuned per term. `fit_and_measure(beta_phases=n)` staircases
  `linspace(1, 0, n)` over warm-started phases within the same step budget
  (mutually exclusive with `warmup_steps`); `beta_anneal_configs()` +
  notebook `MODE="beta"` race it against the hand-tuned `st wu1000`
  baseline. If it matches or beats k-warmup, the builder-facing version is
  a single β schedule in `DistributionBuilder.build` and the per-constraint
  warmup question disappears.

## 15. Peer-reliability benchmark — big-setting test of the CURRENT api

`benchmarks/reliability_experiment.py` + `colab_reliability_sweep.ipynb`
(thin git-pull shell, same pattern as the prereq sweep). 25 people,
`r_i ~ Beta(8, 2)` (strongly weighted reliable); each rates 6 random others
(connected 6-out digraph, resampled if not) with noise
`sigma(r_rater) = 0.40 - 0.35*r_rater` — the self-referential setting
eigenvector methods (EigenTrust) roughly solve. Ground truth by
construction, so no NUTS reference needed for v1: score (a) RMSE of the
posterior mean vs the iterative eigenvector-style baseline and the
prior-shrunk mean, (b) 80%-interval coverage across replications (the
calibration claim point-estimate baselines cannot make). Deliberately ZERO
solver/api changes — misspecification tolerance is part of what is measured:

- `mode="fixed"`: every eval an `ExpectationEstimate` at one global
  sd = 0.12 (rater identity discarded — misspecified). Structural ceiling
  is the shrunk mean; measures degradation under ~150 mutually-inconsistent
  fixed-strength constraints.
- `mode="eqn"`: heteroscedastic model expressed in the EXISTING equation
  language via a variance-stabilised residual
  `r_j = r_j - (r_j - s_ij)*SIG0/sigma(r_i) ~ N(0, SIG0)` -> eqn_dist.
  When the fit thinks rater i is unreliable the residual shrinks — the
  likelihood's reweighting, via moment matching. This arm competes for the
  mean->eig headroom (+0.003 rmse over 40 seeds — small because the shrunk
  mean is already strong at 6 evals; sparsity chosen exactly so intervals
  matter).
- Baseline gotcha (fixed): raw inverse-MSE rater weights OVERFIT at 6 evals
  per rater and lose to the plain shrunk mean; the shipped baseline shrinks
  each rater's noise estimate with 4 pseudo-evals at the average sd.
- Both modes + the eig baseline get the same prior info (`E[r_i]=0.8`,
  sd 0.121) so the comparison is fair.
- Open/v2: NUTS reference posterior for per-dataset shape debugging when a
  replication fails coverage; note eqn_dist matches residual mean/var of
  the joint sample — a maxent relaxation of the true per-observation
  likelihood, and eqn kinds are not annealed (section 14), both of which
  this benchmark now exercises at scale.

## 16. value_penalty="nll" — expectations as noisy observations (new default)

Motivated directly by the reliability benchmark's fixed-mode error-bar plot:
posterior means matched the eig baseline but 80% widths were ~0.45 (vs the
~0.13 Bayes-ideal for 6 evals at sd 0.12), because plain `expect` pins ONLY
the mean — repeated estimates of the same quantity re-argue the mean and
add zero precision, and maxent correctly inflates the marginal to the
widest shape with that mean (the exponential-tilt bars hugging the domain
edge). Precision accumulation is not expressible as moment pins at all.

Fix: `ExpectationEstimate` / `ConditionalExpectationEstimate` (equality,
non-gated) now route to `expect_nll` / `cond_expect_nll` with **k=1,
tau=sd, beta=0** — "this estimate is one noisy observation of the
quantity": target ~ N(E[f], Var[f] + sd^2), the exact marginal likelihood
of a noisy observation. Properties (all verified, scratchpad
`value_nll_smoke.py`):
- 1 estimate: the half-log Occam term saturates at sd^2, entropy wins, the
  marginal stays maxent-wide (measured sd 0.204) — a lone belief does not
  fake precision.
- n agreeing estimates: Var -> sd^2/(n-1), i.e. sd/sqrt(n) Bayesian
  concentration. Measured: n=4 -> 0.058 (Bayes 0.050), n=16 -> 0.026
  (0.025). Startlingly exact.
- tau=sd floors the implied noise -> no variance death spiral, so beta=0
  (full strength) is safe; curvature bounded by 1/sd^2, so no anneal
  needed either (and the builder staircase does not touch expect_nll —
  no takes_beta marker — which is fine at these bounded strengths).
- `value_penalty="square"` restores the legacy weighted squared residual;
  inequalities and robust gates use the legacy machinery in both settings.
Also fixed in passing: `_prob_sd` now honours per-estimate sd under
prob_penalty="nll" (previously logit-only — repeat-collapse widths were
silently ignored by the nll default).

Default-kind map after this: probabilities prob_nll/cond_prob_nll,
expectations expect_nll/cond_expect_nll; correlations (corr weight) and
equations (eqn_det/eqn_dist; eqn_nll exists unrouted) are still NOT nll —
the un-annealed, un-nll eqn path is the reliability benchmark's prime
suspect for the eqn arm's rmse gap + seed spread.

Immediate effect on the reliability smoke (pop=8): fixed-arm width80
0.456 -> 0.225, rmse 0.039 -> 0.035 (now beating the eig baseline at that
scale). The fixed arm is no longer a pure degradation probe — with
accumulation it is a genuine homoscedastic-Bayes competitor; re-run the
Colab sweep to see both arms under the new default.

### 15b. Flip variant — the hard version (decoy mode, quantified)

`flip` knob in the reliability benchmark: an eval is inverted
(`s = 1 - r_j + noise`) with probability `flip * (1 - r_i)^2` — typical
raters flip ~4%, duds ~half. The eqn arm moment-matches the true mixture
(mean `r_j + p(1-2r_j)`, sd `(sigma^2 + p(1-p)(1-2r_j)^2)^0.5` — the
grammar's `**0.5` makes this expressible; validated: residuals at true r
are mean ~0, sd ~SIG0). The fixed arm stays flip-blind; the eig baseline
can down-weight but never invert a rater.

Why it is hard (measured, `flip_residual_check.py`): the mirrored
configuration (all reliabilities inverted) is a DECOY BASIN. At flip=0 its
moment penalty is ~700x the truth's; at flip=1 only ~40x, because calling
everyone a flipper inflates every observation's variance normalizer and
shrinks all stabilized residuals below target — only the (weak) variance
match resists. A tiny 250-step CPU fit fell exactly into the mirror
(rmse 0.67 = inverted estimates, cover80 = 0). The 25 weak priors
(E[r]=0.8) are the intended symmetry-breaker. This makes the flip sweep a
direct test of basin selection (anneal / mixture base) against a
quantified decoy — notebook `MODE="flip"`.

## 17. eqn_penalty="nll" + annealed equations (new default)

Lever #2 from the reliability run (eqn arm calibrated on average but
bimodal across seeds — basin selection, same shape as pre-anneal prereq):

- `eqn_nll` now takes the traced `beta_t` override (`takes_beta`), so
  `fit()`'s staircase anneals equation stiffness on the same power-posterior
  footing as the prob kinds; the anneal trigger includes `eqn_nll`.
- Noisy `EquationEstimate`s route to `eqn_nll` by default
  (`eqn_penalty="nll"`; `"dist"` = legacy moment match). Emitted with baked
  beta=0 — the `v_eff >= sigma^2` floor bounds curvature so full strength is
  safe at convergence, and the schedule supplies the warmup.
- `eqn_conf` is reinterpreted under nll as k, the effective observation
  count behind the belief (informant saw the identity hold on k points).
  Default 10 suits elicited structural identities; the reliability
  benchmark sets eqn_conf=1 (each equation IS one measurement).
- eqn_nll's one-sided variance also structurally weakens the mirror decoy:
  residuals tighter than sigma are free but buy nothing, so the
  "declare everyone a flipper to shrink residuals" escape earns no reward
  (it only ever paid through eqn_dist's two-sided variance match).

Immediate smoke signal (250-step CPU, pop=8, NOT dispositive): the flip=1
eqn fit that previously locked into the mirror basin (rmse 0.67,
cover80=0) now lands at rmse 0.060 — beating the mean baseline — with the
anneal walking it past the decoy. The plain eqn smoke is width-inflated at
this budget (holds eat most of 250 steps); judge on the 3000-step GPU
sweep: pass = eqn <= eig rmse on 5/5 seeds at cover80 ~ 0.8.

## 18. Weak-estimates fusion benchmark (KL vs m, ~1/m target)

`benchmarks/weak_estimates_experiment.py` + `benchmarks/colab_weak_estimates_sweep.ipynb`.
The world is a fixed bivariate normal (X, Y) (mu=(0.5,-0.5), sd=(1.0,1.5),
rho=0.6); the solver only sees weak empirical estimates, each computed from
N_PER_EST=5 iid draws: empirical E[X], E[X*Y], E[X^2], P(X>2), P(X>Y), ...
sampled with replacement from a 20-functional menu. Oracle weighting: every
estimate carries its TRUE sampling sd at n=5 (value units for expectations;
log-odds 1/sqrt(n p(1-p)) for probabilities, Jeffreys-smoothed p-hat), so a
Bayes-optimal fuser's KL falls ~1/m. Claim under test: KL(fit||true)
decreases (near-)monotonically as estimates accumulate — expect_nll FUSES
weak evidence rather than averaging it.

Design points:
- Composite functionals ride through DETERMINISTIC EquationEstimate links
  (xy = x*y, xx = x*x, xpy, xmy, ...) to aux variables, then plain
  Expectation/Probability estimates on the aux var — doubles as a Fermi-link
  load test (`link_err` = RMS(aux - f(x,y))/sd(f) is a first-class metric).
- Aux domains by interval arithmetic over the (x,y) box so a link can never
  be forced outside its variable's domain.
- Per seed ONE pool of 64 estimates; the m-ladder (0,2,4,8,16,32,64) takes
  prefixes — each rung strictly ADDS evidence (monotonicity is about adding,
  not resampling). m=0 = links only = maxent no-information anchor.
- Scoring vs the closed-form truth, no reference posterior: kl_gauss
  (moment-matched Gaussian KL) + kl_knn (Kozachenko-Leonenko 1-NN entropy +
  exact cross-entropy — catches non-Gaussian pathology kl_gauss forgives);
  kl_uniform_ref() ~ 9.98 nats is the uniform-box anchor. summarize() prints
  per-seed ladder-step monotonicity fractions.

Smoke (250-step CPU, pool=16): kl_gauss 2.95 (m=0) -> 1.79 (m=4) -> 0.33
(m=16); link_err ~0.7 at this starved budget — watch it at 3000 GPU steps
before trusting the aux-variable channel.

## 19. Expression quantities: E[x * y], P(x - y > 0) through the front door

The DSL gap inventory (weak-estimates benchmark, correlation stall) showed
the solver grammar takes arbitrary callables everywhere while both DSLs
truncated quantities to bare variable names — the aux-variable + deterministic
link workaround was leaking (link_err ~0.7, corr pinned ~0.1 vs true 0.6,
and the residual ~0.25-0.30 nats of KL at m=64 was almost exactly the
missing-correlation cost).

Change (front-end only, no new constraint kinds):
- equations.py: `compile_expression(expr, ...)` — the residual compiler minus
  the lhs, exposing the existing grammar (+ - * / ** abs min max ind) for
  bare quantities.
- distribution_builder: `_moment_quantity` — an expectation subject that is
  not a variable name compiles as an expression; `_proposition_events` gains
  the same fallback for InequalityProposition (P(x*y > 0), P(x - y > 1)).
  Expressions have no declared domain, so: no target clipping; default sd,
  report scale, and indicator sharpness use the expression's range over a
  fixed 512-point uniform sweep of the variable box (`_expr_quantity`).
  Bad expressions raise -> skipped, as always.
- The STRING DSL gets this for free: `E[x * y] = 0.65` and
  `P(x - y > 0) = 0.8` already parse (main term passes through verbatim);
  only the builder used to reject them.
- eqn_ineq stays hinge-only deliberately: an inequality pins no moment, so
  there is no observation distribution for an nll to synthesize; note
  eqn_nll's variance channel is already one-sided (v_eff = max(v_r, s^2)).

Benchmark A/B: weak_estimates_experiment grows `encode="aux"|"expr"` —
IDENTICAL pools (same rng stream, values, sds), differing only in subject
form (E[xy] on a linked aux var vs E[x * y] on the 2-D joint directly).
`expr_configs()` + MODE="expr" in the sweep notebook. Old jsonl rows parse
as encode="aux".

Smoke signals (250-step CPU, NOT dispositive): 5 expression moment
estimates alone pull corr to +0.54 (true +0.60) where the aux-link route
managed ~+0.27; in the m=16 A/B the expr arm's corr_err flips from -0.32
(aux) to +0.24 and it runs ~40% faster (7 fewer dims + no link penalties).
Judge the KL ladder on the 3000-step GPU sweep.

## 20. String DSL: trailing `~ w` width term (+ Corr honours est.sd)

The last elicitation-facing gap from the constraint inventory: the string
grammar had no syntax for Estimate.sd, so the LLM could not state
confidence in an estimate except by repeating it (repeat-collapse).

- `E[Cost] = 100 ~ 20`, `P(A > 10) = 0.3 ~ 1.0`, `Corr(A, B) = 0.4 ~ 0.2`
  — an optional `~ w` tail on point AND one-sided P/E/Corr forms fills
  Estimate.sd. Units are the penalty's native residual space: value units
  for expectations (expect_nll tau), log-odds for probabilities (prob_nll
  Fisher-matched k), correlation units for Corr.
- One shared `_SD_TAIL` regex fragment feeds both PE_PATTERN and
  CORR_PATTERN, and EXPRESSION_PATTERN (the pydantic field the LLM is
  validated against) is composed from the same patterns — grammar and
  parser cannot drift.
- Equations keep their separate `~ N(0, sigma)` noise tail: different
  semantics (spread of a residual, not confidence in a stated number).
- Companion fix: _correlation_loss now honours est.sd
  (w = 1/(2 sd^2)) instead of silently using the global corr_sd — the
  same silent-ignore bug class as the _prob_sd nll fix in §16.
- NaturalEstimate's field description teaches the LLM the tail with
  calibration hints (log-odds ~1.5 = weak hunch, ~0.5 = confident) and
  says to omit it when unsure.

With §19's expression subjects, the string DSL now reaches everything the
builder front door supports: the playground notebook's extras cell is
pure strings — `est("E[x * y] = 0.65 ~ 0.37")`.

## 21. Effective sample size: Estimate.n + the `@ n` string tail

sd and n are ORTHOGONAL strength dials the nll machinery always had but the
DSL conflated: sd = noise per observation (tau), n = how many observations
the statement summarizes (the pseudo-count k). k couples the constraint to
the FITTED variance ("mean of 25 samples" informs Var[f], not just the
mean) — the part no single sd scalar can carry.

- `Estimate.n` (optional, >0) on the base class; string DSL trailing
  `@ n` after the optional `~ w` (`E[Cost] = 100 ~ 20 @ 25`); equations:
  `total = a + b ~ N(0, 5) @ 30`.
- Plumbing per kind:
  * expect_nll / cond_expect_nll: k=n.  n WITHOUT an explicit sd = "the
    mean of n clean samples": tau=0 and the kind-default beta=0.5 tempers
    the un-floored 1/var mean gradient (with tau=sd>0, beta=0 stays).
    sd-only keeps k=1, tau=sd (unchanged §16 semantics).
  * prob_nll / cond_prob_nll: a stated count IS the binomial pseudo-count —
    k=n takes precedence over the sd-derived Fisher match.
  * eqn_nll / eqn_dist: per-estimate n overrides the global eqn_conf
    ("this identity held on the 30 cases I saw").
  * corr: n parsed and stored, unused until a corr nll form exists.
- Self-weighting consequence: `E[x*y] = 0.9 @ 5` needs NO oracle sd — the
  Var[f]/n channel calibrates the estimate's strength from the fitted
  variance itself.  Smoke: five n=5 moment estimates with no sds anywhere
  fit stably (tau=0) and pull corr to +0.41 at 300 CPU steps.  This is the
  honest encoding for the weak-estimates benchmark (drop the oracle
  weights); a `weights="n"` pool variant is the natural follow-up A/B.

## 22. Post-mortem machinery: what the SpaceX-IPO forecast test exposed

Live calibration test (Aug 2026): beliefs elicited as of the Jan-2026
cutoff, fused, then resolved against reality (SpaceX listed whole-company
on June 12 at $1.75T — outside the valuation variable's [100, 1200]
domain). The fusion itself behaved (fused beat raw stated numbers on 3 of
4 scoreable rows); every serious failure was on the elicitation surface.
Candidate machinery, in rough priority order:

- **Boundary-mass diagnostic** (cheap, high value): report the fraction of
  samples within eps of each continuous variable's bounds; warn above a
  threshold. A hard domain bound is an infinite-confidence estimate the
  constraint report is completely blind to — truth at 1750 vs a 1200 cap
  produced NO diagnostic anywhere. Could live in `constraint_report()` as
  synthetic rows per variable.
- **Sensitivity readout** d P(query) / d target_i: the fit is
  differentiable end-to-end, so the influence of each estimate's target on
  a designated query node is one jacobian away. Would have flagged that
  P(ipo) hinged on P(xai_entanglement) and E[private_capacity] — the two
  numbers that were most wrong. Alternative cheap proxy: leave-one-out
  refits at starved budget.
- **Anchor-vs-mechanism tension**: with the topline anchor on, the fit
  matched it almost exactly (0.06); without it, the mechanism drifted to
  0.09 — which scored better. The machinery exists already (`~ w` / `@ n`
  weaken an anchor), but the *report* should surface "this estimate is
  fighting the rest of the joint by X nats" (per-constraint loss share at
  the optimum, signed by pull direction on a query node), so the user sees
  the disagreement instead of the anchor silently winning.
- **Soft-tailed domains**: an opt-in unbounded/log-scale variable type
  (soft barrier instead of hard box truncation) so bounds stop being
  secret certainty. Interim convention: set bounds at 2x the imaginable
  extreme and state a deliberate tail estimate near the edge.
- **Cumulative-event / hazard family**: `ipo_by_aug` and `ipo_by_eoy` are
  one event observed at two horizons; encoding them as separate binaries
  plus a hand-written implication conditional is ad hoc. A monotone
  P(event by t) family (single latent time variable, thresholds per
  horizon) would make timing beliefs coherent by construction.

Elicitation guidelines that came out of the same test (informant-side, no
code): treat cutoff-adjacent "in discussions" chatter as already in motion
(the xAI acquisition closed 3 weeks past cutoff and was THE causal
driver); never derive two forecasts (timing + structure) from the same
upstream belief without branching on it explicitly; when stated tail
probabilities contradict the stated mean, widen the mean's sd rather than
trimming the tail; base-rate process latencies (S-1 to pricing 3-6
months) don't transfer to priority projects of agentic founders.

## 23. Gas-price + measles calibration tests (Aug 2026): resolutions

Second and third live cutoff-vs-today tests (methodology per
`.claude/skills/solver-forecast/SKILL.md`; SpaceX post-mortem is §22).

**Gas (national average, Aug 12 2026).** RAW $3.05, 80% [2.70, 3.55];
fused E=3.12, P(>3.5)=0.15. Truth: **$4.036** (AAA) after the Feb 28
Hormuz closure — WTI peaked $112.84, gas peaked $4.48 in May. Outcome in
the fused ~2-5% tail; both raw and fused miss. But the designated shock
branch fired and its slice was nearly exact (E[wti|geopol]=85 stated,
83.3 realized); the residual miss was crack margin left uncoupled from
the shock (~$0.40). Also confirmed: `constraint_report` shows raw
`fitted - target` for one-sided rows, so satisfied hinges masquerade as
worst-fit rows (fix still open: store relation direction, report
hinge violation).

**Measles (CDC-confirmed US cases YTD, Aug 12 2026).** First test of
the latent-hub design (`intensity`, unitless graded severity, linear
lower-bound couplings + branch toplines). Truth: **~2,465** (Aug 6), a
35-year record; major_cluster True (SC ~1,000, Utah 514);
num_outbreaks 38 (stated 22~10); elim_lost False (PAHO review ~Nov).

| row | truth | RAW | v1 fused | v0 fused (pre-revision) |
|---|---|---|---|---|
| cases_ytd | ~2,465 | 1,500 | 2,075 (z .30) | 2,420 (z .03) |
| P(>1650) | T | .45 | .59 | .69 |
| P(major_cluster) | T | .50 | .73 | .93 |
| P(elim_lost) | F | .35 | .34 | — |
| P(>4000) | F | .12 | .09 | — |

Findings (n=2 and n=1 respectively — directions, not laws):

- **Fused beat RAW across the board** for the second resolved test where
  the mechanism disagreed with the gut anchor upward. The hub produced
  the correct comoving tail (cases + largest_cluster + num_outbreaks
  jointly high) without any pairwise corr being stated.
- **The revision round cost accuracy (n=1)**: v0's entropy-drifted
  E[intensity]=1.63 was closer to the realized ~record year than the
  1.0 anchor I re-imposed in "transcription" edits. The edits were
  methodologically defensible ex ante; the *belief* they restored was
  miscalibrated. Lesson kept deliberately weak: freeze and score every
  revision round (v0/v1/...) so laundering-vs-correction is decidable
  at resolution — NOT "don't revise".
- **Momentum bias (informant-side, now 2 hits)**: a knowledge-cutoff
  snapshot of a year-to-date count is a *lower bound with momentum*,
  not a total — the 2025 measles "total" known in Nov (~1,650) ended at
  ~2,289 with an outbreak raging. Same shape as the SpaceX
  "in-discussions" bias. Under-shot cases, outbreaks, and imports
  simultaneously.
- **Boundary-mass heuristic breaks on log-scale boxes**: 2% of a
  [100, 30000] span is 698, which overlaps genuine low-end mass —
  false-alarmed at 0.19/0.25 on a healthy fit. Needs a quantile- or
  log-space variant (ties into §22 soft-tailed domains).

## 24. Elicitation protocol v2: the methodology, mechanized (Aug 2026)

`generation/protocol_v2.py` + the `v2` entry in `metaculus/run_protocol.py`
port the manual calibration-test workflow (SKILL.md, §22–23) into the
LLM-elicitation pipeline. Fixed 6-node protocol, 4 LLM calls/question, no
solver in the loop:

1. **gen_structure** — mechanism variables + declared shock cases, outcome
   terms, dependency edges, optional latent hub (mandated [-1, 4] integer
   legend); catch-all `unlisted_shock` binary injected by code.
2. **gen_tier1** — flood of cheap one-sided truths using the full DSL:
   accounting inequalities, prerequisite ceilings, corr signs on edges,
   linear floors (honest-floor caveat in-prompt), tail beliefs near bounds.
3. **propose_matrix_requests** (pure code) — case-coverage-matrix cells:
   both legs per shock × outcome, all-shocks-false baseline row, hub anchor
   package (E + graded tails at fractional span positions) + hub regime rows.
4. **fill_matrix** — answers every cell or marks it `UNCHANGED` (recorded
   decision, never a silent default).
5. **gen_battery** — marginals, p10/p50/p90 spreads, deliberate bound-tails,
   and the direct target stated weakly (`~ 1.5`) so mechanism disagreement
   survives.
6. **validate** (pure code) — drops unknown-name estimates, hard-fails on
   missing direct target / open target-row cells / missing hub E-anchor /
   no coupling; warns on missing complements, spreads, degenerate medians.

Payload additions: `cases/outcomes/edges/hub/unchanged/validation` plus a
frozen `raw` record (direct target + first marginal per variable) for
fused-vs-RAW scoring at resolution. Known informant biases (YTD momentum,
"in discussions") are baked into the shared grammar prompt.

**No repeat collapse**: passes have disjoint scopes and matrix cells are
key-filtered, so duplicate quantities aren't generated; `quantity_key` now
carries the relation (a point belief and a bound on the same quantity never
merge or block each other) and handles equations. `collapse_repeats` remains
for v1 caches and is an identity on v2 output. If replicates return, the
plan is runner-stamped `@ n` budgets (total evidence per quantity capped
regardless of draw count), not a collapse pass.

---

## 25. Deep sigmoidal flow (`flow_type="dsf"`) — *shipped (model layer)*

Huang et al. 2018 sigmoidal transforms as a third flow engine:
`DistributionBuilder(..., flow_type="dsf", n_sigmoids=8)`. Each of the
`n_layers` layers is an affine coupling (dependence, as before) followed by an
elementwise K-component sigmoidal block
`u = sum_i w_i sigmoid(a_i y + b_i)` (`w` simplex, `a > 0` — monotone by
construction, universal approximator of 1-D CDFs), re-logited so blocks stack;
final sigmoid squash unchanged. Log-det stays exact (logsumexp forms
throughout), so the exact-entropy maxent machinery is untouched. Init is
near-identity: one dominant `sigmoid(y)` component, the rest spread over
`b ∈ [-1.5, 1.5]` at low weight to break component symmetry.

Motivation: affine couplings through the fixed sigmoid squash can only bend
logistic-shaped marginals per layer — asymmetric mass against a hinge or a
forbidden middle forces extreme scales. Measured (600 steps, CPU): bimodal
`P(x<-40)=P(x>40)=0.45` + thin middle — dsf worst_err_rel 0.004 vs affine
0.029 (affine splits lobes 0.48/0.41, dsf 0.449/0.448); skew+tail and
wall scenarios ≈ tie. On the *infeasible* y/x/k hinge repro dsf ≈ affine, as
expected — that pathology is infeasibility, not expressiveness. An apparent
dsf regression on `E[y] = 30` (span 200) is actually the soft value-nll/
entropy trade: default sd = 10 lets the mean drift for entropy, and dsf
drifts further because it converts mean-slack into entropy more efficiently;
`E[y] = 30 ~ 2 @ 50` snaps both engines to ~33.

Costs/caveats: ~2× fit wall-clock vs affine on CPU. No closed-form inverse —
`log_prob` (held-out NLL) works via per-element bracketed bisection (exact to
float32, offline only); `constraint_loss(with_logq=True)` is refused for dsf
(no parameter gradient through the iterative solve; implicit-diff custom_vjp
is the known fix if score-function features ever return to use).
