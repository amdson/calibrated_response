---
name: solver-forecast
description: Build a calibrated forecast as a maxent-solver joint — causal variable design, string-DSL elicitation, fit + constraint-report reading, anchor ablation, and resolution scoring. Use for any "what's the probability of X" forecasting question, especially knowledge-cutoff calibration tests.
---

# Forecasting with the maxent solver

Distilled from the SpaceX-IPO calibration test (Aug 2026, see
solver_additions.md §22). The fusion machinery works; every serious
failure was on the elicitation surface. This skill encodes the workflow
that survived.

## 1. Variables: causal drivers, not symptoms

- Model the **mechanism**, not the process trail. "Board approved",
  "S-1 filed" are the outcome measured at earlier timestamps — they
  explain nothing. Ask instead: *what would have to be true of the world
  for this to happen?* (For the IPO: capital demand vs private supply,
  liquidity demand vs tender supply, price incentive, owner's cost.)
- **Every hand-waved assumption becomes a variable.** If the reasoning
  says "X can raise private capital easily", `private_capacity_b` is a
  variable with a distribution, not a premise.
- Encode demand-vs-supply pairs and pin their difference with a
  structural equation: `gap = need - capacity`. The gap variable is then
  plottable and conditionable (`P(outcome | gap > 0) = ...`).
- Include **shock variables** for discrete events that would move the
  engine's inputs (mergers, regulatory action). Cutoff-adjacent "in
  discussions" chatter is already in motion: give it >= 0.5, not a base
  rate.
- **Bounds are infinite-confidence estimates.** Take the most extreme
  headline imaginable, double it, and state a deliberate small tail
  estimate near the edge (`P(v > big) = 0.05 ~ 1.0`) so the tail is a
  belief, not a silent truncation. (Truth landed at 1750 in a [100, 1200]
  box once. No diagnostic fired.)
- Write **resolution criteria into descriptions** (dates, thresholds,
  what counts), and define continuous outcomes so they resolve in every
  branch (e.g. "valuation at IPO pricing, or latest private mark if no
  IPO").
- Outcome nodes stay separate from drivers: the binaries being forecast,
  plus structure/size variables conditioned on the outcome.

## 2. Estimates: string DSL, both legs, weak anchors

Use `parse_natural_syntax` strings only (see natural_response.py):
`P(a = True | b > 0) = 0.3 ~ 0.9`, `E[v | e = True] = 35 ~ 15`,
`Corr(a, b) = 0.4 ~ 0.3`, `gap = need - capacity`, expression subjects
`P(x - y > 0) = 0.25`. `~ w` = self-uncertainty (log-odds for P: ~1.5
hunch, ~0.5 confident; value units for E). `@ n` = effective sample size.

- For every driver: a marginal (E or P) plus its tail if fat.
- For every mechanism conditional, state **both legs**
  (`P(out | gap > 0)` AND `P(out | gap < 0)`) — one leg leaves the other
  branch to maxent drift.
- **Necessary prerequisites get one-sided ceilings**, not tiny point
  estimates: `P(ipo = True | musk_intent = False) < 0.02`. The `<` form
  routes to the hinge penalty, which only charges for violation —
  "this door is closed" without forcing a near-empty conditioning slice
  to hit an exact noisy target (point-estimate legs there produce
  spurious report residuals). Prerequisite beliefs are one-sided;
  encode them that way.
- **Pick the channel by the shape of the knowledge**: conditionals for
  driver -> outcome causation; equations only for true identities
  (`gap = need - capacity` — each one costs a slow-converging soft link
  that blurs conditionals built on it, so never fabricate determinism);
  correlations specifically for **driver-driver overlap**. The last is
  mandatory whenever multiple conditional lifts target the same outcome:
  unstated overlap makes the solver shave every lift to reconcile them
  (the systematic-drag pattern in the report). Two or three
  `Corr(driver_a, driver_b)` / cross-driver conditionals let each lift
  hold near face value inside its own branch.
- **Branch on shared parents.** Two forecasts downstream of one belief
  (timing and structure both hinging on a merger) need explicit
  per-branch conditionals, or one wrong parent breaks both invisibly.
- Cumulative horizons (`by_aug`, `by_eoy`) need BOTH the implication
  `P(by_eoy = True | by_aug = True) = 0.99 ~ 0.4` AND a timing split
  `P(by_aug = True | by_eoy = True) = ...` — dropping the topline anchors
  otherwise silently deletes all timing information.
- **State topline anchors weakly** (`~ 1.0` or wider). With a tight
  anchor the fit matches it almost exactly and the mechanism's
  disagreement is silently discarded — the disagreement is the most
  valuable output (see §4).
- When a stated tail contradicts a stated mean, trust the tail and widen
  the mean's `~ w`.

### Two-tier elicitation: flood with cheap truths first

Weak-but-almost-certainly-correct statements are this solver's
comparative advantage — exploit them systematically:

- **Tier 1 (state many, error risk ~0)**: one-sided bounds
  (`P(x > t) < 0.1`, `E[cost] > 100`), correlation signs
  (`Corr(a, b) > 0 ~ 0.5`), orderings and accounting bounds as equation
  inequalities (`revenue > costs`, `need_b < 3 * capacity_b ~ N(0, 5)`),
  prerequisite ceilings. Three properties make these free: hinge
  penalties cost zero while satisfied; individually TRUE inequalities
  are automatically mutually consistent (their feasible sets all contain
  the truth — the double-counting disease is point-estimate-specific);
  and maxent commits to nothing beyond the carving, so weak statements
  never overclaim. Each one cheaply cuts impossible worlds.
- **Tier 2 (state few, spend error budget here)**: point estimates and
  two-leg conditionals only where there is genuine information. These
  carry the calibration risk, so every one deserves the tail/mean
  coherence check and a `~ w` matched to actual confidence.

## 3. Fit and verify (never ship unsmoke-tested)

Before handing anything over, run a starved local smoke: all strings
parse, `DistributionBuilder(VARS, ests)` has `skipped == []`,
`fit(steps=400, n_samples=512)` on CPU, then assert mechanism
*directions* (`P(out | gap>0) > P(out | gap<0)`), toplines in range, and
structural-equation RMS sane. Full runs: Colab GPU notebook,
clone-or-pull main, `steps≈2500`, `n_samples=2048` (see
benchmarks/colab_spacex_ipo_forecast.ipynb for the cell layout).

Readouts (mind the types):

```python
bins = {n: (s[n] > 0.5) for n in BINARY_NAMES}       # binaries only!
# continuous vars: report means/quantiles, never (v > 0.5).mean()
```

Reading `constraint_report()` (sorted by |err_rel|):

- **Systematic sign pattern** — every high-target driver conditional
  dragged down, every low leg dragged up — means the drivers overlap and
  their lifts were stated as if independent (double counting). Revise the
  most-redundant conditional down, not the solver.
- **A strained pair** (E[v|c] low while P(v > t|c) high) means the stated
  mean and tail were incoherent — widen the mean.
- Structural-equation rows (`EQ_*`) converge slowest; residual link RMS
  blurs conditionals that condition on the linked quantity.

## 4. Anchor ablation (the key diagnostic)

Refit with the topline anchors commented out and read the
mechanism-implied marginals:

- Mechanism ≈ anchor: the causal model independently reproduces the gut
  number — good redundancy, report either.
- Mechanism ≠ anchor: **investigate, don't anchor harder.** In the test
  the mechanism's number beat the tight anchor at resolution. The gap
  either exposes information the skeleton lacks (add the missing
  variable/conditional) or overconfidence in the gut number (weaken it).

## 5. Manual diagnostics the solver doesn't run yet

- **Boundary mass**: `np.mean(v > lo + 0.98*(hi-lo))` (and the low edge)
  per continuous variable; more than a few % piled at an edge = the box
  is truncating a belief — widen bounds and refit.
- **Branch table**: P(outcome) within each shock branch, from samples.
- Sensitivity of the headline number: starved-budget refits with one
  estimate perturbed, when it matters.

## 6. Resolution scoring

Keep a `RAW` dict of the directly-stated pre-solver numbers. At
resolution fill a `TRUTH` dict, then score fused marginals vs `RAW`
side by side (Brier + log loss, clip p to [1e-4, 1-1e-4]). Score
continuous outcomes as z-scores against the fitted mean/sd — and check
whether truth was even inside the variable's domain; a support miss is
the worst failure and scores no rows.

## Known informant biases (priced in from the last test)

- Cutoff-adjacent "discussions" resolve yes far above base rate.
- Base-rate process latencies (filings, approvals) do not transfer to
  priority projects of agentic founders — condition on the actor.
- "Structurally sensible" is not a forecast; it's a prior begging for a
  branch conditional.
