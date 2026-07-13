# Elicitation strategies: getting better statements out of the LLM

Companion to `experiments.md` (which tracks *what to run*); this is the
idea inventory for the elicitation side — how to get a richer, more
consistent, more solver-friendly set of statements per question. Each
entry notes what it needs: **P** = prompt/schema change only, **S** =
new solver constraint kind, **$** = extra LLM calls per question.

Current protocol for reference: one structured call → 4–6 variables
(target injected as `variables[0]`) + ~10 estimates from the grammar
{P, P(·|·), E, E(·|·), Corr (binary-binary banned)}, bounded retry loop
enforcing a direct target estimate + ≥1 coupling.

## 1. Expanding the statement vocabulary

| statement type | why | needs |
|---|---|---|
| quantile / interval beliefs — `P10/P50/P90(X)` | LLMs state ranges more reliably than means; three quantiles pin a marginal far better than one E[X]; directly reduces the "expectation + span-sd" guesswork | P + S (quantile penalty: `P(X < q) = p` is just an expect on an indicator — nearly free) |
| comparative probabilities — `P(A) > P(B)`, `P(A | C) > P(A)` | relative judgments are the thing LLMs are most calibrated at; inequality constraints are cheap one-sided penalties | P + S (hinge loss) |
| odds ratios / risk ratios — `P(T|A) / P(T|¬A) = r` | elicits the *strength* of a dependence without asking for two absolute numbers that may each be biased the same way | P + S (ratio of two soft conditionals, logit-space residual) |
| sign-of-effect / monotonicity — "T is increasing in X" | almost every question has known directional structure; very hard for the LLM to get wrong; constrains the joint where point estimates are silent | P + S (penalize negative correlation of T with X, or E[T·1(X>med)] ≥ E[T·1(X<med)]) |
| mutual exclusivity / partition — "exactly one of A, B, C" | lets the LLM define scenario variables cleanly; currently inexpressible, so scenario structure leaks into badly-correlated binaries | P + S (penalty on Σ soft indicators − 1) |
| implication — "A ⇒ B" | cheap, unambiguous, and the LLM volunteers these naturally in free text | P (sugar for `P(B|A) = 1`, which the certainty clip now handles as 0.99) |
| conditional on ¬A | we only elicit P(T\|A); adding P(T\|¬A) makes each coupling twice as informative and enables an internal LTP check at build time | P |

## 2. Variable extraction schemes

- **Brainstorm → select (two-stage):** first call lists 10–15 candidate
  drivers in free text; second call picks the 4–6 most *informative and
  estimable* and formalizes bounds/units. Rationale: the current
  single-shot forces ideation and formalization simultaneously; bad
  variables (unbounded, unobservable, redundant) are the main source of
  skipped constraints. [$: 2 calls]
- **Observable-proxy bias:** instruct that variables should be things
  with a checkable historical base rate (counts, indices, prices,
  event-happened-by-date), not latent abstractions ("momentum",
  "sentiment"). Estimates on observables anchor better. [P]
- **Target back-chaining:** ask "what would have to be true for
  target = True?" and "…for False?" — variables come out as necessary /
  sufficient conditions, which naturally produce strong conditionals
  instead of weak correlations. [P]
- **Fermi decomposition for quantities:** when the target is a threshold
  on a quantity, elicit the factor decomposition (rate × exposure ×
  share) as the variables. The joint then does real propagation work
  instead of decorating a direct guess. [P]
- **Independent double extraction:** run variable extraction twice at
  temperature, merge/dedupe, keep variables that appear twice (stability
  filter). [$: 2 calls]

## 3. More variables / more estimates

- Scale `--n-estimates` 10 → 15–20 and `--n-variables` 4–6 → 6–8, but
  watch the two known failure modes: LTP-violation rate grows with
  estimate count, and conditionals on rare events (p_cond tail) get
  noisy. Already levers B.3/B.4 in experiments.md.
- **Split elicitation:** call 1 fixes the variable set; call 2 elicits a
  *dense battery* over exactly those variables (every variable gets a
  marginal statement, target gets a conditional on every other
  variable). Removes attention dilution and lets call 2's schema be
  much stricter. [$: 2 calls]
- **Per-variable marginal pinning:** currently non-target variables often
  get only an E[X]; requiring a quantile pair per variable makes the
  non-target marginals real, which is what the conditionals lever on.
  [P, pairs with quantile statements above]

## 4. Structured relationship elicitation

- **Dependency-sketch first:** elicit a mini-DAG ("which variables
  directly influence the target / each other — edges only"), then elicit
  conditionals *only along edges*. Stops the model from inventing
  couplings between unrelated variables to satisfy the ≥3-couplings
  request, which is where infeasible estimates come from. [$ or P]
- **Both-arms conditionals:** for each edge, elicit P(T|A) *and* P(T|¬A)
  (or above/below-median for continuous). Two-sided couplings are what
  the maxent joint actually needs; one-sided ones let entropy fill the
  other arm arbitrarily. [P]
- **Scenario mixture elicitation:** ask for 3–4 named scenarios with
  probabilities and per-scenario variable settings; compile each into
  conditional constraints gated on a scenario partition variable. This is
  the most human-forecaster-like format and gives the joint real
  multimodal structure. Needs partition support (§1). [P + S]
- **Strength-labelled dependence instead of Corr:** replace numeric Corr
  with categorical {weak/moderate/strong, +/−} mapped to fixed corr
  targets with wide sd. LLM numeric correlations were the least reliable
  statements in the pilot (19/40 infeasible before the binary-binary
  ban). [P]

## 5. Consistency and calibration devices (same call, no new kinds)

- **Complement probes:** elicit P(A) and P(¬A) for one variable; if they
  don't sum to ~1, that's a per-response reliability signal → scale all
  sds for that response. [P]
- **Bayes-pair probes:** elicit P(T|A) and P(A|T) plus marginals; check
  consistency at build time; inconsistency → widen sds or raise
  p_broken for that response. [P]
- **Self-rated confidence per estimate:** one enum field
  {low/med/high} per estimate → per-estimate sd multiplier. Cheap and
  directly consumable by the builder. [P]
- **Frequency framing / reference classes:** "out of 100 similar
  worlds…" phrasing and "state the historical base rate for this class
  of event, then adjust" — the two best-evidenced debiasing prompts in
  the forecasting literature. [P]

## 6. Multi-pass protocols ($: one extra call each)

- **Critique-and-revise:** feed the model its own statement list, ask it
  to flag internal inconsistencies (LTP checks, implausible
  conditionals) and return a corrected list. Directly targets the
  LTP-violation subset — the place where flow currently wins.
- **Adversarial pass:** "argue the target is unlikely, then update your
  estimates" — counteracts the acquiescence/optimism drift seen in the
  upward bias on rare events.
- **Statement-level ensembling:** k independent elicitations, aggregate
  per-statement (median target, spread → sd) rather than per-final-
  probability. Feeds the solver *measured* uncertainty instead of a
  global prob_logit_sd. This is experiments.md B.2 and probably the
  highest-value item in this file.

## Suggested first wave (cheap, compatible with current solver)

1. Both-arms conditionals + conditional-on-¬A (P only; makes existing
   machinery twice as informative).
2. Self-rated confidence → per-estimate sd (P only; tiny builder change).
3. Quantile statements P(X < q) = p (small S: indicator expect — reuses
   existing kinds; kills the weakest current statement type, E[X]).
4. Critique-and-revise pass (1 extra call; aims squarely at the LTP
   subset where the joint has its edge).
