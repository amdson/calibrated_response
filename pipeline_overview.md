# Pipeline overview: query → statements → distribution builder → solver

State of the calibrated-response pipeline as of 2026-07-07. The pipeline turns a
natural-language forecasting question into a fitted joint distribution over a
small set of auxiliary variables, then reads off the answer (and full marginals)
from that joint.

```
question text ──► VariableGenerator ──► [Variable, ...]
                        │
                        ▼
              NaturalEstimateGenerator ──► "P(X>5|Y=True)=0.7" ──parse──► [Estimate, ...]
                                                                              │
                                                                              ▼
                                                             DistributionBuilder (per solver)
                                                                              │
                                                          estimates → constraint targets/weights
                                                                              │
                                                                              ▼
                                                                     solver fit (maxent)
                                                                              │
                                                                              ▼
                                                  Distribution marginals + fit report + diagnostics
```

Five solver families implement the same builder interface; the current flagship
is the **flow sampler** (`maxent_sampler`), packaged into a builder in this
round of work.

---

## Stage 1 — Query

A query is a plain-text forecasting question, optionally with background /
resolution criteria. The working benchmark is Metaculus binary questions:

- `metaculus/tiny_dataset.json` — 24 resolved questions (keys: `id`, `question`,
  `resolution_criteria`, `background`, `freeze_datetime_value` = market price at
  freeze, `resolved_to`).
- `metaculus/eval_utils.py` — dataset iteration and scoring helpers.
- `metaculus/llm_cache.json` — cached LLM elicitation output keyed by question
  id, so solver experiments rerun without API calls.

Nothing constrains the pipeline to binary questions — the target is just one
variable among several — but the eval harness currently scores binary
resolution (P(Yes) vs. market vs. outcome).

## Stage 2 — Statements (variables + estimates)

Two LLM calls decompose the question into a typed, machine-checkable belief set
(`calibrated_response/generation/`, prompts in `generation/prompts.py`):

1. **`VariableGenerator.generate(question, n_variables)`** → list of pydantic
   `Variable`s (`models/variable.py`):
   - `BinaryVariable(name, description)`
   - `ContinuousVariable(name, description, lower_bound, upper_bound, unit)` —
     bounded domains are required downstream.

2. **`NaturalEstimateGenerator.generate(question, variables, num_estimates)`**
   → the LLM emits estimates in compact math notation (`P(X > 5) = 0.7`,
   `E[Cost | Y = True] = 100`), parsed by
   `models/natural_response.parse_natural_syntax` into structured objects.

The structured estimate schema (`models/query.py`, `EstimateUnion`) currently
supports four estimate types:

| Type | Example | Payload |
|---|---|---|
| `ProbabilityEstimate` | `P(count > 0.5) = 0.88` | proposition, probability |
| `ExpectationEstimate` | `E[count] = 1.9` | variable, expected_value |
| `ConditionalProbabilityEstimate` | `P(sweep=True \| nl_events > 7) = 0.96` | proposition, conditions, probability |
| `ConditionalExpectationEstimate` | `E[count \| norway > 8] = 3.2` | variable, conditions, expected_value |

Propositions are `EqualityProposition(value: bool/str)` (binary variables) or
`InequalityProposition(threshold, is_lower_bound)` (continuous variables);
conditionals may carry multiple conditions (comma-separated in the natural
syntax). Everything round-trips through JSON via
`TypeAdapter(EstimateUnion).validate_python`, which is how the cache works.

**Known gap:** the natural-estimate prompt advertises richer statement forms
(`Cov`, `Corr`, `Var`, independence, inequalities between estimates) but the
parser regex only accepts `P(...)=v` / `E[...]=v` forms, and `EstimateUnion`
only has the four types above. The flow solver's constraint grammar already
supports `cov`/`corr`/`mmd` targets, so extending the schema + parser is the
cheapest way to widen the pipe.

## Stage 3 — Distribution builders

A `DistributionBuilder` is the adapter from `(variables, estimates)` to one
solver's native constraint language, with a shared interface so the eval loop
can swap solvers by changing an import:

- `build(target_variable, **fit_kw)` → `(Distribution, info)`
- `get_all_marginals()` → `dict[name, Distribution]`
- un-mappable inputs land in `builder.skipped` (with reasons) instead of raising

Outputs use the shared `Distribution` objects (`models/distribution.py`):
`HistogramDistribution(bin_edges, bin_probabilities)` with
mean/std/quantile/cdf/sample/summary, and `BinaryDistribution(probability)`.

### Implementations

| Module | Solver behind it | Status |
|---|---|---|
| `maxent/` | 1-D / small discrete maxent | legacy, first version |
| `maxent_pgmax/` | discretized factor-graph BP (pgmax) | legacy |
| `maxent_large/` | energy model + MCMC | superseded by maxent_smm |
| `maxent_smm/` | energy model + persistent-HMC stochastic moment matching | reference baseline; runs in the metaculus eval |
| `maxent_sampler/` | **flow sampler** (RealNVP, exact entropy) | **new — current focus** |

Related but separate tracks: `tn/` (tensor-network chains/trees — exact
contractions, discrete), `pc/` (polynomial/probabilistic circuits),
`energy_models/`. These have solver code but no `Variable`/`Estimate` builder;
they are compared against the samplers in `examples/` and `benchmarks/` (the
constraint-reconstruction benchmark, where the flow is registered as the
`flow` engine — see `benchmarks/README.md`).

### The flow builder (`maxent_sampler/distribution_builder.py`)

`DistributionBuilder(variables, estimates, prob_sd=0.05, value_rel_sd=0.05,
sharpness=20.0, robust=False, p_broken=0.05, n_layers=8, hidden=64, ...)`.
Mapping decisions:

- **Variables.** Continuous variables keep their original bounded domain (no
  normalizer needed — the flow squashes into the box itself). Binary variables
  become continuous sites on `[0, 1]`, read out as `P(x > 0.5)` and returned as
  `BinaryDistribution`. Unbounded/degenerate domains are skipped.
- **Events.** Inequality propositions → soft sigmoid indicators with
  span-normalized sharpness (effective slope `sharpness / span`, so mixed units
  behave); equality propositions only on binary variables (threshold 0.5).
  Readouts and reports always use *hard* indicators on fresh samples — the soft
  features exist only inside the loss.
- **Conditionals** condition on the **conjunction** (product of indicators) of
  all condition events, unlike maxent_smm which builds one centered feature per
  condition.
- **Weights** come from belief widths: probabilities `w = 1/(2·prob_sd²)`;
  expectations `w = 1/(2·(value_rel_sd·span)²)` so weights are unit-free.
  Out-of-domain expectation targets are clipped with a warning.
- **`robust=True`** wraps every estimate in an `onoff` constraint: a learnable
  Bernoulli credence per estimate (prior `1 − p_broken`) that the optimizer can
  lower — at a KL price — instead of letting one inconsistent estimate distort
  the whole joint. Credences are reported per estimate.

Extra readouts beyond the parity interface: `sample(n)` / `sample_dict(n)`
(original units, binaries thresholded), exact `entropy()` in nats, and
`constraint_report(n_samples)` — per-estimate target vs. fitted vs. error, plus
`p_cond` (probability of each conditioning event = the Monte-Carlo budget behind
that constraint) and credences in robust mode.

## Stage 4 — Solver (flow maxent)

`FlowSamplerModel` (`maxent_sampler/flow_model.py`): an invertible RealNVP-style
flow `x = g_θ(z), z ~ N(0, I)` mapped into the variable box. Because the flow is
invertible, the joint entropy is **exact**:
`H(x) = h_const + E[log|det J_θ(z)|]` — no histogram or kernel proxy, and it
scales O(D) in the number of variables. Invertibility also makes it a tractable
*density* model: `FlowSamplerModel.log_prob(params, x)` evaluates the exact
joint density via the inverse pass (used for held-out NLL in `benchmarks/`).

Fitting is soft-constrained maximum entropy:

```
loss(θ) = Σ_c w_c · (E_θ[f_c] − target_c)²   −   entropy_reg · H(θ)
```

estimated on a fresh latent batch every Adam step (`fit_adam_stochastic`; the
model deliberately refuses non-stochastic backends — optimizing against a fixed
z batch overfits the batch). `entropy_reg=1.0` is the true maxent scale.
Constraint grammar (`maxent_sampler/model.py`): `expect`, `cond_expect`, `cov`,
`corr`, `mmd`, and `onoff` (robust gates).

Design note: the flow *smooths* the piecewise-flat spikes of exact maxent
solutions. For real-world forecasting variables this is desirable — the sharp
plateaus of true maxent joints are an artifact of the constraint form, not a
belief anyone holds.

## Stage 5 — Evaluation and current results

- Eval loop: `metaculus/run_maxent_smm.ipynb` — per question, load cached
  variables/estimates, build, fit, `P(Yes) = mean(samples[:, target])`, score
  against market price and resolution. The flow builder is drop-in compatible
  but has not yet been swept over the 24-question set.
- Showcase: `examples/constraints_tests/flow_distribution_builder.ipynb`
  (executed, no API key needed) — Metaculus 41370 "podium sweep at the 2026
  Winter Olympics", 4 variables / 10 real cached estimates. Fitted
  **P(sweep) = 0.899** vs. LLM direct 0.88, market 0.90, resolved YES. The
  constraint report surfaces a genuine LLM inconsistency
  (`E[total_sweeps_count] = 1.9` vs. `P(count > 0.5) = 0.88` and
  `E[count | norway > 8] = 3.2`); robust mode assigns that estimate the lowest
  credence. Exact joint entropy 5.89 nats.
- Solver-side validation: `examples/constraints_tests/flow_chain.ipynb` — chain
  dependency propagation, unfit marginals land on the analytic recursion and
  entropy sits near the analytic maxent optimum.
- Cost: ~1.5–3 min per question on CPU at 2000 Adam steps (per-step cost, not
  compile); `steps=1000` is nearly as good for a fast pass.

## Open items

1. **Run the metaculus sweep** with the flow builder and compare Brier/log
   scores against `maxent_smm` and the direct-LLM baseline.
2. **Widen the statement schema**: parser + `EstimateUnion` support for
   `Corr`/`Cov`/`Var` statements (solver grammar already has `cov`/`corr`).
3. **Rare conditioning events**: a conditional with small `p_cond` gets little
   gradient signal per batch; the report exposes `p_cond`, but there is no
   automatic batch-size escalation or importance trick yet.
4. **Discrete/count variables** are modeled as continuous; integer-valued
   readouts (e.g. `total_sweeps_count`) rely on threshold events rather than a
   proper discrete site.
5. **Consistency as a first-class output**: maxent residuals + robust credences
   already flag inconsistent estimate sets; not yet fed back into elicitation
   (e.g. re-asking the LLM about its lowest-credence estimate).
