# Codex Instructions: Implement Large-Scale MaxEnt via JAX Stochastic Moment Matching (Non-pgmax)

## 0. Objective
Implement a new large-scale MaxEnt package based on **stochastic moment matching (SMM) with persistent MCMC**, not belief propagation and not `pgmax`.

Model family:

`p_theta(x) ∝ exp(sum_f <theta_f, phi_f(x_f)>)`

Training condition:

`E_{p_theta}[phi_f] = target_moments_f`

Stochastic update:

`theta_f <- theta_f + eta * (target_moments_f - model_moments_f)`

Use JAX end-to-end (`jit`, `vmap`, `lax.scan`) for scalable training.

## 1. Constraints From Existing Repository (Must Respect)

### Existing model contracts
- Variables come from `calibrated_response/models/variable.py`
  - `BinaryVariable`
  - `ContinuousVariable`
- Estimates come from `calibrated_response/models/query.py`
  - `ProbabilityEstimate`
  - `ExpectationEstimate`
  - `ConditionalProbabilityEstimate`
  - `ConditionalExpectationEstimate`
- Output distribution type remains `HistogramDistribution` from `calibrated_response/models/distribution.py`.

### Existing builder usage pattern
Any new integration path should preserve the ergonomics of:
- `DistributionBuilder(...).build(target_variable=...) -> (HistogramDistribution, info)`
- `DistributionBuilder(...).get_all_marginals(info) -> dict[str, HistogramDistribution]`

### Architectural requirement
- This new package must be separate from `calibrated_response/maxent_pgmax`.
- Do not add `pgmax` dependency to this pathway.
- Do not rely on dense joint tensors like `calibrated_response/maxent/multivariate_solver.py` for large-N.

## 2. New Package Layout

Create:

`calibrated_response/maxent_smm/`
- `__init__.py`
- `graph.py`
- `features.py`
- `energy.py`
- `targets.py`
- `training.py`
- `diagnostics.py`
- `adapters.py`  (bridge from existing `EstimateUnion`/`Variable` models)
- `distribution_builder.py` (repo-compatible builder)

`calibrated_response/maxent_smm/mcmc/`
- `__init__.py`
- `kernels.py`
- `hybrid.py`
- `buffer.py`

Tests:
- `tests/maxent_smm/test_small_binary.py`
- `tests/maxent_smm/test_adapters.py`
- `tests/maxent_smm/test_distribution_builder_integration.py`
- `tests/maxent_smm/test_scaling.py`

Examples:
- `examples/maxent_smm_pairwise.py`
- `examples/maxent_smm_triples.py`
- `examples/maxent_smm_fit_from_samples.py`

## 3. Core Data Structures

### 3.1 Variable representation (internal)
Define internal `SMMVariable`:
- `name: str`
- `kind: Literal["categorical", "bounded_continuous"]`
- categorical: `num_states: int`
- bounded continuous: `low: float`, `high: float`, `num_bins: int`

Provide mapping tables from repository models:
- `BinaryVariable -> categorical(2)`
- `ContinuousVariable -> bounded_continuous(num_bins=config.max_bins)`

### 3.2 Factor representation
Define internal `Factor`:
- `id: int`
- `var_ids: tuple[int, ...]` (length 1, 2, or 3)
- `theta: jnp.ndarray`
- `table_shape: tuple[int, ...]`

Define `FactorGraph`:
- `variables: list[SMMVariable]`
- `factors: list[Factor]`
- `var_to_factors`
- `factor_to_vars`

## 4. State & Energy

State PyTree:
- `state["cat"]: int32[n_cat]`
- `state["cont"]: float32[n_cont]` (v1 discretized to bins before factor lookup)

Implement:
- `energy(graph, state) -> scalar`
- `local_energy_delta(graph, state, var_id, proposed_value) -> scalar`

Use factor-local recomputation only for delta.

## 5. MCMC & Persistent Buffer

Implement persistent MCMC chains:
- `PersistentBuffer.initialize(graph, num_chains, rng)`
- `PersistentBuffer.sample_step(graph, kernel, n_sweeps, rng)`
- `PersistentBuffer.get_states()`

Kernels:
- categorical Gibbs update per variable
- bounded continuous v1 handled through discretized bins
- full sweep over variables

All training iterations must reuse persistent chains.

## 6. Targets and Existing Estimate Adapters

Implement target builders:
- `build_targets_from_marginals(graph, marginal_tables)`
- `build_targets_from_samples(graph, sample_states)`

Implement adapters from repository estimates:
- `ProbabilityEstimate` -> unary moment targets
- `ExpectationEstimate` -> unary expected-value moments
- `ConditionalProbabilityEstimate` -> pairwise/triple moments where representable
- `ConditionalExpectationEstimate` -> conditional moment targets (or explicit unsupported reason if excluded in v1)

Rules:
- Unsupported estimate forms must be captured in diagnostics (`skipped_constraints`).
- Keep unsupported handling explicit and deterministic.

## 7. Training Loop

Implement `fit_smm(...)` in `training.py`:
- Inputs:
  - graph
  - targets
  - num_iterations
  - mcmc_steps_per_iteration
  - learning_rate
  - optimizer (`"sgd"` or `"adam"`)
  - l2_regularization
  - rng
- Per iteration:
  1. advance persistent chains
  2. estimate model moments from chain states
  3. compute gradient = target - model moments
  4. apply regularization
  5. optimizer update on each factor theta
  6. record diagnostics
- Return:
  - trained graph
  - history dict

## 8. Optimizers

Implement:
- SGD: `theta += lr * grad`
- Adam with per-factor `m`, `v`, bias correction

All optimizer state must be JAX-friendly.

## 9. Diagnostics

Implement:
- `max_moment_error(...)`
- `mean_moment_error(...)`
- training history fields:
  - `iteration`
  - `max_error`
  - `mean_error`
  - `acceptance` (if kernel reports it)
  - runtime per iter
  - `skipped_constraints`

## 10. Repo-Compatible Distribution Builder

Add `calibrated_response/maxent_smm/distribution_builder.py` with API parallel to existing builders:
- Constructor accepts `variables`, `estimates`, and SMM config.
- `build(target_variable=None)`:
  - run adapter to build graph + targets
  - train via `fit_smm`
  - estimate marginals from persistent chains
  - return `HistogramDistribution` for target variable
  - include info diagnostics
- `get_all_marginals(info)` returns histograms for every variable.

Important:
- Do not reconstruct a dense joint tensor by default.
- If a joint approximation is included, mark it clearly as approximate.

## 11. Configuration

Define `SMMConfig` (new module-local config dataclass):
- `max_bins`
- `num_chains`
- `num_iterations`
- `mcmc_steps_per_iteration`
- `learning_rate`
- `optimizer`
- `l2_regularization`
- `seed`
- optional stability knobs (`grad_clip`, `logit_clip`, `temperature`)

Keep config independent from legacy `maxent` and `maxent_pgmax` configs.

## 12. Testing Requirements

### Functional correctness
- Small binary graph: recover known moments within tolerance against brute force.
- Adapter tests: each `EstimateUnion` mapping produces expected targets or explicit skip reason.

### Integration
- Builder returns normalized `HistogramDistribution`.
- `get_all_marginals` returns all variable names with normalized probs.

### Scaling
- Sparse graph with ~100 variables runs one training step and one marginal extraction without OOM.

## 13. Performance Requirements

- JIT compile inner loops (energy eval, Gibbs sweeps, moment accumulation, updates).
- Avoid Python loops inside `jit` hot paths.
- Use `vmap` across chains and `lax.scan` across sweeps/iterations where practical.

## 14. Implementation Order
1. `graph.py`
2. `energy.py`
3. `mcmc/kernels.py`
4. `mcmc/buffer.py`
5. `targets.py`
6. `adapters.py`
7. `training.py`
8. `diagnostics.py`
9. `distribution_builder.py`
10. tests
11. examples

## 15. Non-Goals (v1)
- Exact dense-joint MaxEnt optimization at large N.
- Any dependency on `pgmax`.
- Full support for arbitrary high-order conditional constraints beyond supported factor templates.

## 16. Deliverable
A new package under `calibrated_response/maxent_smm` that scales to large variable counts via JAX SMM + persistent MCMC, integrates with existing repo models/Builder interfaces, and is explicitly independent from `maxent_pgmax`.
