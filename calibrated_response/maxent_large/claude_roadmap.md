## Plan: Implement `maxent_large` Module — Continuous HMC Energy-Based MaxEnt

This module implements a scalable maximum entropy solver using continuous state-space HMC (Hamiltonian Monte Carlo) with persistent chains, declarative feature specifications compiled to JIT'd JAX functions, and optax-based parameter optimization. It is a **distinct model** from `maxent_large_v1` (which uses discrete Gibbs sampling over factor tables). All variables are treated as continuous on a normalized [0, 1] domain; binary variables use a continuous relaxation. Estimates are converted to `(feature_spec, target_expectation)` pairs via the existing `maxent.constraints` types as an intermediate representation.

**Steps**

1. **Create [calibrated_response/maxent_large/\_\_init\_\_.py](calibrated_response/maxent_large/__init__.py)** — Export `DistributionBuilder`, `MaxEntSolver`, `JAXSolverConfig`, and feature spec types. The package is currently not importable due to this missing file.

2. **Implement [calibrated_response/maxent_large/features.py](calibrated_response/maxent_large/features.py)** — Declarative feature specification system:
   - Define frozen dataclasses for each feature type:
     - `MomentFeature(var_idx, order=1)` — compiles to `x[var_idx]**order`
     - `SoftThresholdFeature(var_idx, threshold, direction, sharpness=20.0)` — compiles to `jax.nn.sigmoid(sharpness * (x[var_idx] - threshold))` (or its negation for "less")
     - `SoftIndicatorFeature(var_idx, lower, upper, sharpness=20.0)` — product of two sigmoids for bin membership
     - `ProductMomentFeature(var_indices)` — compiles to `prod(x[i] for i in var_indices)`
     - `ConditionalSoftThresholdFeature(target_var, target_threshold, target_direction, cond_var, cond_threshold, cond_direction, sharpness)` — product of two soft thresholds for conditional constraints
     - `ConditionalSoftIndicatorFeature` — similar, for conditional-on-bins
   - Define a `FeatureSpec` union type alias over all feature dataclasses
   - Implement `compile_feature(spec: FeatureSpec) -> Callable[[jnp.ndarray], jnp.ndarray]` — returns a pure JAX function `f(x) -> scalar` for a single feature
   - Implement `compile_feature_vector(specs: Sequence[FeatureSpec]) -> Callable[[jnp.ndarray], jnp.ndarray]` — stacks all features into one JIT-compiled function `f(x) -> jnp.ndarray` of shape `(K,)` using `jax.vmap` or manual stacking
   - Smooth sigmoid surrogates are essential: HMC needs differentiable energy w.r.t. state `x`, and hard indicator functions have zero gradients almost everywhere

3. **Implement [calibrated_response/maxent_large/mcmc.py](calibrated_response/maxent_large/mcmc.py)** — JIT-compiled HMC with persistent chains:
   - `HMCConfig` dataclass: `step_size=0.01`, `num_leapfrog_steps=10`, `target_accept_rate=0.65`, `adapt_step_size=True`
   - `hmc_step(energy_fn, state, rng_key, step_size, num_leapfrog)` — single HMC transition: sample momentum from N(0,I), leapfrog integration, Metropolis accept/reject. Returns `(new_state, accepted, new_key)`. Mark `@jit`.
   - `leapfrog(energy_fn, state, momentum, step_size, num_steps)` — leapfrog integrator using `jax.grad(energy_fn)` for position updates. Use `jax.lax.fori_loop` for JIT compatibility.
   - `hmc_chain_step(energy_fn, states, rng_key, step_size, num_leapfrog)` — `vmap` over the chain dimension to advance C chains in parallel
   - `PersistentBuffer` frozen dataclass: `states` shape `(C, n_vars)` as `jnp.float32`, `rng_key`, `step_size`. Method `advance(energy_fn, n_steps, ...)` that calls `hmc_chain_step` repeatedly via `jax.lax.scan`, returns new buffer. Method `initialize(n_chains, n_vars, rng_key)` — uniform [0, 1] initialization.
   - Step-size adaptation: dual-averaging scheme tracking accept rate, or simple multiplicative rule adjusting toward target acceptance.
   - **Boundary handling**: Clamp or reflect states into [0, 1] after each leapfrog step since all normalized variables live on this interval.

4. **Rewrite [calibrated_response/maxent_large/maxent_solver.py](calibrated_response/maxent_large/maxent_solver.py)** — Core solver:
   - Keep `JAXSolverConfig` but add HMC-specific fields: `num_chains=128`, `num_iterations=300`, `mcmc_steps_per_iteration=5`, `hmc_step_size=0.01`, `hmc_leapfrog_steps=10`, `learning_rate=0.01`, `l2_regularization=1e-4`, `grad_clip=5.0`, `adapt_step_size=True`
   - `MaxEntSolver.__init__(config)` — stores config
   - `build(variables, feature_specs, feature_targets)`:
     - Compile `feature_specs` → single JIT'd feature vector function via `compile_feature_vector`
     - Build energy function: `energy(theta, x) = -jnp.dot(theta, feature_vector(x))`. JIT-compile it.
     - Build `grad_x_energy = jax.grad(energy, argnums=1)` for HMC
     - Initialize `theta = jnp.zeros(K)`, optax optimizer state
     - Initialize `PersistentBuffer`
     - Store all compiled artifacts on `self`
   - `solve()`:
     - Training loop (per roadmap Steps A–D):
       - **A**: `buffer.advance(energy_fn, n_steps)` — advance persistent HMC chains
       - **B**: Compute feature expectations: `vmap(feature_vector)(buffer.states).mean(axis=0)` — shape `(K,)`
       - **C**: Gradient: `g = feature_targets - model_expectations - l2_reg * theta`, clip
       - **D**: `optax` update on `theta`
     - Convergence: monitor `max(|feature_targets - model_expectations|)`
     - Returns `(theta, info_dict)` where info contains chain states, history, diagnostics

5. **Implement constraint-to-feature adapter in [calibrated_response/maxent_large/distribution_builder.py](calibrated_response/maxent_large/distribution_builder.py)** — Complete rewrite:
   - `__init__(variables, estimates, solver_config)`:
     - Create `ContinuousDomainNormalizer` from [normalizer.py](calibrated_response/maxent_large/normalizer.py) (the reference `self.normalizer` that was missing)
     - Convert estimates → `ConstraintUnion` list (reuse `_estimate_to_constraint` logic from [calibrated_response/maxent/distribution_builder.py](calibrated_response/maxent/distribution_builder.py))
     - Convert constraints → `(FeatureSpec, target_value)` pairs:
       - `ProbabilityConstraint(var, lower, upper, p)` → `SoftIndicatorFeature(var_idx, norm_lower, norm_upper)`, target = `p`
       - `MeanConstraint(var, value)` → `MomentFeature(var_idx, order=1)`, target = normalized value
       - `ThresholdConstraint(var, threshold, p)` → `SoftThresholdFeature(var_idx, norm_threshold)`, target = `p`
       - `ConditionalThresholdConstraint` → `ConditionalSoftThresholdFeature(...)`, target derived from conditional probability
       - `ConditionalMeanConstraint` → pair of features: `ProductMomentFeature([target, cond_indicator])` and `SoftThresholdFeature` for the condition, with targets computed from the conditional expectation relationship: $E[X|C] = E[X \cdot I(C)] / P(C)$
     - Store feature specs and targets
   - `build(target_variable=None)`:
     - Call `solver.build(...)` then `solver.solve()`
     - Extract marginal for target variable by histogramming the continuous chain samples into bins (using `normalizer.normalized_bin_edges`)
     - Denormalize bin edges back to original domain via `normalizer.denormalize_edges`
     - Return `(HistogramDistribution, info_dict)` — same interface as `maxent/DistributionBuilder` and `maxent_large_v1/DistributionBuilder`
   - `get_all_marginals(info)` — histogram all variables from chain states stored in info
   - Helper: `_histogram_marginal(states, var_idx, n_bins)` — `np.histogram(states[:, var_idx], bins=n_bins, range=(0, 1), density=True)` → normalized probabilities

6. **Keep [calibrated_response/maxent_large/normalizer.py](calibrated_response/maxent_large/normalizer.py) as-is** — already fully implemented and correct.

7. **Add tests in [tests/maxent_large/](tests/maxent_large/)**:
   - `test_features.py` — verify each feature spec compiles to a correct JAX function, gradients are nonzero (smooth surrogate check), `compile_feature_vector` output shape
   - `test_mcmc.py` — verify HMC samples from a known Gaussian energy converge to correct mean/variance; test `PersistentBuffer` advance; test boundary reflection
   - `test_solver.py` — single-variable mean constraint → correct expectation; multi-variable with conditional → correct marginals
   - `test_distribution_builder.py` — end-to-end: `Variable` + `Estimate` objects → `HistogramDistribution` with reasonable properties (mean near target, probabilities sum to 1)

**Verification**

- Run `pytest tests/maxent_large/` for unit tests
- Sanity test: create a single `ContinuousVariable(0, 100)`, add `ExpectationEstimate(mean=70)`, verify the resulting `HistogramDistribution.mean()` ≈ 70
- Multi-variable test: two continuous variables with a conditional probability estimate, verify the conditional structure appears in the joint samples
- Convergence diagnostics: check `info['history']` shows decreasing feature-moment mismatch

**Decisions**

- **Smooth sigmoids over hard indicators**: HMC requires differentiable energy gradients w.r.t. state. Hard indicator features $I(x > t)$ have zero gradient a.e., making HMC blind to those constraints. Sigmoid surrogates with configurable sharpness (default 20.0) solve this while closely approximating the indicator behavior.
- **Boundary handling via reflection**: States are clamped/reflected to [0, 1] after each leapfrog step, since all normalized continuous variables live on this compact domain. This avoids the need for unconstrained reparameterization (e.g., logit transform), which would complicate feature interpretation.
- **Conditional constraints decomposition**: Conditional expectations $E[X|C]$ are encoded as pairs of features exploiting $E[X|C] \cdot P(C) = E[X \cdot I(C)]$, which are directly expressible as product features with known targets. This avoids ratio-of-expectations in the loss.
- **Binary variables as continuous [0, 1]**: Binary variables are treated as continuous on [0, 1] during HMC, allowing gradient flow. Marginals are extracted by thresholding at 0.5 for the `BinaryDistribution` case.
