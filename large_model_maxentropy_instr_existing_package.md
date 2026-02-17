# Codex Instructions: Implement a JAX Stochastic Moment Matching Library ("jax_smm")

## 0. Objective

Implement a Python library using JAX that fits maximum-entropy / energy-based models
via stochastic moment matching (SMM) with sparse 1-, 2-, and 3-variable factors.

The model form is:

    p_theta(x) ∝ exp( sum_f <theta_f, phi_f(x_f)> )

Training objective:

    E_{p_theta}[phi_f] = target_moments_f

Training update (stochastic approximation):

    theta_f ← theta_f + η * (target_moments_f - model_moments_f)

where model_moments_f are estimated using persistent MCMC chains.

The library must support:

- Sparse factor graphs
- Mixed variable types:
    - categorical discrete
    - bounded continuous (initially via binning)
- Persistent MCMC sampling buffer
- JAX backend with jit, vmap, scan compatibility
- Ability to train from:
    - explicit marginal constraints
    - empirical samples


---

## 1. Package Structure

Create the following directory structure:

jax_smm/
    __init__.py
    graph.py
    features.py
    energy.py
    targets.py
    training.py
    diagnostics.py

    mcmc/
        __init__.py
        kernels.py
        hybrid.py
        buffer.py

    examples/
    tests/


---

## 2. Variable Representation

Define Variable dataclass:

Fields:

    name: str
    kind: Literal["categorical", "bounded_continuous"]

For categorical:

    num_states: int

For bounded continuous:

    low: float
    high: float
    num_bins: int   # discretization bins for v1

Internally maintain:

    categorical variables packed into array shape (n_cat,)
    continuous variables packed into array shape (n_cont,)

Provide mapping:

    var_id → categorical index OR continuous index


---

## 3. Factor Representation

Define Factor dataclass:

Fields:

    id: int
    var_ids: tuple[int, ...]     # length 1, 2, or 3
    theta: jnp.ndarray           # parameter table
    table_shape: tuple[int, ...] # shape matches variable cardinalities

Example shapes:

    unary categorical: (K,)
    pair categorical: (K_i, K_j)
    triple categorical: (K_i, K_j, K_k)

Graph object must contain:

    variables: list[Variable]
    factors: list[Factor]

    adjacency structures:

        var_to_factors: list[list[int]]
        factor_to_vars: list[tuple[int,...]]


---

## 4. State Representation

Define state PyTree:

    state = {
        "cat": jnp.ndarray[int32] shape (n_cat,)
        "cont": jnp.ndarray[float32] shape (n_cont,)
    }

For v1, continuous values must be discretized to bins before factor lookup.


---

## 5. Energy Evaluation

Implement:

    energy(graph, state) -> scalar

Compute:

    sum over factors of theta indexed by state assignments.

Example triple factor lookup:

    theta[i,j,k]

Avoid constructing one-hot vectors.

Use direct gather.


Also implement:

    local_energy_delta(graph, state, var_id, proposed_value)

This must only recompute energy contributions of factors connected to var_id.


---

## 6. Persistent Chain Buffer

Implement class PersistentBuffer:

Fields:

    states
    rng_keys

Methods:

    initialize(graph, num_chains)
    sample_step(graph, kernel)
    get_states()

Persistent chains must be reused across training iterations.


---

## 7. MCMC Kernels

### 7.1 Discrete Gibbs Update

For categorical variable i:

Compute:

    logits[k] = sum of adjacent factor energies if x_i = k

Sample:

    new_value = categorical(logits)

Replace value in state.

Implement:

    gibbs_update(graph, state, var_id, rng_key)


### 7.2 Continuous Variables (v1)

Use bin discretization.

Treat as categorical.


### 7.3 Full Sweep

Implement:

    gibbs_sweep(graph, state, rng_key)

Loop over variables sequentially.


---

## 8. Moment Estimation

Implement:

    estimate_moments(graph, states)

Return structure matching factor theta shapes.

Algorithm:

    initialize zero arrays matching theta

    for each chain state:
        for each factor:
            increment table cell corresponding to state assignment

    divide by number of chains

Must be implemented using JAX-friendly scan/vmap.


---

## 9. Target Moment Construction

Support two methods.

### 9.1 From explicit marginals

User provides:

    dict[factor_id] = probability table

Return as target moments.


### 9.2 From sample data

Input:

    dataset of states

Compute empirical counts for each factor.


---

## 10. Training Loop

Implement:

    fit_smm(graph, targets, config)

config fields:

    num_iterations
    num_chains
    mcmc_steps_per_iteration
    learning_rate
    optimizer: "sgd" or "adam"
    l2_regularization


Algorithm:

    initialize persistent chains

    for iteration in range(num_iterations):

        run mcmc_steps_per_iteration Gibbs sweeps

        model_moments = estimate_moments(graph, chains)

        gradient = target_moments - model_moments

        apply L2 regularization

        update theta with optimizer

        record diagnostics


Return:

    trained graph
    training history


---

## 11. Optimizers

Implement simple optimizers:

### SGD:

    theta += lr * gradient

### Adam:

Maintain:

    m
    v

Update using standard Adam equations.


---

## 12. Diagnostics

Implement:

    max_moment_error(graph, targets, model_moments)

Compute:

    max absolute difference across all factor tables

Also implement:

    mean_moment_error


---

## 13. Public API

Required functions:

    Variable(...)
    Factor(...)
    FactorGraph(...)

    build_targets_from_marginals(graph, marginal_tables)
    build_targets_from_samples(graph, sample_states)

    PersistentBuffer.initialize(...)
    fit_smm(...)

    energy(...)


---

## 14. Example Scripts

Create examples:

example_pairwise.py

example_triples.py

example_fit_from_samples.py

Each example:

    build small graph
    generate synthetic marginals
    train
    print moment error vs iteration


---

## 15. Testing

Implement tests:

tests/test_small_binary.py

Case:

    5 binary variables
    random theta
    compute exact marginals via brute force
    train via SMM
    verify marginals match within tolerance


tests/test_scaling.py

Case:

    100 variables sparse graph
    ensure training step runs


---

## 16. Performance Requirements

All core functions must be:

    JIT compatible

Avoid:

    Python loops inside jit functions

Use:

    jax.lax.scan
    jax.vmap


---

## 17. Implementation Order

Implement in this sequence:

1. graph.py
2. energy.py
3. mcmc/kernels.py
4. mcmc/buffer.py
5. targets.py
6. training.py
7. diagnostics.py
8. examples
9. tests


---

## 18. Future Extensions (not required v1)

- spline basis continuous variables
- HMC sampling
- block Gibbs sampling
- low-rank factor tables
- belief propagation inference


---

## 19. Deliverable

Working Python package:

    jax_smm

that can train sparse factor graphs via stochastic moment matching using JAX.

Must run on CPU and GPU.
