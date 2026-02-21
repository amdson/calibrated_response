We just talked and came up with the following:

Moment-Matching Loss for Energy-Based Models
Problem
Fit a sparse MRF on $[0,1]^n$ by matching moments:
$$J(\theta) = \frac{1}{2}|\mathbb{E}{p\theta}[f(x)] - \mu|^2 + R(\theta, p_\theta)$$
where $p_\theta(x) = \frac{1}{Z(\theta)} e^{-E_\theta(x)}$ and $E_\theta(x) = \sum_i \theta_i \phi_i(x_i) + \sum_{(i,j)\in\mathcal{E}} \theta_{ij}\phi_{ij}(x_i,x_j)$.
Gradient Derivation
Using the log-derivative trick and the energy model identity $\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}{p\theta}[\nabla_\theta E_\theta(x)]$:
$$\nabla_\theta \mathbb{E}{p\theta}[f(x)] = -\text{Cov}{p\theta}[f(x), \nabla_\theta E_\theta(x)]$$
Full gradient:
$$\nabla_\theta J = -(\mathbb{E}{p\theta}[f(x)] - \mu)^\top \text{Cov}{p\theta}[f(x), \nabla_\theta E_\theta(x)]$$
For the MRF, $\nabla_\theta E = f(x)$, so this becomes $-\text{Cov}{p\theta}[f,f] \cdot (\hat{g}-\mu)$.
Efficient Estimator (avoids forming $d \times p$ covariance matrix)
Given $N$ samples $x_i \sim p_\theta$:

```
g_hat = (1/N) * sum f(x_i)
delta = g_hat - mu
w_i   = -dot(delta, f(x_i) - g_hat)       # scalar per sample
grad_J = (1/N) * sum w_i * f(x_i)          # p-dimensional

```

Cost: $O(N(d+p))$ time, $O(d+p+N)$ memory. No covariance matrix formed.

---

# Implementation Plan

## Overview of Current State

`maxent_smm` is currently a verbatim copy of `maxent_large` with no algorithmic changes. Every
file imports from `calibrated_response.maxent_large.*` rather than its own local modules, and
the solver still runs the dual objective. The goal is to make `maxent_smm` a self-contained
module that trains on the SMM objective instead.

The dual objective (`maxent_large`) trains by gradient ascent on:

```
L(θ) = θ·μ - log Z(θ)
grad_L = μ - E_θ[f]   →   passed to optax as  targets - model_expectations
```

The SMM objective (`maxent_smm`) minimises instead:

```
J(θ) = (1/2) || E_θ[f] - μ ||²
grad_J = -Cov_θ[f(x), ∇_θ E(θ, x)] · (E_θ[f] - μ)   →   efficient estimator (see below)
```

Both objectives share identical MCMC, energy, feature, normalizer, and variable-spec
infrastructure.

### The general vs. linear case

In `maxent_large`, energy is always linear in θ: `E(θ, x) = θ·f(x) + prior(x)`, so
`∇_θ E(θ, x) = f(x)` trivially. The dual gradient thus collapses to
`-Cov[f, f] · delta`, which is what the roadmap's simplified efficient estimator assumes.

For a **general** parametrized energy model (the goal of `maxent_smm`), `∇_θ E(θ, x)` is a
separate quantity from the feature vector `f(x)` — it is the Jacobian of the energy with
respect to the parameters, evaluated at each sample point. The full efficient estimator is:

```
g_hat      = (1/N) * sum f(x_i)              # (p,) mean feature
delta      = g_hat - mu                       # (p,) moment residual
w_i        = -dot(delta, f(x_i) - g_hat)     # (1,) scalar weight, same as before
grad_theta_i = ∇_θ E(θ, x_i)                 # (K,) gradient of energy w.r.t. θ at x_i
grad_J     = (1/N) * sum w_i * grad_theta_i  # (K,) ← uses energy gradient, NOT f(x_i)
```

The weights `w_i` still depend only on the feature mismatch and are O(p) to compute.
The energy parameter gradients `∇_θ E(θ, x_i)` are the new ingredient.

In the linear MRF case `grad_theta_i = f(x_i)`, recovering the simple formula exactly.
For nonlinear models (e.g. a neural-network energy), `grad_theta_i` is the full parameter
Jacobian at each sample, computed via `jax.grad(energy_fn, argnums=0)`.

---

## Required Changes

### 1. Fix imports across all files (infrastructure prerequisite)

All files that currently import from `maxent_large` must be updated to import from
`maxent_smm` (or from local relative paths). Nothing else changes in these files — this is
pure path surgery required before any algorithmic change can take effect.

Files affected and the import lines to update:

- `maxent_solver.py` lines 22-27: imports `variable_spec`, `features`, `mcmc` from `maxent_large`
- `distribution_builder.py` lines 16-38: imports `variable_spec`, `features`, `maxent_solver`,
  `normalizer`, `energy_model` from `maxent_large`
- `energy_model.py` lines 20-26: imports `mcmc`, `normalizer` from `maxent_large`
- `__init__.py` lines 3-17: all imports from `maxent_large`

`estimate_r.py` and `variable_spec.py` only import from JAX / stdlib — no changes needed there.

---

### 2. Compile `_batch_grad_theta_fn` in `maxent_solver.py::build()` (new compiled function)

The existing `build()` at line 182 compiles two JAX functions:

```python
self._energy_fn         = jax.jit(_energy)                            # (θ, x) → scalar
self._grad_energy_fn    = jax.jit(jax.grad(_energy, argnums=1))      # (θ, x) → (D,) grad w.r.t. x
```

`_grad_energy_fn` is used by HMC (gradient w.r.t. position `x`). For SMM we need the
gradient w.r.t. the parameters `θ`, batched over all chain states. Add immediately after
the two existing lines:

```python
_grad_theta_fn = jax.grad(_energy, argnums=0)                         # (θ, x) → (K,) grad w.r.t. θ
self._batch_grad_theta_fn = jax.jit(
    jax.vmap(_grad_theta_fn, in_axes=(None, 0))                       # (θ, (C,D)) → (C, K)
)
```

`in_axes=(None, 0)` keeps θ fixed (not vmapped) and maps over the chain-state batch
dimension. Because θ is passed explicitly (not closed over), JAX traces the function once
and reuses the compiled kernel as θ changes each iteration — the same design pattern already
used for `_energy_fn` and `_grad_energy_fn`.

---

### 3. Replace the gradient computation in `maxent_solver.py::solve()` (core change)

In `solve()` at lines 283-287, Step C currently reads:

```python
grad = targets - model_expectations + cfg.l2_regularization * theta
```

Replace with the general efficient SMM estimator:

```python
# chain_features: (C, p) — feature evaluations at chain states, computed in step B
# model_expectations: (p,) = chain_features.mean(axis=0)

# Energy-parameter gradients at each chain state: (C, K)
chain_grad_theta = self._batch_grad_theta_fn(theta, buffer.states)

# SMM efficient estimator
delta    = model_expectations - targets               # (p,)  moment residual = E_θ[f] − μ
centered = chain_features - model_expectations        # (C, p) centred feature values
w        = -(centered @ delta)                        # (C,)  scalar weight per chain
grad     = (w[:, None] * chain_grad_theta).mean(axis=0)   # (K,)  ≈ ∇_θ J
grad     = grad + cfg.l2_regularization * theta       # L2 regularisation: ∇R = λθ
```

Note the key difference from the naive linear-MRF formula: the weighted sum is over
`chain_grad_theta` (shape `(C, K)`, the energy's parameter Jacobian at each sample), not
over `chain_features` (shape `(C, p)`, the moment-matching features). When the energy is
linear in θ, `chain_grad_theta == chain_features` and the two are identical. In the general
case they differ.

**Sign note.** `grad` here is $\nabla_\theta J$ (pointing uphill on J). Optax minimises, so
passing `grad` directly to `optimizer.update(grad, opt_state, theta)` implements gradient
descent on J — no extra negation is needed. This is the opposite convention from the dual
case, where `targets - model_expectations` was passed (which optax also minimised, because
minimising the dual residual is equivalent to maximising L).

**Clip.** The existing `jnp.clip(grad, -cfg.grad_clip, cfg.grad_clip)` line should be
retained after computing `grad`.

**Regularisation.** L2 regularisation adds $R(\theta) = \frac{\lambda}{2}||\theta||^2$,
whose gradient $\nabla R = \lambda\theta$ is added to the SMM gradient exactly as in the
current code.

**Roughness penalty.** The term `cfg.roughness_gamma * (self._R @ theta)` encodes a
smoothness prior on $\theta$ via a quadratic form in the feature Gram matrix. Its gradient
$\gamma R\theta$ adds directly to $\nabla_\theta J$ in the same way as L2 regularisation.
No sign change is needed — retain the existing block unchanged.

---

### 4. Update diagnostics in `solve()`

The per-iteration error `err = |targets - model_expectations|` (lines 296-298) and the
convergence check `max_err < cfg.tolerance` remain valid — they measure the moment-matching
residual, which is exactly what J minimises. No convergence-criterion changes are needed.

Optionally, add the SMM loss value to the history dict:

```python
# after computing delta above:
smm_loss = float(0.5 * jnp.sum(delta ** 2))
history["smm_loss"].append(smm_loss)   # initialise "smm_loss": [] in the history dict above
```

---

### 5. Update docstrings and config description

- Module docstring (lines 1-9): change "dual objective" / "gradient ascent" to "SMM
  objective" / "gradient descent on the moment-matching loss".
- `MaxEntSolver` class docstring (lines 76-84): same framing update.
- `JAXSolverConfig`: no field changes required. All existing hyperparameters
  (`learning_rate`, `l2_regularization`, `grad_clip`, `roughness_gamma`, etc.) carry over
  without modification.

---

## What Does NOT Change

- `mcmc.py` — HMC sampling is entirely objective-agnostic.
- `features.py` — Feature compilation is unchanged.
- `energy_model.py` — The trained model wrapper is unchanged beyond the import fix.
- `normalizer.py` — Unchanged.
- `variable_spec.py` — Unchanged.
- `estimate_r.py` — The roughness-matrix estimator is unchanged.
- `distribution_builder.py` — Unchanged beyond the import fix; the builder calls
  `solver.build()` and `solver.solve()` and reads `solver_info` identically.
- `__init__.py` — Only the import paths change, not the exported names.

---

## Summary of File-by-File Edits

| File | Change type | Description |
|---|---|---|
| `maxent_solver.py` | Imports + algorithm + docstring | Fix import paths; add `_batch_grad_theta_fn` in `build()`; replace Step C gradient; optionally add `smm_loss` to history; update docstrings |
| `distribution_builder.py` | Imports only | Point all imports at `maxent_smm` local modules |
| `energy_model.py` | Imports only | Point `mcmc` and `normalizer` imports at `maxent_smm` |
| `__init__.py` | Imports only | Point all imports at `maxent_smm` local modules |
| `features.py` | None | No changes |
| `mcmc.py` | None | No changes |
| `normalizer.py` | None | No changes |
| `variable_spec.py` | None | No changes |
| `estimate_r.py` | None | No changes |
