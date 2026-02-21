# Markov Random Field Energy Model

## Overview

This module implements an MRF energy model whose total energy is a sum of
clique potentials:

```
E(θ, x) = Σ_c  φ_c(θ_c, x[indices_c])
```

Each clique `c` is a small subset of variables. Its potential `φ_c` is a
lookup table: the variables are discretised into bins and the energy is read
off a multi-dimensional tensor indexed by the bin coordinates. This is the
`clique_potential` function already in `markov_random.py`.

For a clique over `F` variables each with `B` bins, the parameter tensor
`θ_c` has shape `(B, B, ..., B)` (`F` axes, `B^F` entries). Cliques should
therefore be kept small (unary or pairwise) to avoid exponential growth.

---

## Clique Potential (existing)

```python
clique_potential(binned_distr, x, factor_indices) -> scalar
```

- `binned_distr`: `(B1, B2, ...)` — energy table for one clique.
- `x`: `(d,)` — full state vector in `[0, 1)^d`.
- `factor_indices`: `(F,)` — which variables belong to this clique.

Maps each variable to its bin index via `floor(x[i] * B_i)`, then indexes
into the tensor. The TODO in the file notes a future interpolated version;
for now the lookup is nearest-bin.

---

## `MarkovRandomField` Class Design

### Direct initialisation

The primary constructor takes a fully explicit specification:

```python
MRF = MarkovRandomField(
    cliques         : list[tuple[int, ...]],   # variable index sets, e.g. [(0,), (1,), (0,1)]
    bins_per_var    : list[int],               # B_i for each of the d variables
    n_vars          : int,                     # total number of variables d
)
```

From this the class can compute, at construction time:

- The shape of each clique's parameter tensor: `tuple(bins_per_var[i] for i in clique)`.
- The total parameter count `K = Σ_c prod(shape_c)`.
- The flat-vector slice for each clique: `offsets[c]` to `offsets[c+1]`.

These are stored as Python-level constants so JAX can trace through any
function that uses them without recompilation.

### Class method: `from_estimates`

```python
@classmethod
MarkovRandomField.from_estimates(
    variables  : list[Variable],
    estimates  : list[EstimateUnion],
    bins_per_var : int | list[int] = 10,
) -> MarkovRandomField
```

Infers the clique structure from the constraint graph:

1. Add a unary clique `(i,)` for every variable `i`.
2. For every estimate that mentions two or more variables together (i.e.
   `ConditionalProbabilityEstimate`, `ConditionalExpectationEstimate`), add a
   pairwise clique `(i, j)` for each pair of variables that co-occur in that
   estimate. Deduplicate so each unordered pair appears at most once.

This mirrors the structure already visualised by
`visualization/factor_graph.py` — the cliques are exactly the edges of the
variable-interaction graph. Variables that the estimates treat as independent
get no pairwise clique and therefore no learned dependency.

---

## Exported Energy Functions

The class exports two energy functions with different parameter
representations.

### 1. Pytree interface — `energy_fn_pytree`

```python
energy_fn_pytree(params: list[jnp.ndarray], x: jnp.ndarray) -> scalar
```

`params` is a list of per-clique tensors, one per clique in order:
`params[c]` has shape `(B_{c,1}, B_{c,2}, ...)`. This is the natural
representation for inspection, visualisation, and manual initialisation
(e.g. setting a pairwise clique's table to zero initially).

Internally calls `clique_potential` for each clique and sums the results.

### 2. Flat-vector interface — `energy_fn_flat`

```python
energy_fn_flat(theta: jnp.ndarray, x: jnp.ndarray) -> scalar
```

`theta` is a single `(K,)` array — all clique parameters concatenated in
clique order, with each clique's tensor flattened in C (row-major) order.
Internally slices `theta` using the precomputed offsets, reshapes each slice
into its clique tensor shape, and delegates to `energy_fn_pytree`.

This is the interface required by `MaxEntSolver.build()` in `maxent_smm`:
the solver assumes `energy_fn(theta, x)` with a flat `theta` so that
`jax.grad(energy_fn, argnums=0)` returns a flat gradient vector of the same
shape.

### Conversion helpers

```python
pack_params(params: list[jnp.ndarray]) -> jnp.ndarray     # pytree  → flat (K,)
unpack_params(theta: jnp.ndarray)      -> list[jnp.ndarray]  # flat (K,) → pytree
```

Both are pure JAX operations so they are differentiable and JIT-compatible.
`unpack_params` uses the stored offsets and shapes; because these are
compile-time constants the slicing is static and JAX does not need to trace
through dynamic indexing.

---

## Parameter Count and Scaling

| Clique type | Variables | Bins per var | Parameters |
|-------------|-----------|--------------|------------|
| Unary       | 1         | B            | B          |
| Pairwise    | 2         | B            | B²         |
| Triple      | 3         | B            | B³         |

For `d` variables, `B = 10` bins, and a fully pairwise graph:

```
K = d * B  +  d*(d-1)/2 * B²
```

With `d = 20`, `B = 10`: `K = 200 + 190 * 100 = 19 200` parameters — well
within the flat-vector regime. Higher-order cliques should be used sparingly.

---

## Integration with `maxent_smm`

```python
mrf = MarkovRandomField.from_estimates(variables, estimates, bins_per_var=10)

solver = MaxEntSolver(config)
solver.build(
    var_specs       = var_specs,
    feature_specs   = feature_specs,
    feature_targets = targets,
    energy_fn       = mrf.energy_fn_flat,   # replaces the default linear energy
)
theta, info = solver.solve()

# Recover per-clique tables for inspection
params = mrf.unpack_params(theta)
```

The solver's `_batch_grad_theta_fn = jit(vmap(grad(energy_fn, argnums=0)))`
then computes gradients of the MRF energy w.r.t. the flat parameter vector,
which is what the SMM estimator needs.

---

## Future: Interpolated Potential (TODO in stub)

The current nearest-bin lookup is not differentiable w.r.t. `x` (used by
HMC), but the energy value itself is differentiable w.r.t. `theta` (used by
SMM), so training works. HMC still needs `∇_x E`; the nearest-bin version
gives a zero gradient almost everywhere, which is problematic.

The planned fix (noted in the stub) is multilinear interpolation: compute a
weighted average of the `2^F` corner values surrounding `x` in the bin grid.
This makes `φ_c` differentiable w.r.t. both `x` and `θ_c` everywhere.
