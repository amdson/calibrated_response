"""Markov Random Field energy model for MaxEnt-SMM training.

The total energy is a sum of clique potentials:

    E(θ, x) = Σ_c  φ_c(θ_c, x[indices_c])

Each clique potential can be one of three types:

* ``'table'`` — multilinear-interpolated lookup table (default).
* ``'rbf'``   — weighted sum of radial basis functions.
* ``'nn'``    — small two-layer feedforward neural network (MLP).

All three are differentiable w.r.t. both x (needed by HMC) and θ (needed by
the SMM gradient estimator).
"""

from __future__ import annotations

import itertools
from typing import Union

import jax
import jax
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from calibrated_response.models.variable import Variable
from calibrated_response.models.query import (
    EstimateUnion,
    ConditionalProbabilityEstimate,
    ConditionalExpectationEstimate,
)


# ---------------------------------------------------------------------------
# Module-level defaults for non-table clique types
# ---------------------------------------------------------------------------

_DEFAULT_N_CENTERS: int = 20   # default number of RBF centres per clique
_DEFAULT_HIDDEN_SIZE: int = 16  # default hidden-layer width for NN cliques

# ---------------------------------------------------------------------------
# Clique potentials
# ---------------------------------------------------------------------------

def clique_potential(
    binned_distr: jnp.ndarray,
    x: jnp.ndarray,
    factor_indices,
) -> jnp.ndarray:
    """Nearest-bin clique potential lookup (not differentiable w.r.t. x).

    Parameters
    ----------
    binned_distr : (B1, B2, ...) jnp.ndarray
        Energy lookup table for this clique.
    x : (d,) jnp.ndarray
        Full state vector in [0, 1)^d.
    factor_indices : int sequence of length F
        Indices of the variables that belong to this clique.

    Returns
    -------
    Scalar energy contribution from this clique.
    """
    factor_x = x[np.array(factor_indices)]  # (F,)
    shape = np.array(binned_distr.shape, dtype=float)  # (F,) — static
    bin_idx = jnp.minimum(
        jnp.floor(factor_x * shape).astype(int),
        jnp.array(binned_distr.shape) - 1,
    )  # (F,) bin indices, clamped to valid range
    return binned_distr[tuple(bin_idx)]


def clique_potential_interp(
    binned_distr: jnp.ndarray,
    x: jnp.ndarray,
    factor_indices,
) -> jnp.ndarray:
    """Multilinear-interpolated clique potential (differentiable w.r.t. x).

    For F variables each discretised into B_i bins, computes a weighted average
    over the 2^F corners of the hypercube surrounding x in the bin grid.  This
    makes the potential continuously differentiable w.r.t. x everywhere, which
    is required by HMC's leapfrog integrator.

    The potential is also differentiable w.r.t. the table values (θ_c), so
    jax.grad(energy_fn, argnums=0) works for the SMM gradient.

    Parameters
    ----------
    binned_distr : (B1, B2, ...) jnp.ndarray
        Energy lookup table for this clique.
    x : (d,) jnp.ndarray
        Full state vector in [0, 1]^d.
    factor_indices : int sequence of length F
        Indices of the variables that belong to this clique.

    Returns
    -------
    Scalar energy contribution from this clique.
    """
    factor_x = x[np.array(factor_indices)]  # (F,)
    shape = binned_distr.shape               # (B1, B2, ...) — static Python tuple
    F = len(shape)

    # Float bin coordinates, clipped so they stay within the table extent.
    t = jnp.clip(factor_x, 0.0, 1.0) * jnp.array(shape, dtype=float)  # (F,)

    # Lower corner bin index (floor, clamped to [0, B_i - 1]).
    lo = jnp.minimum(
        jnp.floor(t).astype(int), jnp.array(shape, dtype=int) - 1
    )  # (F,)

    # Upper corner bin index (lo + 1, clamped to [0, B_i - 1]).
    hi = jnp.minimum(lo + 1, jnp.array(shape, dtype=int) - 1)  # (F,)

    # Fractional weight for the upper corner along each axis, in [0, 1].
    w = t - lo.astype(float)  # (F,)

    # Accumulate the weighted sum over all 2^F corners of the hypercube.
    # The Python loop is over a static range — JAX traces through it once.
    total = jnp.array(0.0)
    for corner in range(1 << F):
        # Bit i of `corner` selects lo[i] (0) or hi[i] (1) for dimension i.
        idx = tuple(hi[i] if (corner >> i) & 1 else lo[i] for i in range(F))

        # Weight for this corner: product of per-axis weights.
        corner_w = jnp.prod(
            jnp.stack([w[i] if (corner >> i) & 1 else (1.0 - w[i]) for i in range(F)])
        )

        total = total + corner_w * binned_distr[idx]

    return total

def clique_potential_rbf(
    centers: jnp.ndarray,
    weights: jnp.ndarray,
    x: jnp.ndarray,
    factor_indices,
) -> jnp.ndarray:
    """RBF-based clique potential (differentiable w.r.t. x).

    Represents the potential as a weighted sum of RBFs centred at fixed points
    in the variable space.  This is an alternative to multilinear interpolation
    that can be more compact for high-dimensional cliques, at the cost of
    potentially less accurate representation of sharp features.

    Parameters
    ----------
    centers : (M, F) jnp.ndarray
        RBF centre locations for this clique, where F is the number of variables
        in the clique and M is the number of RBFs.
    weights : (M,) jnp.ndarray
        Weights for each RBF.
    x : (d,) jnp.ndarray
        Full state vector in [0, 1]^d.
    factor_indices : int sequence of length F
        Indices of the variables that belong to this clique.

    Returns
    -------
    Scalar energy contribution from this clique.
    """
    factor_x = x[np.array(factor_indices)]  # (F,)
    dists = jnp.linalg.norm(centers - factor_x, axis=1)  # (M,)
    rbf_values = jnp.exp(-0.5 * (dists / 5.0)**2)  # (M,) using a fixed bandwidth of 5.0
    return jnp.dot(weights, rbf_values)  # Scalar energy contribution

def clique_nn_potential(
    params: dict,
    x: jnp.ndarray,
    factor_indices,
) -> jnp.ndarray:
    """NN-based clique potential (differentiable w.r.t. x).

    Represents the potential as a small feedforward neural network that takes
    the clique variables as input.  This is a flexible representation that can
    capture complex interactions, but it may be less interpretable and require
    more parameters than the other options.

    Parameters
    ----------
    params : dict
        Dictionary of NN parameters for this clique, e.g. {'W1': ..., 'b1': ..., 'W2': ..., 'b2': ...}.
    x : (d,) jnp.ndarray
        Full state vector in [0, 1]^d.
    factor_indices : int sequence of length F
        Indices of the variables that belong to this clique.

    Returns
    -------
    Scalar energy contribution from this clique.
    """
    factor_x = x[np.array(factor_indices)]  # (F,)
    
    # Example 2-layer MLP with ReLU activation
    hidden = jnp.dot(params['W1'], factor_x) + params['b1']  # (H,)
    hidden = jnp.maximum(hidden, 0)  # ReLU activation
    output = jnp.dot(params['W2'], hidden) + params['b2']  # Scalar energy contribution
    
    return output


# ---------------------------------------------------------------------------
# MarkovRandomField class
# ---------------------------------------------------------------------------

class MarkovRandomField:
    """Markov Random Field energy model.

    The energy decomposes as a sum of clique potentials::

        E(θ, x) = Σ_c  φ_c(θ_c, x[indices_c])

    where each φ_c is a multilinear-interpolated lookup table (see
    ``clique_potential_interp``), an RBF-based potential (see ``clique_potential_rbf``),
    or a neural network-based potential (see ``clique_nn_potential``).

    Typical usage::

        # Build from an existing constraint graph
        mrf = MarkovRandomField.from_estimates(variables, estimates, bins_per_var=10)

        # Plug into the SMM solver
        solver.build(..., energy_fn=mrf.energy_fn_flat)
        theta, info = solver.solve()

        # Inspect per-clique tables
        params = mrf.unpack_params(theta)
    """

    def __init__(
        self,
        cliques: list[tuple[int, ...]],
        bins_per_var: list[int],
        n_vars: int,
        clique_types: Union[str, list[str], None] = None,
        clique_configs: Union[list[dict], None] = None,
    ) -> None:
        """Construct a MRF from an explicit clique specification.

        Parameters
        ----------
        cliques : list of tuples of int
            Variable-index sets for each clique, e.g. ``[(0,), (1,), (0, 1)]``.
            Indices must be in ``[0, n_vars)``.
        bins_per_var : list of int, length n_vars
            B_i — number of bins for variable i.
        n_vars : int
            Total number of variables d.
        clique_types : str or list of str, optional
            Potential type for each clique: ``'table'`` (default),
            ``'rbf'``, or ``'nn'``.  A single string applies to all cliques.
        clique_configs : list of dict, optional
            Per-clique configuration.  Recognised keys:

            * ``'table'`` cliques — no extra keys (bins come from ``bins_per_var``).
            * ``'rbf'`` cliques   — ``n_centers`` (int, default
              ``_DEFAULT_N_CENTERS``).
            * ``'nn'`` cliques    — ``hidden_size`` (int, default
              ``_DEFAULT_HIDDEN_SIZE``).
        """
        self.cliques: list[tuple[int, ...]] = [tuple(c) for c in cliques]
        self.bins_per_var: list[int] = list(bins_per_var)
        self.n_vars: int = n_vars
        n_cliques = len(self.cliques)

        # Normalise clique_types.
        if clique_types is None:
            self.clique_types: list[str] = ['table'] * n_cliques
        elif isinstance(clique_types, str):
            self.clique_types = [clique_types] * n_cliques
        else:
            self.clique_types = list(clique_types)

        # Normalise clique_configs.
        if clique_configs is None:
            self.clique_configs: list[dict] = [{} for _ in range(n_cliques)]
        else:
            self.clique_configs = [c if c is not None else {} for c in clique_configs]

        # Validate types eagerly so the error is raised at construction time.
        for ctype in self.clique_types:
            if ctype not in ('table', 'rbf', 'nn'):
                raise ValueError(
                    f"Unknown clique type {ctype!r}. "
                    "Expected 'table', 'rbf', or 'nn'."
                )

        # Build the zero pytree once; ravel_pytree derives n_params and the
        # inverse function (_unravel_fn) from it.  All packing/unpacking is
        # then delegated to JAX's pytree machinery.
        _zero = self.zero_params()
        _flat, self._unravel_fn = ravel_pytree(_zero)
        self.n_params: int = int(_flat.shape[0])

    @classmethod
    def from_estimates(
        cls,
        variables: list[Variable],
        estimates: list[EstimateUnion],
        bins_per_var: Union[int, list[int]] = 10,
        clique_type: str = 'table',
        clique_config: Union[dict, None] = None,
    ) -> "MarkovRandomField":
        """Build an MRF whose clique structure mirrors the constraint graph.

        Adds:
        - A unary clique ``(i,)`` for every variable i.
        - A pairwise clique ``(i, j)`` for each pair of variables that
          co-occur in a conditional estimate (``ConditionalProbabilityEstimate``
          or ``ConditionalExpectationEstimate``).  Each unordered pair appears
          at most once.

        Parameters
        ----------
        variables : list of Variable
            All variables in the model, in index order.
        estimates : list of EstimateUnion
            Constraints from which the dependency graph is inferred.
        bins_per_var : int or list of int
            Uniform bin count, or per-variable bin counts.
        clique_type : str
            Potential type applied to every clique: ``'table'`` (default),
            ``'rbf'``, or ``'nn'``.
        clique_config : dict, optional
            Configuration dict forwarded to every clique (see ``__init__``
            for recognised keys per type).

        Returns
        -------
        MarkovRandomField
        """
        n_vars = len(variables)
        name_to_idx = {v.name: i for i, v in enumerate(variables)}

        if isinstance(bins_per_var, int):
            bins_list = [bins_per_var] * n_vars
        else:
            bins_list = list(bins_per_var)

        # Start with unary cliques for all variables.
        cliques: list[tuple[int, ...]] = [(i,) for i in range(n_vars)]
        clique_set: set[tuple[int, ...]] = set(cliques)

        for est in estimates:
            # Collect variable names mentioned in this estimate.
            var_names: list[str] = []
            if isinstance(est, ConditionalProbabilityEstimate):
                var_names.append(est.proposition.variable)
                for cond in est.conditions:
                    var_names.append(cond.variable)
            elif isinstance(est, ConditionalExpectationEstimate):
                var_names.append(est.variable)
                for cond in est.conditions:
                    var_names.append(cond.variable)

            # Map to indices; drop any variable not in the model.
            indices = sorted({name_to_idx[n] for n in var_names if n in name_to_idx})

            # Add pairwise cliques for all pairs that co-occur in this estimate.
            for a, b in itertools.combinations(indices, 2):
                clique = (a, b)
                if clique not in clique_set:
                    cliques.append(clique)
                    clique_set.add(clique)

        n_cliques = len(cliques)
        return cls(
            cliques=cliques,
            bins_per_var=bins_list,
            n_vars=n_vars,
            clique_types=[clique_type] * n_cliques,
            clique_configs=[clique_config] * n_cliques,
        )

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def zero_params(self) -> list:
        """Return a pytree of zero-initialised per-clique parameters.

        The structure mirrors what ``unpack_params`` produces:

        * ``'table'`` cliques — ``jnp.ndarray`` of shape ``(B1, ..., BF)``.
        * ``'rbf'`` cliques   — ``{'centers': (M, F), 'weights': (M,)}``.
        * ``'nn'`` cliques    — ``{'W1': (H, F), 'b1': (H,), 'W2': (H,), 'b2': ()}``.
        """
        result = []
        for clique, ctype, cfg in zip(self.cliques, self.clique_types, self.clique_configs):
            F = len(clique)
            if ctype == 'table':
                shape = tuple(self.bins_per_var[i] for i in clique)
                result.append(jnp.zeros(shape if shape else ()))
            elif ctype == 'rbf':
                M = cfg.get('n_centers', _DEFAULT_N_CENTERS)
                result.append({
                    'centers': jnp.zeros((M, F)),
                    'weights': jnp.zeros(M),
                })
            elif ctype == 'nn':
                H = cfg.get('hidden_size', _DEFAULT_HIDDEN_SIZE)
                result.append({
                    'W1': jnp.zeros((H, F)),
                    'b1': jnp.zeros(H),
                    'W2': jnp.zeros(H),
                    'b2': jnp.zeros(()),
                })
        return result
    
    def init_params(self, key: jnp.ndarray) -> list:
        """Return a pytree of randomly initialised per-clique parameters.

        Uses the same structure as ``zero_params()``, but with random values
        drawn from a standard normal distribution.  The random key is split
        for each clique to ensure different random values.

        Parameters
        ----------
        key : jnp.ndarray
            JAX PRNG key for random number generation.

        Returns
        -------
        list
            Pytree of per-clique parameters with the same structure as
            ``zero_params()``, but with random initial values.
        """
        result = []
        keys = jax.random.split(key, len(self.cliques))
        for clique, ctype, cfg, k in zip(self.cliques, self.clique_types, self.clique_configs, keys):
            F = len(clique)
            if ctype == 'table':
                shape = tuple(self.bins_per_var[i] for i in clique)
                result.append(jax.random.normal(k, shape if shape else ()))
            elif ctype == 'rbf':
                M = cfg.get('n_centers', _DEFAULT_N_CENTERS)
                result.append({
                    'centers': jax.random.uniform(k, (M, F)),
                    'weights': jax.random.normal(k, (M,)) / jnp.sqrt(M),  # He initialization
                })
            elif ctype == 'nn':
                H = cfg.get('hidden_size', _DEFAULT_HIDDEN_SIZE)
                result.append({
                    'W1': jax.random.normal(k, (H, F)) / jnp.sqrt(F),  # He initialization
                    'b1': jax.random.normal(k, (H,)) * 0.01,  # Small bias initialization
                    'W2': jax.random.normal(k, (H,)) / jnp.sqrt(H),  # He initialization
                    'b2': jax.random.normal(k, ()) * 0.01,  # Small bias initialization
                })
        return result

    def pack_params(self, params: list) -> jnp.ndarray:
        """Flatten a pytree of per-clique params into a single (K,) vector.

        Uses ``jax.flatten_util.ravel_pytree`` — pure JAX, differentiable and
        JIT-compatible.
        """
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jnp.ndarray) -> list:
        """Reconstruct the per-clique pytree from a flat (K,) vector.

        Uses the inverse function captured at construction time from
        ``ravel_pytree(zero_params())`` — pure JAX, differentiable and
        JIT-compatible.
        """
        return self._unravel_fn(theta)
    
    def energy_fn_pytree(
        self,
        params: list,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the MRF energy from a structured per-clique pytree.

        Natural interface for inspection, visualisation, and manual
        initialisation.  Differentiable w.r.t. both ``params`` and ``x``.

        Parameters
        ----------
        params : list
            Output of ``zero_params()`` or ``unpack_params()``.  Each element
            is a ``jnp.ndarray`` (table) or a parameter dict (rbf / nn).
        x : (n_vars,) jnp.ndarray
            State vector in [0, 1]^d.

        Returns
        -------
        Scalar energy value.
        """
        energy = jnp.array(0.0)
        for c, clique in enumerate(self.cliques):
            ctype = self.clique_types[c]

            if ctype == 'table':
                energy = energy + clique_potential_interp(params[c], x, clique)

            elif ctype == 'rbf':
                p = params[c]
                energy = energy + clique_potential_rbf(
                    p['centers'], p['weights'], x, clique
                )
            elif ctype == 'nn':
                energy = energy + clique_nn_potential(params[c], x, clique)

        return energy

    def energy_fn_flat(
        self,
        theta: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the MRF energy from a flat (K,) parameter vector.

        This is the interface expected by ``MaxEntSolver.build(energy_fn=...)``.
        JAX can differentiate this function w.r.t. ``theta`` (``argnums=0``)
        for the SMM gradient and w.r.t. ``x`` (``argnums=1``) for HMC.

        Parameters
        ----------
        theta : (n_params,) jnp.ndarray
            All clique parameters concatenated in clique order, each clique's
            tensor flattened in C (row-major) order.
        x : (n_vars,) jnp.ndarray
            State vector in [0, 1]^d.

        Returns
        -------
        Scalar energy value.
        """
        params = self.unpack_params(theta)
        return self.energy_fn_pytree(params, x)
