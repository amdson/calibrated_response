"""Markov Random Field energy model for MaxEnt-SMM training.

The total energy is a sum of clique potentials:

    E(θ, x) = Σ_c  φ_c(θ_c, x[indices_c])

Each clique potential is a multilinear-interpolated lookup table over binned
variables, which makes the energy differentiable w.r.t. both x (needed by HMC)
and θ (needed by the SMM gradient estimator).
"""

from __future__ import annotations

import itertools
from typing import Union

import numpy as np
import jax.numpy as jnp

from calibrated_response.models.variable import Variable
from calibrated_response.models.query import (
    EstimateUnion,
    ConditionalProbabilityEstimate,
    ConditionalExpectationEstimate,
)


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


# ---------------------------------------------------------------------------
# MarkovRandomField class
# ---------------------------------------------------------------------------

class MarkovRandomField:
    """Markov Random Field energy model.

    The energy decomposes as a sum of clique potentials::

        E(θ, x) = Σ_c  φ_c(θ_c, x[indices_c])

    where each φ_c is a multilinear-interpolated lookup table (see
    ``clique_potential_interp``).

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
        """
        self.cliques: list[tuple[int, ...]] = [tuple(c) for c in cliques]
        self.bins_per_var: list[int] = list(bins_per_var)
        self.n_vars: int = n_vars

        # Shape of each clique's parameter tensor: (B_{c,1}, B_{c,2}, ...)
        self.clique_shapes: list[tuple[int, ...]] = [
            tuple(bins_per_var[i] for i in clique)
            for clique in self.cliques
        ]

        # Number of scalar parameters in each clique tensor.
        self.clique_sizes: list[int] = [
            int(np.prod(shape)) if shape else 1
            for shape in self.clique_shapes
        ]

        # Cumulative offsets into the flat parameter vector.
        self.offsets: list[int] = [0]
        for size in self.clique_sizes:
            self.offsets.append(self.offsets[-1] + size)

        # Total parameter count K.
        self.n_params: int = self.offsets[-1]

    @classmethod
    def from_estimates(
        cls,
        variables: list[Variable],
        estimates: list[EstimateUnion],
        bins_per_var: Union[int, list[int]] = 10,
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

        return cls(cliques=cliques, bins_per_var=bins_list, n_vars=n_vars)

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def zero_params(self) -> list[jnp.ndarray]:
        """Return a pytree of zero-initialised per-clique parameter tensors."""
        return [jnp.zeros(shape) for shape in self.clique_shapes]

    def pack_params(self, params: list[jnp.ndarray]) -> jnp.ndarray:
        """Flatten a pytree of per-clique tensors into a single (K,) vector.

        Pure JAX — differentiable and JIT-compatible.
        """
        return jnp.concatenate([p.ravel() for p in params])

    def unpack_params(self, theta: jnp.ndarray) -> list[jnp.ndarray]:
        """Reshape a flat (K,) vector into a pytree of per-clique tensors.

        Uses precomputed static offsets so JAX does not need to trace through
        dynamic indexing — the slices are resolved at compile time.

        Pure JAX — differentiable and JIT-compatible.
        """
        return [
            theta[self.offsets[c] : self.offsets[c + 1]].reshape(self.clique_shapes[c])
            for c in range(len(self.cliques))
        ]

    # ------------------------------------------------------------------ #
    # Energy functions
    # ------------------------------------------------------------------ #

    def energy_fn_pytree(
        self,
        params: list[jnp.ndarray],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the MRF energy from a pytree of per-clique tensors.

        Natural interface for inspection, visualisation, and manual
        initialisation.  Internally uses multilinear interpolation so the
        result is differentiable w.r.t. both ``params`` and ``x``.

        Parameters
        ----------
        params : list of jnp.ndarray
            ``params[c]`` has shape ``clique_shapes[c]``.
        x : (n_vars,) jnp.ndarray
            State vector in [0, 1]^d.

        Returns
        -------
        Scalar energy value.
        """
        energy = jnp.array(0.0)
        for c, clique in enumerate(self.cliques):
            energy = energy + clique_potential_interp(params[c], x, clique)
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
