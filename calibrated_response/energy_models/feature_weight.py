"""Feature-weight (classical MaxEnt) energy model for MaxEnt-SMM training.

The energy is a linear combination of pre-compiled feature functions:

    E(θ, x) = −θ · f(x)

where f = [f_1, …, f_K] is the vector of feature functions built from the
user's estimates (``FeatureSpec`` objects) and θ ∈ R^K are the learned weights.

This is the simplest parameterised energy model and the one MaxEnt is
mathematically designed around.  It implements the same interface as
``MarkovRandomField`` and ``NeuralEnergyModel`` so it can be used as a drop-in
replacement in the SMM solver.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from calibrated_response.maxent_smm.features import FeatureSpec, compile_feature_vector


class FeatureWeightModel:
    """Classical MaxEnt energy model: a dot product of weights and features.

    The energy is::

        E(θ, x) = −θ · f(x)

    where ``f`` is compiled from a list of ``FeatureSpec`` objects (the same
    specs that are passed to ``MaxEntSolver.build`` as constraints).  Zero-
    initialising ``θ`` starts training from the uniform distribution, which is
    the correct MaxEnt prior.

    Typical usage::

        # feature_specs built from estimates via DistributionBuilder
        model  = FeatureWeightModel(feature_specs)
        theta  = model.pack_params(model.zero_params())

        solver.build(
            ...,
            energy_fn=model.energy_fn_flat,
            init_theta=theta,
        )
        theta, info = solver.solve()
    """

    def __init__(self, feature_specs: Sequence[FeatureSpec]) -> None:
        """Construct a feature-weight energy model.

        Parameters
        ----------
        feature_specs : sequence of FeatureSpec
            The feature functions whose weighted sum defines the energy.
            These are the same specs passed to ``MaxEntSolver.build`` as
            ``feature_specs``.
        """
        self.feature_specs: list[FeatureSpec] = list(feature_specs)

        # Compile all specs into a single JIT-compatible f(x) → (K,) function.
        self._feature_fn = compile_feature_vector(self.feature_specs)

        # Derive n_params and _unravel_fn from the zero pytree, for interface
        # consistency with the other energy models.
        _zero = self.zero_params()
        _flat, self._unravel_fn = ravel_pytree(_zero)
        self.n_params: int = int(_flat.shape[0])

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def zero_params(self) -> jax.Array:
        """Return a zero-initialised weight vector of shape ``(K,)``.

        Zero initialisation corresponds to the uniform distribution, which is
        the correct starting point for MaxEnt training.
        """
        return jnp.zeros(len(self.feature_specs))

    def init_params(self, key: jax.Array) -> jax.Array:
        """Return a randomly initialised weight vector (small Gaussian noise).

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        """
        return jax.random.normal(key, (len(self.feature_specs),)) * 0.01

    def pack_params(self, params: jax.Array) -> jax.Array:
        """Flatten the parameter pytree into a single ``(K,)`` vector.

        For this model the params *are* a flat vector, so this is a no-op
        beyond the ``ravel_pytree`` call used for interface consistency.
        """
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jax.Array) -> jax.Array:
        """Reconstruct the parameter pytree from a flat ``(K,)`` vector.

        For this model the result is the vector itself.
        """
        return self._unravel_fn(theta)

    # ------------------------------------------------------------------ #
    # Energy functions
    # ------------------------------------------------------------------ #

    def energy_fn_pytree(
        self,
        params: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """Evaluate the energy from the weight vector.

        Parameters
        ----------
        params : (K,) jax.Array
            Weight vector (output of ``zero_params()`` or ``unpack_params()``).
        x : (n_vars,) jax.Array
            State vector in ``[0, 1]^d``.

        Returns
        -------
        Scalar energy value ``−θ · f(x)``.
        """
        return -jnp.dot(params, self._feature_fn(x))

    def energy_fn_flat(
        self,
        theta: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """Evaluate the energy from a flat ``(K,)`` parameter vector.

        This is the interface expected by ``MaxEntSolver.build(energy_fn=...)``.
        JAX can differentiate this function w.r.t. ``theta`` (``argnums=0``)
        for the SMM gradient and w.r.t. ``x`` (``argnums=1``) for HMC.

        Parameters
        ----------
        theta : (K,) jax.Array
            Flat weight vector.
        x : (n_vars,) jax.Array
            State vector in ``[0, 1]^d``.

        Returns
        -------
        Scalar energy value ``−θ · f(x)``.
        """
        return self.energy_fn_pytree(self.unpack_params(theta), x)
