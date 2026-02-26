"""MLP energy model for MaxEnt-SMM training.

A single feedforward neural network maps the full state vector to a scalar energy:

    E(θ, x) = MLP(φ(x); θ)

where φ is a parameter-free Fourier feature encoding that lifts each scalar
variable from 1-D to (2K+1)-D, giving the network sensitivity at K frequency
scales without adding any trainable parameters.

The network itself is a standard ReLU MLP with a linear output layer.  It
implements the same interface as ``MarkovRandomField`` so it can be used as a
drop-in replacement in the SMM solver.
"""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class NeuralEnergyModel:
    """MLP energy model.

    Maps the full state vector ``x ∈ [0, 1]^d`` to a scalar energy through a
    feedforward neural network::

        h_0 = x
        h_i = ReLU(W_i h_{i-1} + b_i)   for each hidden layer i
        E   = w · h_last + c             (linear readout)

    Typical usage::

        model = NeuralEnergyModel(n_vars=10, hidden_sizes=[64, 32])

        # Initialise parameters
        params = model.init_params(jax.random.PRNGKey(0))
        theta  = model.pack_params(params)

        # Plug into the SMM solver
        solver.build(..., energy_fn=model.energy_fn_flat, init_theta=theta)
        theta, info = solver.solve()

        # Inspect learned parameters
        params = model.unpack_params(theta)
    """

    def __init__(
        self,
        n_vars: int,
        hidden_sizes: Union[int, list[int]] = 64,
    ) -> None:
        """Construct a neural energy model.

        Parameters
        ----------
        n_vars : int
            Dimensionality of the state vector x.
        hidden_sizes : int or list of int
            Width(s) of the hidden layers.  A single int creates one hidden
            layer.  For example ``hidden_sizes=[64, 32]`` gives a network with
            two hidden layers of widths 64 and 32.
        """
        self.n_vars: int = n_vars

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.hidden_sizes: list[int] = list(hidden_sizes)

        # Layer widths: input → h0 → h1 → ... → output (scalar readout).
        # The output "layer" is stored as {'W': (H_last,), 'b': ()}.
        widths = [n_vars] + self.hidden_sizes

        self._layer_shapes: list[tuple] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            self._layer_shapes.append((fan_out, fan_in))   # hidden W shapes
        # output readout: W is (H_last,), b is ()
        self._layer_shapes.append((self.hidden_sizes[-1],))

        # Build the zero pytree once to derive n_params and _unravel_fn.
        _zero = self.zero_params()
        _flat, self._unravel_fn = ravel_pytree(_zero)
        self.n_params: int = int(_flat.shape[0])

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def zero_params(self) -> list[dict]:
        """Return zero-initialised parameters.

        Returns a list of per-layer dicts:

        * Hidden layers — ``{'W': (H_out, H_in), 'b': (H_out,)}``.
        * Output layer  — ``{'W': (H_last,), 'b': ()}``.
        """
        widths = [self.n_vars] + self.hidden_sizes
        params = [
            {'W': jnp.zeros((fan_out, fan_in)), 'b': jnp.zeros(fan_out)}
            for fan_in, fan_out in zip(widths[:-1], widths[1:])
        ]
        # Linear output readout.
        params.append({'W': jnp.zeros(self.hidden_sizes[-1]), 'b': jnp.zeros(())})
        return params

    def init_params(self, key: jax.Array) -> list[dict]:
        """Return randomly initialised parameters.

        Hidden weights use He normal initialisation (scaled by ``1/sqrt(fan_in)``).
        Biases and output weights are initialised near zero.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        """
        widths = [self.n_vars] + self.hidden_sizes
        n_layers = len(widths)   # number of weight matrices (hidden + output)
        keys = jax.random.split(key, n_layers)

        params = [
            {
                'W': jax.random.normal(keys[i], (fan_out, fan_in)) / jnp.sqrt(fan_in),
                'b': jax.random.normal(keys[i], (fan_out,)) * 0.01,
            }
            for i, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:]))
        ]
        # Linear output readout.
        H_last = self.hidden_sizes[-1]
        params.append({
            'W': jax.random.normal(keys[-1], (H_last,)) * 0.01,
            'b': jnp.zeros(()),
        })
        return params

    def pack_params(self, params: list[dict]) -> jax.Array:
        """Flatten the parameter pytree into a single ``(n_params,)`` vector.

        Pure JAX — differentiable and JIT-compatible.
        """
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jax.Array) -> list[dict]:
        """Reconstruct the parameter pytree from a flat ``(n_params,)`` vector.

        Uses the inverse function captured at construction time.
        Pure JAX — differentiable and JIT-compatible.
        """
        return self._unravel_fn(theta)

    # ------------------------------------------------------------------ #
    # Energy functions
    # ------------------------------------------------------------------ #

    def energy_fn_pytree(
        self,
        params: list[dict],
        x: jax.Array,
    ) -> jax.Array:
        """Evaluate the energy from a structured parameter pytree.

        Parameters
        ----------
        params : list of dict
            Output of ``zero_params()`` or ``unpack_params()``.
        x : (n_vars,) jax.Array
            State vector in ``[0, 1]^d``.

        Returns
        -------
        Scalar energy value.
        """
        h = x
        for layer in params[:-1]:
            h = jnp.maximum(jnp.dot(layer['W'], h) + layer['b'], 0.0)
        output = params[-1]
        return jnp.dot(output['W'], h) + output['b']

    def energy_fn_flat(
        self,
        theta: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """Evaluate the energy from a flat ``(n_params,)`` parameter vector.

        This is the interface expected by ``MaxEntSolver.build(energy_fn=...)``.
        JAX can differentiate this function w.r.t. ``theta`` (``argnums=0``)
        for the SMM gradient and w.r.t. ``x`` (``argnums=1``) for HMC.

        Parameters
        ----------
        theta : (n_params,) jax.Array
            Flat parameter vector.
        x : (n_vars,) jax.Array
            State vector in ``[0, 1]^d``.

        Returns
        -------
        Scalar energy value.
        """
        return self.energy_fn_pytree(self.unpack_params(theta), x)
