"""MLP sampler model for MaxEnt-by-moment-matching.

A single feedforward network maps a Gaussian latent vector to a state vector::

    x = g_θ(z),   z ~ N(0, I) ∈ R^latent_dim,   x ∈ (0, 1)^n_vars

The output layer is passed through a sigmoid so samples land in ``(0, 1)^d``,
matching the domain the feature functions (``MomentFeature``, threshold
surrogates, …) expect.

The interface mirrors ``NeuralEnergyModel``: ``init_params`` / ``pack_params`` /
``unpack_params`` and a flat-parameter forward pass ``sample_fn_flat(theta, z)``
that ``SamplerSolver.build(sampler_fn=...)`` expects.  Because the whole map is
differentiable in ``θ``, ``jax.grad`` flows through it directly (the
reparametrisation trick).
"""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class NeuralSampler:
    """MLP sampler ``g_θ : z ↦ x`` with a sigmoid output onto ``(0, 1)^d``.

    ::

        h_0   = z
        h_i   = ReLU(W_i h_{i-1} + b_i)          for each hidden layer i
        x     = sigmoid(W_out h_last + b_out)     (maps to (0, 1)^n_vars)

    Typical usage::

        sampler = NeuralSampler(n_vars=3, latent_dim=8, hidden_sizes=[64, 64])
        theta   = sampler.pack_params(sampler.init_params(jax.random.PRNGKey(0)))

        solver.build(..., sampler_fn=sampler.sample_fn_flat,
                     init_theta=theta, latent_dim=sampler.latent_dim)
        theta, info = solver.solve()

        # draw samples from the trained model
        z = jax.random.normal(key, (1000, sampler.latent_dim))
        x = jax.vmap(sampler.sample_fn_flat, in_axes=(None, 0))(theta, z)
    """

    def __init__(
        self,
        n_vars: int,
        latent_dim: int,
        hidden_sizes: Union[int, list[int]] = 64,
        out_scale: float = 1.0,
    ) -> None:
        """Construct a neural sampler.

        Parameters
        ----------
        n_vars : int
            Dimensionality of the output state vector x.
        latent_dim : int
            Dimensionality of the Gaussian latent z.  A larger latent gives the
            sampler more stochastic degrees of freedom (helps it represent
            higher-entropy / multi-modal targets).
        hidden_sizes : int or list of int
            Width(s) of the hidden layers.  A single int creates one hidden
            layer; ``[64, 32]`` gives two hidden layers of widths 64 and 32.
        out_scale : float
            Pre-sigmoid standard deviation of the output layer at initialisation
            (in units of ``1/sqrt(fan_in)``).  ``0`` reproduces the old near-flat
            ``x ≈ 0.5`` start; the default ``1.0`` gives the sampler real spread
            across ``(0, 1)`` so threshold/indicator features are not saturated
            (dead-gradient) before training moves any mass.
        """
        self.n_vars: int = n_vars
        self.latent_dim: int = latent_dim
        self.out_scale: float = float(out_scale)

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.hidden_sizes: list[int] = list(hidden_sizes)

        # Layer widths: latent_dim → h0 → ... → h_last → n_vars (readout).
        self._widths: list[int] = (
            [self.latent_dim] + self.hidden_sizes + [self.n_vars]
        )

        # Build the zero pytree once to derive n_params and _unravel_fn.
        _zero = self.zero_params()
        _flat, self._unravel_fn = ravel_pytree(_zero)
        self.n_params: int = int(_flat.shape[0])

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def zero_params(self) -> list[dict]:
        """Return zero-initialised parameters (list of ``{'W', 'b'}`` dicts)."""
        return [
            {'W': jnp.zeros((fan_out, fan_in)), 'b': jnp.zeros(fan_out)}
            for fan_in, fan_out in zip(self._widths[:-1], self._widths[1:])
        ]

    def init_params(self, key: jax.Array) -> list[dict]:
        """Return randomly initialised parameters.

        Hidden weights use He-style ``1/sqrt(fan_in)`` scaling.  The output layer
        is scaled by ``out_scale/sqrt(fan_in)`` so that, with ``out_scale=1`` (the
        default), the pre-sigmoid output has ~unit spread and the initial sampler
        already covers ``(0, 1)`` — avoiding the dead-gradient start where all
        samples sit at ``x ≈ 0.5`` and threshold features are saturated.
        """
        keys = jax.random.split(key, len(self._widths) - 1)
        layers = []
        n_layers = len(self._widths) - 1
        for i, (fan_in, fan_out) in enumerate(
            zip(self._widths[:-1], self._widths[1:])
        ):
            is_output = i == n_layers - 1
            scale = (self.out_scale if is_output else 1.0) / jnp.sqrt(fan_in)
            layers.append({
                'W': jax.random.normal(keys[i], (fan_out, fan_in)) * scale,
                'b': jnp.zeros(fan_out),
            })
        return layers

    def pack_params(self, params: list[dict]) -> jax.Array:
        """Flatten the parameter pytree into a single ``(n_params,)`` vector."""
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jax.Array) -> list[dict]:
        """Reconstruct the parameter pytree from a flat ``(n_params,)`` vector."""
        return self._unravel_fn(theta)

    # ------------------------------------------------------------------ #
    # Forward pass  z -> x
    # ------------------------------------------------------------------ #

    def sample_fn_pytree(self, params: list[dict], z: jax.Array) -> jax.Array:
        """Map a single latent ``z`` (latent_dim,) to a state ``x`` (n_vars,)."""
        h = z
        for layer in params[:-1]:
            h = jnp.maximum(jnp.dot(layer['W'], h) + layer['b'], 0.0)
        out = params[-1]
        return jax.nn.sigmoid(jnp.dot(out['W'], h) + out['b'])

    def sample_fn_flat(self, theta: jax.Array, z: jax.Array) -> jax.Array:
        """Map ``z`` to ``x`` from a flat parameter vector.

        This is the interface expected by ``SamplerSolver.build(sampler_fn=...)``.
        Differentiable in ``theta`` (argnums=0) for the moment-matching gradient.
        """
        return self.sample_fn_pytree(self.unpack_params(theta), z)


# ---------------------------------------------------------------------------
# Optional entropy proxy for regularisation
# ---------------------------------------------------------------------------

def neg_gaussian_entropy_proxy(theta, z, x: jax.Array) -> jax.Array:
    """Negative Gaussian entropy proxy for a batch of samples ``x`` (N, D).

    Estimates ``-Σ_d 0.5 * log(2πe · Var[x_d])`` — the negative differential
    entropy of a Gaussian with the batch's per-dimension variance.  Adding this
    to the loss with a **positive** weight *encourages* spread (larger variance
    → lower loss), a cheap stand-in for the maximum-entropy objective that the
    energy-model formulation gets for free.

    This is only a proxy: it assumes per-dimension Gaussianity and ignores
    cross-dimension structure.  Use as a regulariser, not a calibrated entropy.

    Signature matches the ``reg_fn(theta, z, x)`` hook of ``SamplerSolver``.
    """
    var = jnp.var(x, axis=0) + 1e-8                    # (D,)
    entropy = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * var))
    return -entropy


def _soft_bin_assign(x: jax.Array, lower, span, n_bins: int) -> jax.Array:
    """Soft one-hot bin assignments ``(N, D, n_bins)`` over equal cells on the
    domain (Gaussian kernel, bandwidth = one bin width, so gradients flow)."""
    s = jnp.clip((x - lower) / span, 0.0, 1.0)
    centers = (jnp.arange(n_bins) + 0.5) / n_bins
    a = jnp.exp(-0.5 * ((s[:, :, None] - centers) * n_bins) ** 2)
    return a / (jnp.sum(a, axis=-1, keepdims=True) + 1e-12)


def neg_marginal_entropy_proxy(x: jax.Array, lower, span,
                               n_bins: int = 32) -> jax.Array:
    """Negative soft-histogram marginal entropy for samples ``x`` (N, D) on
    ``[lower, lower+span]`` — the regulariser to use on bounded domains.

    The Gaussian proxy above rewards raw *variance*, which on a bounded domain
    is degenerate twice over: its true maximiser is a two-point mass at the
    endpoints (max variance ≠ max entropy), and a sampler can game it further by
    sending a vanishing fraction of samples to extreme values (variance grows
    quadratically in the outliers, so ``log Var`` inflates without the bulk
    spreading at all).

    This version instead soft-bins each marginal into ``n_bins`` equal cells
    (Gaussian kernel assignment, bandwidth = one bin width, so gradients flow)
    and sums the per-dimension discrete entropies ``-Σ_b p_b log p_b``.  Each
    term is bounded by ``log n_bins``, attained exactly by the uniform marginal,
    and outliers cannot inflate it — piling mass anywhere only *lowers* it.  The
    sample-native mirror of the tensor networks' structural bin entropy (per-site
    marginals only; cross-dimension structure is not scored).
    """
    a = _soft_bin_assign(x, lower, span, n_bins)          # (N, D, n_bins)
    p = jnp.mean(a, axis=0)                               # (D, n_bins)
    entropy = -jnp.sum(p * jnp.log(p + 1e-12))
    return -entropy


def neg_pairwise_entropy_proxy(x: jax.Array, lower, span, pairs,
                               n_bins: int = 16) -> jax.Array:
    """Negative soft-histogram entropy of 2-D *pair* marginals ``x[:, (i, j)]``.

    The 1-D proxy above only scores per-site marginals, so a sampler can satisfy
    it with a degenerate **joint** — e.g. all mass on the line ``x_j = 1 - x_i``
    has perfectly uniform 1-D marginals.  Scoring 2-D pair histograms (each
    bounded by ``2 log n_bins``, attained by the independent-uniform pair) breaks
    exactly that failure mode: mass concentrated on any curve leaves most of the
    ``n_bins x n_bins`` cells empty and is penalised.

    ``pairs`` is a sequence of ``(i, j)`` site index tuples.  Cost is one einsum
    of ``O(N * len(pairs) * n_bins^2)`` — for many variables pass a subsample of
    pairs rather than all ``D(D-1)/2``.
    """
    a = _soft_bin_assign(x, lower, span, n_bins)          # (N, D, n_bins)
    ii = jnp.asarray([p[0] for p in pairs])
    jj = jnp.asarray([p[1] for p in pairs])
    P = jnp.einsum("npb,npc->pbc", a[:, ii, :], a[:, jj, :]) / x.shape[0]
    entropy = -jnp.sum(P * jnp.log(P + 1e-12))
    return -entropy
