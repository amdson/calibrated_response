"""Invertible (normalizing-flow) sampler with **exact** joint entropy.

Sample-statistic entropy regularisers do not scale to the full joint: 1-D
histograms are blind to mass collapsing onto a curve, pairwise histograms to
mass on a higher-order surface, and joint histogram / kNN estimators are
exponential in dimension or noisy and gameable.  The scalable fix is
architectural: make the sampler *invertible* with ``latent_dim = n_vars``, so
the joint differential entropy is exact by change of variables::

    x = g_theta(z),  z ~ N(0, I_D)
    H(x) = H(z) + E_z[ log |det J_g(z)| ]          # O(D) per sample

For affine coupling layers the log-det is a sum of predicted scales, so the
entropy term costs one extra reduction per sample, is constant-compile-time in
``D``, and cannot be gamed: putting mass on *any* lower-dimensional manifold
drives ``log |det J| -> -inf``, so degenerate joints are infinitely penalised
by construction.  Fitting with ``constraint penalties - lam * H(x)`` is then a
true soft-constrained maximum-entropy problem.

Architecture (RealNVP-style)::

    y_0 = z
    y_i = m_i * y + (1 - m_i) * (y * exp(s_i(m_i * y)) + t_i(m_i * y))
    u   = sigmoid(y_L)                    in (0, 1)^D
    x   = lower + span * u                (done by the model)

with alternating half masks ``m_i``, ``s_i`` clamped by ``s_max * tanh`` for
stability, and the sigmoid squash contributing ``sum log sigmoid'(y)`` to the
log-det (so saturating against a domain edge is also paid for exactly).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


class FlowSampler:
    """Affine-coupling flow ``z -> u in (0,1)^D`` with exact log-det Jacobian.

    Same parameter interface as :class:`NeuralSampler` (``init_params`` /
    ``pack_params`` / ``unpack_params`` / ``sample_fn_flat``), plus
    ``forward_flat(theta, z) -> (u, logdet)`` for the entropy term.

    Parameters
    ----------
    n_vars : int
        Dimensionality of both the latent and the output (invertibility
        requires them equal).
    n_layers : int
        Number of coupling layers (alternating half masks).
    hidden : int
        Hidden width of each coupling layer's scale/shift MLP.
    s_max : float
        Scale clamp: ``s = s_max * tanh(s_raw)`` keeps ``exp(s)`` bounded.
    """

    def __init__(self, n_vars: int, n_layers: int = 6, hidden: int = 64,
                 s_max: float = 3.0):
        self.n_vars = int(n_vars)
        self.latent_dim = self.n_vars          # invertible: same dim
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.s_max = float(s_max)

        # Alternating half masks; for D=1 coupling degenerates, use a plain
        # elementwise affine (still invertible, logdet = sum s).
        d = self.n_vars
        half = np.arange(d) < (d + 1) // 2
        self.masks = [
            jnp.asarray(half if i % 2 == 0 else ~half, jnp.float32)
            for i in range(self.n_layers)
        ]

        _flat, self._unravel_fn = ravel_pytree(self.zero_params())
        self.n_params = int(_flat.shape[0])

    # ---- parameters --------------------------------------------------
    def _layer_shapes(self):
        d, h = self.n_vars, self.hidden
        return [("W1", (h, d)), ("b1", (h,)), ("W2", (2 * d, h)), ("b2", (2 * d,))]

    def zero_params(self):
        return [{k: jnp.zeros(s) for k, s in self._layer_shapes()}
                for _ in range(self.n_layers)]

    def init_params(self, key: jax.Array):
        """He-init hidden weights; **zero** output weights, so the flow starts
        as the identity (``u = sigmoid(z)``: full-domain spread, exact entropy
        gradient from step one, no dead-gradient start)."""
        keys = jax.random.split(key, self.n_layers)
        d, h = self.n_vars, self.hidden
        layers = []
        for i in range(self.n_layers):
            k1, _ = jax.random.split(keys[i])
            layers.append({
                "W1": jax.random.normal(k1, (h, d)) / jnp.sqrt(d),
                "b1": jnp.zeros(h),
                "W2": jnp.zeros((2 * d, h)),
                "b2": jnp.zeros(2 * d),
            })
        return layers

    def pack_params(self, params) -> jax.Array:
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jax.Array):
        return self._unravel_fn(theta)

    # ---- forward pass with log-det -----------------------------------
    def forward_pytree(self, params, z: jax.Array):
        """One latent ``z`` (D,) -> ``(u in (0,1)^D, logdet scalar)``."""
        y = z
        logdet = 0.0
        for layer, m in zip(params, self.masks):
            hcond = jnp.maximum(jnp.dot(layer["W1"], m * y) + layer["b1"], 0.0)
            st = jnp.dot(layer["W2"], hcond) + layer["b2"]
            s = self.s_max * jnp.tanh(st[: self.n_vars])
            t = st[self.n_vars:]
            y = m * y + (1.0 - m) * (y * jnp.exp(s) + t)
            logdet = logdet + jnp.sum((1.0 - m) * s)
        # sigmoid squash onto (0,1): log sigma'(y) = log_sigmoid(y) + log_sigmoid(-y)
        logdet = logdet + jnp.sum(jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y))
        return jax.nn.sigmoid(y), logdet

    def forward_flat(self, theta: jax.Array, z: jax.Array):
        return self.forward_pytree(self.unpack_params(theta), z)

    # ---- inverse pass (exact density evaluation) ----------------------
    def inverse_pytree(self, params, u: jax.Array):
        """One point ``u`` (D,) in (0,1)^D -> ``(z, logdet)``, where ``logdet``
        is the *forward* log|det J| at the recovered ``z`` — so the exact
        density is ``log p(u) = log N(z) - logdet``.

        Coupling layers invert in closed form: the masked half passes through
        unchanged, so ``s, t`` are recomputable from the output, and the
        transformed half is ``(y - t) * exp(-s)``. This makes the flow a
        tractable density model, not just a sampler."""
        u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
        y = jnp.log(u) - jnp.log1p(-u)                        # logit
        logdet = jnp.sum(jnp.log(u) + jnp.log1p(-u))          # sum log sigmoid'(y)
        for layer, m in zip(reversed(params), reversed(self.masks)):
            hcond = jnp.maximum(jnp.dot(layer["W1"], m * y) + layer["b1"], 0.0)
            st = jnp.dot(layer["W2"], hcond) + layer["b2"]
            s = self.s_max * jnp.tanh(st[: self.n_vars])
            t = st[self.n_vars:]
            y = m * y + (1.0 - m) * (y - t) * jnp.exp(-s)
            logdet = logdet + jnp.sum((1.0 - m) * s)
        return y, logdet

    def inverse_flat(self, theta: jax.Array, u: jax.Array):
        return self.inverse_pytree(self.unpack_params(theta), u)

    def sample_fn_pytree(self, params, z: jax.Array) -> jax.Array:
        return self.forward_pytree(params, z)[0]

    def sample_fn_flat(self, theta: jax.Array, z: jax.Array) -> jax.Array:
        """NeuralSampler-compatible forward (drops the log-det)."""
        return self.forward_pytree(self.unpack_params(theta), z)[0]
