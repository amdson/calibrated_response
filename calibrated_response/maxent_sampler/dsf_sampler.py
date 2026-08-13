"""Deep sigmoidal flow (Huang et al. 2018, "Neural Autoregressive Flows") —
the paper-faithful conditioned variant.

Each flow step is an IAF-ordered autoregressive block: a MADE-masked
conditioner reads the block input and emits, for every dimension ``i``, the
pseudo-parameters ``(w, a, b)`` of a **stack** of sigmoidal transformer
layers applied to that dimension::

    t_0 = y_i
    for s in 1..S:   t_s = logit( sum_k w_sk * sigmoid(a_sk * t_{s-1} + b_sk) )

with ``w`` on the simplex (softmax) and ``a > 0`` (softplus), so every layer
is strictly monotone and the stack is a universal 1-D CDF approximator —
"deep" in the paper's sense: depth *inside* the transformer, parameterized by
the conditioner.  (The DDSF dense generalization — full matrices between
sigmoid layers — is not implemented.)

The conditioner for dimension ``i`` sees only ``y_{<i}`` (MADE masks), so the
Jacobian is triangular and the log-det is the sum of the per-dimension
transformer log-derivatives, accumulated in log space exactly as in
:mod:`sigmix_sampler` (whose ``_sigmix_forward`` is reused as the layer
primitive).  Ordering alternates (identity / reversed) between blocks so no
dimension is permanently first (the first dimension of a block gets
bias-only, i.e. unconditioned, pseudo-parameters).

IAF ordering means the **sampling direction z -> x is the parallel one** —
the conditioner reads the block *input* — so training (forward + log-det
only) is a single masked pass per block, exactly what the exact-entropy
maxent objective needs.  The price is the inverse: x -> z is sequential —
``d`` fixed-point passes per block, each inverting the monotone transformer
stack by bracketed bisection.  Offline density evaluation (``log_prob`` /
held-out NLL) only; **not differentiable** in the parameters, so
``constraint_loss(with_logq=True)`` is refused for this flow type.

Same interface as the other samplers (``init_params`` / ``pack_params`` /
``unpack_params`` / ``forward_flat`` -> ``(u, logdet)`` / ``inverse_flat`` /
``sample_fn_flat`` / ``n_params``); the final sigmoid squash onto (0,1) is
kept identical to the other flows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from calibrated_response.maxent_sampler.sigmix_sampler import (
    _INIT_A_RAW, _sigmix_forward)


class DSFSampler:
    """Neural-autoregressive deep sigmoidal flow with exact log-det Jacobian.

    Parameters
    ----------
    n_vars : int
        Latent = output dimensionality (invertible).
    n_layers : int
        Number of autoregressive blocks (alternating variable order).
    hidden : int
        Hidden width of each block's MADE conditioner.
    n_ds_layers : int
        ``S``: sigmoidal transformer layers stacked *within* each block
        (the paper's "deep").
    n_sigmoids : int
        ``K``: sigmoid components per transformer layer.
    """

    def __init__(self, n_vars: int, n_layers: int = 6, hidden: int = 64,
                 n_ds_layers: int = 2, n_sigmoids: int = 8):
        self.n_vars = int(n_vars)
        self.latent_dim = self.n_vars          # invertible: same dim
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.S = int(n_ds_layers)
        self.K = int(n_sigmoids)

        d, h = self.n_vars, self.hidden
        self.P = d * self.S * 3 * self.K       # conditioner output size

        # MADE masks in natural order; blocks permute their input instead of
        # carrying per-block masks.  deg_h cycles 1..d-1 (bias-only outputs
        # for dimension 0; for d == 1 the whole conditioner is bias-only).
        deg_in = np.arange(1, d + 1)
        deg_h = 1 + (np.arange(h) % max(1, d - 1))
        deg_out = np.repeat(deg_in, self.S * 3 * self.K)
        self._m1 = jnp.asarray(deg_h[:, None] >= deg_in[None, :], jnp.float32)
        self._m2 = jnp.asarray(deg_out[:, None] > deg_h[None, :], jnp.float32)

        perms = [np.arange(d) if i % 2 == 0 else np.arange(d)[::-1]
                 for i in range(self.n_layers)]
        self._perms = [jnp.asarray(p) for p in perms]
        self._invps = [jnp.asarray(np.argsort(p)) for p in perms]

        _flat, self._unravel_fn = ravel_pytree(self.zero_params())
        self.n_params = int(_flat.shape[0])

    # ---- parameters --------------------------------------------------
    def _layer_shapes(self):
        d, h = self.n_vars, self.hidden
        return [("W1", (h, d)), ("b1", (h,)), ("W2", (self.P, h)),
                ("b2", (self.P,))]

    def zero_params(self):
        return [{k: jnp.zeros(s) for k, s in self._layer_shapes()}
                for _ in range(self.n_layers)]

    def init_params(self, key: jax.Array):
        """Near-identity start: zero conditioner output weights, so the
        pseudo-parameters come from the biases alone — set to a transformer
        stack dominated by one ``sigmoid(t)`` component per layer (``a = 1,
        b = 0``, weight ~0.88) with the remaining components spread over
        ``b in [-1.5, 1.5]`` at low weight (+ jitter to break component
        symmetry).  Each layer is then ~= identity through its logit, the
        block ~= identity, and the full flow starts at ``u = sigmoid(z)`` —
        same warm-start convention as the other engines."""
        d, h, S, K = self.n_vars, self.hidden, self.S, self.K
        b2 = np.zeros((d, S, 3, K), np.float32)
        b2[:, :, 0, 0] = 4.0    # dominant identity comp (stronger than the
        # sigmix init: S stacked layers per block compound the deviation)
        b2[:, :, 1, :] = _INIT_A_RAW                  # a ~= 1
        if K > 1:
            b2[:, :, 2, 1:] = np.linspace(-1.5, 1.5, K - 1, dtype=np.float32)
        keys = jax.random.split(key, self.n_layers)
        layers = []
        for i in range(self.n_layers):
            k1, kj = jax.random.split(keys[i])
            jitter = np.zeros((d, S, 3, K), np.float32)
            jitter[:, :, 1:, :] = 0.05 * np.asarray(
                jax.random.normal(kj, (d, S, 2, K)))
            layers.append({
                "W1": jax.random.normal(k1, (h, d)) / jnp.sqrt(d),
                "b1": jnp.zeros(h),
                "W2": jnp.zeros((self.P, h)),
                "b2": jnp.asarray((b2 + jitter).reshape(-1)),
            })
        return layers

    def pack_params(self, params) -> jax.Array:
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jax.Array):
        return self._unravel_fn(theta)

    # ---- MADE conditioner -> transformer pseudo-parameters -----------
    def _conditioner(self, layer, yp):
        """Block input ``yp`` (d, natural order) -> pseudo-params, each
        ``(d, S, K)``.  Row ``i`` depends only on ``yp[:i]`` (masks)."""
        h1 = jnp.maximum(jnp.dot(layer["W1"] * self._m1, yp) + layer["b1"],
                         0.0)
        raw = jnp.dot(layer["W2"] * self._m2, h1) + layer["b2"]
        raw = raw.reshape(self.n_vars, self.S, 3, self.K)
        return raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]

    def _stack(self, t, w_logits, a_raw, b):
        """Sigmoidal transformer stack, elementwise: ``t (d,) -> (t', ld)``
        with ``ld`` the per-dimension log-derivative of the whole stack."""
        ld_tot = jnp.zeros_like(t)
        for s in range(self.S):
            t, ld = _sigmix_forward(t, w_logits[:, s], a_raw[:, s], b[:, s])
            ld_tot = ld_tot + ld
        return t, ld_tot

    def _stack_invert(self, t_target, w_logits, a_raw, b,
                      n_expand: int = 40, n_bisect: int = 60):
        """Elementwise inverse of the monotone transformer stack by bracketed
        bisection (offline density use only — not differentiable)."""
        f = lambda t: self._stack(t, w_logits, a_raw, b)[0]
        lo = t_target - 1.0
        hi = t_target + 1.0
        step = 1.0
        for _ in range(n_expand):
            lo = jnp.where(f(lo) > t_target, lo - step, lo)
            hi = jnp.where(f(hi) < t_target, hi + step, hi)
            step = step * 1.5
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            gt = f(mid) > t_target
            hi = jnp.where(gt, mid, hi)
            lo = jnp.where(gt, lo, mid)
        return 0.5 * (lo + hi)

    # ---- forward pass with log-det -----------------------------------
    def forward_pytree(self, params, z: jax.Array):
        """One latent ``z`` (D,) -> ``(u in (0,1)^D, logdet scalar)``."""
        y = z
        logdet = 0.0
        for layer, perm, invp in zip(params, self._perms, self._invps):
            yp = y[perm]
            wl, ar, bb = self._conditioner(layer, yp)
            tp, ld = self._stack(yp, wl, ar, bb)
            y = tp[invp]
            logdet = logdet + jnp.sum(ld)
        logdet = logdet + jnp.sum(jax.nn.log_sigmoid(y)
                                  + jax.nn.log_sigmoid(-y))
        return jax.nn.sigmoid(y), logdet

    def forward_flat(self, theta: jax.Array, z: jax.Array):
        return self.forward_pytree(self.unpack_params(theta), z)

    # ---- inverse pass (offline density evaluation) --------------------
    def inverse_pytree(self, params, u: jax.Array):
        """One point ``u`` (D,) in (0,1)^D -> ``(z, logdet)`` with ``logdet``
        the *forward* log|det J| at the recovered ``z``.  Each block inverts
        by ``d`` fixed-point passes (pass ``k`` fixes dimension ``k``: its
        pseudo-params depend only on already-recovered dimensions), each pass
        a bracketed bisection of the monotone transformer stack.  Offline use
        only — no parameter gradients through the solve."""
        u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
        y = jnp.log(u) - jnp.log1p(-u)                        # logit
        logdet = jnp.sum(jnp.log(u) + jnp.log1p(-u))          # sum log sig'(y)
        for layer, perm, invp in zip(reversed(params), reversed(self._perms),
                                     reversed(self._invps)):
            tp = y[perm]
            yp = jnp.zeros_like(tp)
            for _ in range(self.n_vars):
                wl, ar, bb = self._conditioner(layer, yp)
                yp = self._stack_invert(tp, wl, ar, bb)
            wl, ar, bb = self._conditioner(layer, yp)
            _, ld = self._stack(yp, wl, ar, bb)
            logdet = logdet + jnp.sum(ld)
            y = yp[invp]
        return y, logdet

    def inverse_flat(self, theta: jax.Array, u: jax.Array):
        return self.inverse_pytree(self.unpack_params(theta), u)

    def sample_fn_pytree(self, params, z: jax.Array) -> jax.Array:
        return self.forward_pytree(params, z)[0]

    def sample_fn_flat(self, theta: jax.Array, z: jax.Array) -> jax.Array:
        """NeuralSampler-compatible forward (drops the log-det)."""
        return self.forward_pytree(self.unpack_params(theta), z)[0]
