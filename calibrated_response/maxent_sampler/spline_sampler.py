"""Rational-quadratic **neural spline** coupling flow — a drop-in, more
expressive alternative to the affine :class:`FlowSampler`.

Same interface (``init_params`` / ``pack_params`` / ``unpack_params`` /
``forward_flat`` -> ``(u, logdet)`` / ``inverse_flat`` / ``sample_fn_flat`` /
``n_params``) so :class:`~calibrated_response.maxent_sampler.flow_model.FlowSamplerModel`
can swap it in and keep its exact-entropy machinery unchanged.

Why: affine coupling (``y*exp(s) + t``) transports each half with a single
scale+shift, so representing sharp or heavy-tailed conditionals — and placing
correct mass in the joint tail of a near-deterministic relationship — needs many
layers.  A monotonic **rational-quadratic spline** (Durkan et al. 2019) replaces
that with a piecewise transform of ``K`` bins on ``[-B, B]`` with linear
(identity) tails, so one coupling layer can already bend a marginal into shape.
The log-det is still a closed-form sum of per-element spline derivatives, so the
exact-entropy property is preserved.

Architecture mirrors :class:`FlowSampler`::

    y_0 = z
    y_{i+1} = m_i * y_i + (1 - m_i) * RQspline_{theta_i(m_i * y_i)}(y_i)
    u = sigmoid(y_L)  in (0, 1)^D

with alternating half masks.  At init the spline is the identity (uniform knots,
unit derivatives) so the flow starts as ``u = sigmoid(z)`` — same warm start as
the affine version.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

_MIN_BIN = 1e-3      # floor on bin width/height (numerical stability)
_MIN_DERIV = 1e-3    # floor on knot derivative
# inverse-softplus of (1 - _MIN_DERIV): init raw so derivative starts at ~1.0
_INIT_DERIV_RAW = float(np.log(np.expm1(1.0 - _MIN_DERIV)))


def _softplus(x):
    return jax.nn.softplus(x)


def _rq_decode(raw, B, K):
    """raw (d, 3K-1) -> (xk, yk, derivs) knot arrays, each (d, K+1)."""
    uw, uh, ud = raw[:, :K], raw[:, K:2 * K], raw[:, 2 * K:]
    widths = _MIN_BIN + (2.0 * B - K * _MIN_BIN) * jax.nn.softmax(uw, axis=-1)
    heights = _MIN_BIN + (2.0 * B - K * _MIN_BIN) * jax.nn.softmax(uh, axis=-1)
    d_int = _MIN_DERIV + _softplus(ud)                       # (d, K-1)
    ones = jnp.ones((raw.shape[0], 1), raw.dtype)
    derivs = jnp.concatenate([ones, d_int, ones], axis=-1)   # (d, K+1)
    zeros = jnp.zeros((raw.shape[0], 1), raw.dtype)
    xk = -B + jnp.concatenate([zeros, jnp.cumsum(widths, axis=-1)], axis=-1)
    yk = -B + jnp.concatenate([zeros, jnp.cumsum(heights, axis=-1)], axis=-1)
    return xk, yk, derivs


def _bin_index(vals, knots, K):
    """Per-row bin index of ``vals`` in sorted ``knots`` (d, K+1) -> (d,)."""
    idx = jnp.sum(vals[:, None] >= knots, axis=-1) - 1
    return jnp.clip(idx, 0, K - 1)


def _gather(A, idx):
    return jnp.take_along_axis(A, idx[:, None], axis=-1)[:, 0]


def _rq_forward(x, raw, B, K):
    """Elementwise RQ spline ``x (d,) -> (y (d,), logabsdet (d,))``."""
    xk, yk, derivs = _rq_decode(raw, B, K)
    inside = jnp.abs(x) <= B
    b = _bin_index(x, xk, K)
    xlo, xhi = _gather(xk, b), _gather(xk, b + 1)
    ylo, yhi = _gather(yk, b), _gather(yk, b + 1)
    dlo, dhi = _gather(derivs, b), _gather(derivs, b + 1)
    w, h = xhi - xlo, yhi - ylo
    s = h / w
    xi = jnp.clip((x - xlo) / w, 0.0, 1.0)
    xi1 = 1.0 - xi
    den = s + (dhi + dlo - 2.0 * s) * xi * xi1
    y_in = ylo + (h * (s * xi * xi + dlo * xi * xi1)) / den
    deriv = (s * s * (dhi * xi * xi + 2.0 * s * xi * xi1 + dlo * xi1 * xi1)
             / (den * den))
    y = jnp.where(inside, y_in, x)
    ld = jnp.where(inside, jnp.log(deriv), 0.0)
    return y, ld


def _rq_inverse(y, raw, B, K):
    """Inverse spline ``y (d,) -> (x (d,), logabsdet_forward (d,))``."""
    xk, yk, derivs = _rq_decode(raw, B, K)
    inside = jnp.abs(y) <= B
    b = _bin_index(y, yk, K)
    xlo, xhi = _gather(xk, b), _gather(xk, b + 1)
    ylo, yhi = _gather(yk, b), _gather(yk, b + 1)
    dlo, dhi = _gather(derivs, b), _gather(derivs, b + 1)
    w, h = xhi - xlo, yhi - ylo
    s = h / w
    yrel = y - ylo
    delta = dhi + dlo - 2.0 * s
    a = h * (s - dlo) + yrel * delta
    bb = h * dlo - yrel * delta
    c = -s * yrel
    disc = jnp.clip(bb * bb - 4.0 * a * c, a_min=0.0)
    xi = jnp.clip(2.0 * c / (-bb - jnp.sqrt(disc)), 0.0, 1.0)
    xi1 = 1.0 - xi
    x_in = xi * w + xlo
    den = s + delta * xi * xi1
    deriv = (s * s * (dhi * xi * xi + 2.0 * s * xi * xi1 + dlo * xi1 * xi1)
             / (den * den))
    x = jnp.where(inside, x_in, y)
    ld = jnp.where(inside, jnp.log(deriv), 0.0)
    return x, ld


class SplineFlowSampler:
    """RQ neural-spline coupling flow with exact log-det Jacobian.

    Parameters
    ----------
    n_vars : int
        Latent = output dimensionality (invertible).
    n_layers : int
        Coupling layers (alternating half masks).
    hidden : int
        Hidden width of each layer's conditioner MLP.
    num_bins : int
        Spline bins ``K`` per transformed dimension.
    tail_bound : float
        ``B``: spline acts on ``[-B, B]``, identity (slope-1) outside.
    """

    def __init__(self, n_vars: int, n_layers: int = 6, hidden: int = 64,
                 num_bins: int = 8, tail_bound: float = 4.0):
        self.n_vars = int(n_vars)
        self.latent_dim = self.n_vars
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.K = int(num_bins)
        self.B = float(tail_bound)
        self._per_dim = 3 * self.K - 1          # widths K, heights K, derivs K-1
        self._P = self.n_vars * self._per_dim

        d = self.n_vars
        half = np.arange(d) < (d + 1) // 2
        self.masks = [jnp.asarray(half if i % 2 == 0 else ~half, jnp.float32)
                      for i in range(self.n_layers)]

        _flat, self._unravel_fn = ravel_pytree(self.zero_params())
        self.n_params = int(_flat.shape[0])

    # ---- parameters --------------------------------------------------
    def _layer_shapes(self):
        d, h, P = self.n_vars, self.hidden, self._P
        return [("W1", (h, d)), ("b1", (h,)), ("W2", (P, h)), ("b2", (P,))]

    def zero_params(self):
        return [{k: jnp.zeros(s) for k, s in self._layer_shapes()}
                for _ in range(self.n_layers)]

    def _identity_b2(self):
        """Bias that makes the spline the identity at init (W2 = 0):
        uniform widths/heights (raw 0) and unit derivatives."""
        b2 = np.zeros((self.n_vars, self._per_dim), np.float32)
        b2[:, 2 * self.K:] = _INIT_DERIV_RAW
        return jnp.asarray(b2.reshape(-1))

    def init_params(self, key: jax.Array):
        keys = jax.random.split(key, self.n_layers)
        d, h = self.n_vars, self.hidden
        b2 = self._identity_b2()
        layers = []
        for i in range(self.n_layers):
            k1, _ = jax.random.split(keys[i])
            layers.append({
                "W1": jax.random.normal(k1, (h, d)) / jnp.sqrt(d),
                "b1": jnp.zeros(h),
                "W2": jnp.zeros((self._P, h)),   # zero -> spline = identity init
                "b2": b2,
            })
        return layers

    def pack_params(self, params) -> jax.Array:
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jax.Array):
        return self._unravel_fn(theta)

    # ---- forward pass with log-det -----------------------------------
    def forward_pytree(self, params, z: jax.Array):
        y = z
        logdet = 0.0
        for layer, m in zip(params, self.masks):
            hcond = jnp.maximum(jnp.dot(layer["W1"], m * y) + layer["b1"], 0.0)
            raw = (jnp.dot(layer["W2"], hcond) + layer["b2"]).reshape(
                self.n_vars, self._per_dim)
            yt, ld = _rq_forward(y, raw, self.B, self.K)
            y = m * y + (1.0 - m) * yt
            logdet = logdet + jnp.sum((1.0 - m) * ld)
        logdet = logdet + jnp.sum(jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y))
        return jax.nn.sigmoid(y), logdet

    def forward_flat(self, theta: jax.Array, z: jax.Array):
        return self.forward_pytree(self.unpack_params(theta), z)

    # ---- inverse pass (exact density evaluation) ----------------------
    def inverse_pytree(self, params, u: jax.Array):
        u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
        y = jnp.log(u) - jnp.log1p(-u)
        logdet = jnp.sum(jnp.log(u) + jnp.log1p(-u))
        for layer, m in zip(reversed(params), reversed(self.masks)):
            hcond = jnp.maximum(jnp.dot(layer["W1"], m * y) + layer["b1"], 0.0)
            raw = (jnp.dot(layer["W2"], hcond) + layer["b2"]).reshape(
                self.n_vars, self._per_dim)
            xt, ld = _rq_inverse(y, raw, self.B, self.K)
            y = m * y + (1.0 - m) * xt
            logdet = logdet + jnp.sum((1.0 - m) * ld)
        return y, logdet

    def inverse_flat(self, theta: jax.Array, u: jax.Array):
        return self.inverse_pytree(self.unpack_params(theta), u)

    def sample_fn_pytree(self, params, z: jax.Array) -> jax.Array:
        return self.forward_pytree(params, z)[0]

    def sample_fn_flat(self, theta: jax.Array, z: jax.Array) -> jax.Array:
        return self.forward_pytree(self.unpack_params(theta), z)[0]
