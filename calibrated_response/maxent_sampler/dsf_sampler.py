"""Deep sigmoidal flow — affine couplings interleaved with elementwise
K-component **sigmoidal transforms** (Huang et al. 2018, "Neural Autoregressive
Flows").

Same interface as :class:`FlowSampler` / :class:`SplineFlowSampler`
(``init_params`` / ``pack_params`` / ``unpack_params`` / ``forward_flat`` ->
``(u, logdet)`` / ``inverse_flat`` / ``sample_fn_flat`` / ``n_params``) so
:class:`~calibrated_response.maxent_sampler.flow_model.FlowSamplerModel` can
swap it in and keep the exact-entropy machinery unchanged.

Why: an affine coupling stack pushed through a fixed sigmoid can only produce
logistic-shaped marginals per layer — piling mass asymmetrically against a
hinge, or splitting it bimodally, forces extreme scales and box-corner
collapse.  The sigmoidal transform

    u = sum_i w_i * sigmoid(a_i * y + b_i),   w on the simplex, a_i > 0

is strictly monotone by construction and a universal approximator of 1-D CDFs,
so a single block can bend a marginal into (multimodal, skewed, plateaued)
shape.  Blocks map R -> R by taking ``logit(u)`` so they stack; the **final**
sigmoid squash onto (0,1) is kept identical to the other flows.

Architecture::

    y_0 = z
    for each layer:  y <- affine_coupling(y);  y <- logit(sigmix(y))
    u = sigmoid(y_L)  in (0, 1)^D

Couplings carry cross-variable dependence (as before); the elementwise
sigmoidal blocks carry marginal shape — the O(d*K)-parameter middle ground
before a full neural-autoregressive conditioner.

Log-det: everything accumulates in log space.  Per element,

    d/dy sigmix(y) = sum_i w_i a_i sig(z_i) sig(-z_i),   z_i = a_i y + b_i
    -> logsumexp(log w + log a + logsig(z) + logsig(-z))

and the interior logit adds ``-log u - log(1-u)`` — with both ``log u`` and
``log(1-u)`` computed directly as logsumexps (never via ``1 - u``), so
saturated components cost no precision.

Inverse: no closed form — each block is inverted per element by bracketed
bisection (monotone, so this is exact to float tolerance).  Fine for the
offline ``log_prob`` / held-out-NLL path; **not differentiable** in the flow
parameters (the iterative solve carries no implicit-function gradient), so
``constraint_loss(with_logq=True)`` is refused for this flow type.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

_A_MIN = 1e-3        # floor on component slopes (keeps every component alive)
# inverse-softplus of (1 - _A_MIN): raw value at which the slope starts at 1.0
_INIT_A_RAW = float(np.log(np.expm1(1.0 - _A_MIN)))


def _sigmix_forward(y, w_logits, a_raw, b):
    """Elementwise sigmoidal-mixture block ``y (d,) -> (y' (d,), ld (d,))``.

    ``y'`` is the pre-logit-space output ``logit(sum w sig(a y + b))`` and
    ``ld`` the elementwise log-derivative dy'/dy (both exact, log-space)."""
    log_w = jax.nn.log_softmax(w_logits, axis=-1)            # (d, K)
    a = _A_MIN + jax.nn.softplus(a_raw)                      # (d, K)
    zc = a * y[:, None] + b                                  # (d, K)
    log_u = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(zc), axis=-1)
    log_1mu = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(-zc), axis=-1)
    y_out = log_u - log_1mu                                  # logit(u), exact
    ld = (jax.nn.logsumexp(log_w + jnp.log(a) + jax.nn.log_sigmoid(zc)
                           + jax.nn.log_sigmoid(-zc), axis=-1)
          - log_u - log_1mu)
    return y_out, ld


def _sigmix_invert(y_target, w_logits, a_raw, b, n_expand: int = 40,
                   n_bisect: int = 60):
    """Elementwise inverse of :func:`_sigmix_forward` by bracketed bisection.

    Monotone in ``y``, so expand a bracket around the target then bisect.
    Returns ``y`` such that ``sigmix(y) ~= y_target`` (float32 tolerance).
    NOT differentiable in the block parameters — offline density use only."""
    f = lambda y: _sigmix_forward(y, w_logits, a_raw, b)[0]
    lo = y_target - 1.0
    hi = y_target + 1.0
    step = 1.0
    for _ in range(n_expand):
        lo = jnp.where(f(lo) > y_target, lo - step, lo)
        hi = jnp.where(f(hi) < y_target, hi + step, hi)
        step = step * 1.5
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        gt = f(mid) > y_target
        hi = jnp.where(gt, mid, hi)
        lo = jnp.where(gt, lo, mid)
    return 0.5 * (lo + hi)


class DSFSampler:
    """Coupling + deep-sigmoidal flow with exact log-det Jacobian.

    Parameters
    ----------
    n_vars : int
        Latent = output dimensionality (invertible).
    n_layers : int
        Number of (coupling, sigmoidal-block) layer pairs.
    hidden : int
        Hidden width of each coupling layer's scale/shift MLP.
    s_max : float
        Coupling scale clamp (as in :class:`FlowSampler`).
    n_sigmoids : int
        ``K``: sigmoid components per dimension in each sigmoidal block.
    """

    def __init__(self, n_vars: int, n_layers: int = 6, hidden: int = 64,
                 s_max: float = 3.0, n_sigmoids: int = 8):
        self.n_vars = int(n_vars)
        self.latent_dim = self.n_vars          # invertible: same dim
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.s_max = float(s_max)
        self.K = int(n_sigmoids)

        d = self.n_vars
        half = np.arange(d) < (d + 1) // 2
        self.masks = [jnp.asarray(half if i % 2 == 0 else ~half, jnp.float32)
                      for i in range(self.n_layers)]

        _flat, self._unravel_fn = ravel_pytree(self.zero_params())
        self.n_params = int(_flat.shape[0])

    # ---- parameters --------------------------------------------------
    def _layer_shapes(self):
        d, h, K = self.n_vars, self.hidden, self.K
        return [("W1", (h, d)), ("b1", (h,)), ("W2", (2 * d, h)),
                ("b2", (2 * d,)), ("w_logits", (d, K)), ("a_raw", (d, K)),
                ("b", (d, K))]

    def zero_params(self):
        return [{k: jnp.zeros(s) for k, s in self._layer_shapes()}
                for _ in range(self.n_layers)]

    def init_params(self, key: jax.Array):
        """Near-identity start: zero coupling outputs (as in the other flows)
        and a sigmoidal block dominated by one ``sigmoid(y)`` component
        (``a = 1, b = 0``, weight ~0.88), with the remaining ``K - 1``
        components spread over ``b in [-1.5, 1.5]`` at low weight — this
        breaks component symmetry (identical components would receive
        identical gradients forever) while keeping the map ~= identity, so
        the full-domain-spread warm start and day-one entropy gradient are
        preserved."""
        keys = jax.random.split(key, self.n_layers)
        d, h, K = self.n_vars, self.hidden, self.K
        w0 = np.zeros((d, K), np.float32)
        w0[:, 0] = 2.0                                   # dominant identity comp
        b0 = np.zeros((d, K), np.float32)
        if K > 1:
            b0[:, 1:] = np.linspace(-1.5, 1.5, K - 1, dtype=np.float32)
        layers = []
        for i in range(self.n_layers):
            k1, kb = jax.random.split(keys[i])
            jitter = 0.05 * jax.random.normal(kb, (d, K))
            layers.append({
                "W1": jax.random.normal(k1, (h, d)) / jnp.sqrt(d),
                "b1": jnp.zeros(h),
                "W2": jnp.zeros((2 * d, h)),
                "b2": jnp.zeros(2 * d),
                "w_logits": jnp.asarray(w0),
                "a_raw": jnp.full((d, K), _INIT_A_RAW) + jitter,
                "b": jnp.asarray(b0) + jitter,
            })
        return layers

    def pack_params(self, params) -> jax.Array:
        flat, _ = ravel_pytree(params)
        return flat

    def unpack_params(self, theta: jax.Array):
        return self._unravel_fn(theta)

    # ---- coupling sublayer (identical maths to FlowSampler) -----------
    def _coupling_st(self, layer, m, y):
        hcond = jnp.maximum(jnp.dot(layer["W1"], m * y) + layer["b1"], 0.0)
        st = jnp.dot(layer["W2"], hcond) + layer["b2"]
        s = self.s_max * jnp.tanh(st[: self.n_vars])
        t = st[self.n_vars:]
        return s, t

    # ---- forward pass with log-det -----------------------------------
    def forward_pytree(self, params, z: jax.Array):
        """One latent ``z`` (D,) -> ``(u in (0,1)^D, logdet scalar)``."""
        y = z
        logdet = 0.0
        for layer, m in zip(params, self.masks):
            s, t = self._coupling_st(layer, m, y)
            y = m * y + (1.0 - m) * (y * jnp.exp(s) + t)
            logdet = logdet + jnp.sum((1.0 - m) * s)
            y, ld = _sigmix_forward(y, layer["w_logits"], layer["a_raw"],
                                    layer["b"])
            logdet = logdet + jnp.sum(ld)
        logdet = logdet + jnp.sum(jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y))
        return jax.nn.sigmoid(y), logdet

    def forward_flat(self, theta: jax.Array, z: jax.Array):
        return self.forward_pytree(self.unpack_params(theta), z)

    # ---- inverse pass (offline density evaluation) ---------------------
    def inverse_pytree(self, params, u: jax.Array):
        """One point ``u`` (D,) in (0,1)^D -> ``(z, logdet)`` with ``logdet``
        the *forward* log|det J| at the recovered ``z``.  The sigmoidal blocks
        invert by bisection (exact to float tolerance, ~100 evals/block);
        offline use only — no parameter gradients through the solve."""
        u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
        y = jnp.log(u) - jnp.log1p(-u)                        # logit
        logdet = jnp.sum(jnp.log(u) + jnp.log1p(-u))          # sum log sigmoid'(y)
        for layer, m in zip(reversed(params), reversed(self.masks)):
            y = _sigmix_invert(y, layer["w_logits"], layer["a_raw"],
                               layer["b"])
            _, ld = _sigmix_forward(y, layer["w_logits"], layer["a_raw"],
                                    layer["b"])
            logdet = logdet + jnp.sum(ld)
            s, t = self._coupling_st(layer, m, y)
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
