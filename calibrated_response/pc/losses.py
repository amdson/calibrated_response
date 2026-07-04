"""Regularizers and constraint-loss helpers (spec §6).

These are *optional conveniences*. The training loop accepts any
``loss_fn(params) -> scalar``; the constraint helpers below simply build such
scalars from the circuit's query functions, and the regularizers act directly on
the parameter pytree. Nothing here is privileged — a caller can write their own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# constraint-matching terms (built on circuit queries)
# ---------------------------------------------------------------------------

def match_expectation(circuit, params, a, target, event=None, weight=1.0):
    """Squared error on E[a·X (| event)]."""
    return weight * (circuit.expectation(params, a, event) - target) ** 2


def match_prob(circuit, params, event, target, weight=1.0):
    """Squared error on P(event)."""
    return weight * (circuit.prob(params, event) - target) ** 2


def match_cond_prob(circuit, params, event, given, target, weight=1.0):
    """Squared error on P(event | given)."""
    return weight * (circuit.cond_prob(params, event, given) - target) ** 2


def marginal_nll(circuit, params, points, subset=None, weight=1.0):
    """Mean negative log marginal density of ``points`` under the model.

    ``points`` is ``(B, |S|)`` aligned to ``subset``. Minimising this fits the
    model's marginal over ``subset`` to a sample / dataset (equivalent to
    forward-KL up to a constant). Built on :meth:`Circuit.marginal_log_prob`.
    """
    return weight * -jnp.mean(circuit.marginal_log_prob(params, points, subset))


def match_marginal(circuit, params, points, target, subset=None, weight=1.0, log=False):
    """Squared error between the model marginal and a ``target`` at ``points``.

    Use ``log=True`` to match in log-space (target is then a log-density). Handy
    for fitting a marginal *table*: pass the grid of cells as ``points`` and the
    target masses/densities as ``target``.
    """
    if log:
        pred = circuit.marginal_log_prob(params, points, subset)
    else:
        pred = circuit.marginal_prob(params, points, subset)
    return weight * jnp.mean((pred - jnp.asarray(target)) ** 2)


# ---------------------------------------------------------------------------
# regularizers
# ---------------------------------------------------------------------------

def dirichlet_regularizer(circuit, params, alpha0=1.5):
    """Symmetric-Dirichlet weight prior over every sum node (spec §6).

    ``-(alpha0-1) * sum_c log w_c`` summed over all simplex weight groups: the
    leaf input-mixtures, every einsum layer, and the root mixture. Pushes weights
    toward uniform (anti-degeneracy). Cheapest useful regularizer.
    """
    coef = -(alpha0 - 1.0)
    total = 0.0

    if "g_mix" in params:
        total = total + jnp.sum(jax.nn.log_softmax(params["g_mix"], axis=-1))
    if "c_mix" in params:
        total = total + jnp.sum(jax.nn.log_softmax(params["c_mix"], axis=-1))

    for w in params["layer_W"]:
        P, C = w.shape[0], w.shape[1]
        total = total + jnp.sum(jax.nn.log_softmax(w.reshape(P, C, C * C), axis=-1))

    total = total + jnp.sum(jax.nn.log_softmax(params["root_W"]))
    return coef * total


def isotropy_regularizer(circuit, params, key, n_dirs=8, target=1.0):
    """Projection-isotropy penalty (spec §6).

    Draws ``n_dirs`` random directions ``a`` and penalises
    ``(Var(a·X)/||a||^2 - target)^2``, two-sidedly. Guards against collapse
    without imposing a Gaussianizing pressure. Each direction is one exact
    moment pass (§5.3); several are averaged to reduce the rank-1 variance.
    """
    A = jax.random.normal(key, (n_dirs, circuit.n_vars))
    lo = jnp.full((circuit.n_vars,), -jnp.inf)
    hi = jnp.full((circuit.n_vars,), jnp.inf)
    cmask = jnp.ones((circuit.n_vars, circuit.M)) if circuit.M else jnp.zeros((circuit.n_vars, 0))

    def one(a):
        m0, m1, m2 = circuit._moment_pass(params, a, lo, hi, cmask)
        var = jnp.clip(m2 / m0 - (m1 / m0) ** 2, a_min=0.0)
        return var / jnp.sum(a ** 2)

    ratios = jax.vmap(one)(A)
    return jnp.mean((ratios - target) ** 2)


def leaf_entropy_regularizer(circuit, params):
    """Negative average leaf entropy — minimise to *maximise* entropy (spec §6).

    A tractable, factorization-preserving surrogate for the (intractable, hence
    excluded) joint-entropy regularizer: it raises a lower bound on H(p) by
    widening the per-leaf distributions. Because it touches only individual
    leaves — Gaussian log-variances and categorical logits — it cannot introduce
    cross-variable dependence, so it pushes the under-determined part of ``p``
    toward the factorized maximum-entropy default instead of fabricating
    correlation (contrast :func:`uniform_coverage_regularizer`, a marginal-shape
    target that is blind to correlation).

    Returns ``-mean(leaf entropy)`` averaged over the leaf families present.
    Gaussian entropy is ``0.5*log(2*pi*e*sigma^2)``; the constant is dropped, so
    the term reduces to ``-0.5*log_var`` for gaussians and the exact discrete
    entropy for categoricals. Use a modest weight: it is a gentle "be reasonable"
    prior, not an aggressive density-shaper, and an unbounded weight would inflate
    variances without limit.
    """
    ent, n = 0.0, 0
    if "g_logvar" in params:
        ent = ent + jnp.mean(0.5 * params["g_logvar"])
        n += 1
    if "c_logits" in params:
        log_p = jax.nn.log_softmax(params["c_logits"], axis=-1)
        ent = ent + jnp.mean(-jnp.sum(jnp.exp(log_p) * log_p, axis=-1))
        n += 1
    return -(ent / max(n, 1))


def uniform_coverage_regularizer(circuit, params, key, n_points=256):
    """Coverage cross-entropy ``H(uniform, p) = -E_{x~Uniform(domain)}[log p(x)]``.

    Averaged over each gaussian variable's marginal: penalises the model for
    having near-zero marginal density anywhere inside the declared bounds. This
    fills spurious troughs — e.g. the dip a mixture carves at a conditioning
    threshold — by spreading mass across the domain, while leaving the constraint
    masses free (it is the tractable *reverse*-KL direction; cf. the moment-only
    :func:`gaussian_crossentropy_ref`). Built on :meth:`Circuit.marginal_log_prob`.

    Higher weight => flatter, more uniform marginals (fully fills the trough) at
    the cost of softening conditional crispness. Set the weight to taste.
    """
    keys = jax.random.split(key, circuit.n_vars)
    total, n_g = 0.0, 0
    for i, spec in enumerate(circuit.var_specs):
        if spec.kind != "gaussian":
            continue
        pts = jax.random.uniform(keys[i], (n_points, 1), minval=spec.lower, maxval=spec.upper)
        total = total + -jnp.mean(circuit.marginal_log_prob(params, pts, subset=[spec.name]))
        n_g += 1
    return total / max(n_g, 1)


def gaussian_crossentropy_ref(circuit, params, ref_mean, ref_var):
    """Cross-entropy ``-E_p[log r(X)]`` to a fully-factorized Gaussian reference.

    Tractable because it only needs per-variable marginal moments E[X_v], E[X_v^2]
    (one moment pass each), which the circuit computes exactly. Shrinks the
    under-determined part of ``p`` toward a sane default (spec §6, KL-to-reference;
    the entropy term of the KL is dropped — see §10/§6 exclusion).

    Note: being a function of only the first two marginal moments, this controls
    *where* mass sits (mean) and *how spread* it is (variance) — it cannot see
    fine shape, so it will not smoothly fill a density trough (it tends to shrink
    mass into a spike at ``ref_mean`` instead). To fill troughs use
    :func:`uniform_coverage_regularizer`.
    """
    total = 0.0
    for i, spec in enumerate(circuit.var_specs):
        a = jnp.zeros((circuit.n_vars,)).at[i].set(1.0)
        mean, var = circuit.linear_moment(params, a)
        ex2 = var + mean ** 2
        rv = ref_var[i]
        ce = 0.5 * (jnp.log(2 * jnp.pi * rv) + (ex2 - 2 * ref_mean[i] * mean + ref_mean[i] ** 2) / rv)
        total = total + ce
    return total
