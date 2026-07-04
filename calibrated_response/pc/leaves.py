"""Univariate leaf distributions for the probabilistic circuit.

Two leaf families are supported (spec §2):

* **Gaussian** (continuous), parametrised by ``(mu, log_var)``.
* **Categorical** (discrete / binary), parametrised by logits over a fixed
  domain of integer-or-real *values*.

Every family exposes three primitives, all written to broadcast over arbitrary
leading dimensions (so the circuit can vmap / batch them over ``K`` input
distributions per region and over leaf regions):

``log_pdf``
    log density / mass at an observation. Used by the likelihood pass (§5.1).

``restricted_moments``
    the triple ``(e0, t1, t2)`` of *restricted raw moments* of the leaf
    variable ``X`` under an event ``A``::

        e0 = E[ 1{X in A} ]            (probability mass of the event)
        t1 = E[ X * 1{X in A} ]        (restricted first moment)
        t2 = E[ X^2 * 1{X in A} ]      (restricted second moment)

    With ``A`` the whole line this gives ``(1, E[X], E[X^2])``; with ``A`` an
    indicator it gives the truncated moments. This single triple is what every
    query in the circuit is built from (see :mod:`circuit`): marginal
    probabilities are ``e0``, expectations are ``t1`` (mixed up the circuit),
    variances come from ``t2``, and conditionals are ratios ``t1 / e0``.

``sample``
    draw a value given a chosen input-distribution index (used by the numpy
    ancestral sampler, not differentiated).

Nothing here imports from the rest of the codebase.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax.scipy.special import erf, logsumexp

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Gaussian leaf
# ---------------------------------------------------------------------------

def gaussian_log_pdf(mu, log_var, x):
    """log N(x; mu, exp(log_var)). Broadcasts over leading dims."""
    var = jnp.exp(log_var)
    return -0.5 * (jnp.log(2.0 * math.pi) + log_var + (x - mu) ** 2 / var)


def _std_normal_pdf(z):
    return _INV_SQRT_2PI * jnp.exp(-0.5 * z ** 2)


def _std_normal_cdf(z):
    return 0.5 * (1.0 + erf(z / _SQRT2))


def _gaussian_cum_moments(mu, sigma, tau):
    """Raw moments of ``X * 1{X < tau}`` for ``X ~ N(mu, sigma^2)``.

    Returns ``(c0, c1, c2) = (E[1{X<tau}], E[X 1{X<tau}], E[X^2 1{X<tau}])``.
    ``tau = +inf`` yields the full moments ``(1, mu, mu^2+sigma^2)``;
    ``tau = -inf`` yields zeros.
    """
    # Sanitise infinite thresholds *before* the closed form: at tau=+-inf the
    # expression hits 0*inf = NaN, and although we ``where``-select the correct
    # limit below, the NaN in the unselected branch poisons reverse-mode
    # gradients. Computing with a finite stand-in keeps both branches finite.
    pos_inf = jnp.isinf(tau) & (tau > 0)
    neg_inf = jnp.isinf(tau) & (tau < 0)
    tau_safe = jnp.where(jnp.isinf(tau), 0.0, tau)

    z = (tau_safe - mu) / sigma
    Phi = _std_normal_cdf(z)
    phi = _std_normal_pdf(z)

    c0 = Phi
    c1 = mu * Phi - sigma * phi
    c2 = (mu ** 2 + sigma ** 2) * Phi - sigma * (mu + tau_safe) * phi

    full0 = jnp.ones_like(c0)
    full1 = jnp.broadcast_to(mu, c1.shape)
    full2 = mu ** 2 + sigma ** 2
    c0 = jnp.where(pos_inf, full0, jnp.where(neg_inf, jnp.zeros_like(c0), c0))
    c1 = jnp.where(pos_inf, full1, jnp.where(neg_inf, jnp.zeros_like(c1), c1))
    c2 = jnp.where(pos_inf, full2, jnp.where(neg_inf, jnp.zeros_like(c2), c2))
    return c0, c1, c2


def gaussian_restricted_moments(mu, log_var, lo, hi):
    """``(e0, t1, t2)`` of ``X ~ N(mu, exp(log_var))`` restricted to ``(lo, hi)``.

    ``lo``/``hi`` may be ``-inf``/``+inf`` to leave a side unbounded (a plain
    threshold event), or both infinite to marginalise (gives ``(1, mu, m2)``).
    """
    sigma = jnp.exp(0.5 * log_var)
    c0_hi, c1_hi, c2_hi = _gaussian_cum_moments(mu, sigma, hi)
    c0_lo, c1_lo, c2_lo = _gaussian_cum_moments(mu, sigma, lo)
    return c0_hi - c0_lo, c1_hi - c1_lo, c2_hi - c2_lo


# ---------------------------------------------------------------------------
# Categorical leaf
# ---------------------------------------------------------------------------

def categorical_log_probs(logits):
    """log p over the domain. ``logits`` shape ``(..., M)``."""
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def categorical_log_pdf_at(logits, values, x):
    """log p(X = x). ``x`` is a *value* (matched against ``values``)."""
    logp = categorical_log_probs(logits)                      # (..., M)
    # one-hot over the matching value (assumes values are distinct).
    onehot = (jnp.abs(values - x) < 1e-9).astype(logits.dtype)  # (..., M)
    return logsumexp(logp + jnp.log(onehot + 1e-30), axis=-1)


def categorical_restricted_moments(logits, values, mask):
    """``(e0, t1, t2)`` for a categorical leaf over ``values`` (shape ``(..., M)``).

    ``mask`` (broadcastable to ``(..., M)``) selects which categories are in the
    event; all-ones marginalises.
    """
    probs = jnp.exp(categorical_log_probs(logits))            # (..., M)
    w = probs * mask
    e0 = jnp.sum(w, axis=-1)
    t1 = jnp.sum(w * values, axis=-1)
    t2 = jnp.sum(w * values ** 2, axis=-1)
    return e0, t1, t2
