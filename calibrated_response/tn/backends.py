"""Shared machinery for the tensor-network density models.

Extracted from :mod:`chain` so that both :class:`~calibrated_response.tn.tree.TensorTree`
and its path-topology subclass :class:`~calibrated_response.tn.chain.TensorChain` can
depend on it without an import cycle (``chain`` imports ``tree``; both import here):

* :func:`_apply_kind` — the raw-core -> contraction-core map for the two model kinds.
* the fit *backends* (:data:`FIT_BACKENDS`) — pluggable optimisation / decomposition
  processes swapped freely by name, plus :func:`reusable_adam`, the compile-once
  driver for a two-argument ``loss_fn(params, targets)``.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


# ======================================================================
# core parameterisation
# ======================================================================

def _apply_kind(cores, kind):
    """Map raw (unconstrained) cores to the cores used in contractions."""
    if kind == "born":
        return cores                      # real, may be negative
    if kind == "nonneg":
        return [jnp.abs(c) for c in cores]  # nonnegative TT
    raise ValueError(f"unknown model kind {kind!r}")


# ======================================================================
# fit backends  (pluggable "decomposition processes")
# ======================================================================

def _fit_adam(loss_fn, params, steps=1500, lr=5e-2, grad_clip=5.0, log_every=0):
    import optax

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    state = opt.init(params)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    history = []
    for it in range(steps):
        loss, g = vg(params)
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  [adam] step {it:5d}  loss {float(loss):.6f}")
    return params, history


def reusable_adam(loss_fn, steps=2000, lr=2e-2, grad_clip=5.0):
    """A reusable Adam driver for a *two-argument* loss ``loss_fn(params, targets)``.

    The compile-once companion to :func:`_fit_adam`: ``value_and_grad`` is jitted
    **once** with ``targets`` as a traced argument, so refitting the same
    constraint structure with new target values (see
    :func:`losses.batched_constraint_loss`) reuses the XLA executable instead of
    recompiling. Returns ``fit(init_params, targets) -> (params, history)``; call
    it repeatedly with different ``init_params`` / ``targets`` and only the first
    call pays the compile.
    """
    import optax

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    vg = jax.jit(jax.value_and_grad(loss_fn))

    def fit(init_params, targets):
        params = init_params
        state = opt.init(params)
        history = []
        for _ in range(steps):
            loss, g = vg(params, targets)
            updates, state = opt.update(g, state, params)
            params = optax.apply_updates(params, updates)
            history.append(float(loss))
        return params, history

    return fit


def _fit_lbfgs(loss_fn, params, maxiter=800, log_every=0):
    """Scipy L-BFGS-B over flattened cores (a genuinely different process)."""
    from scipy.optimize import minimize
    from jax.flatten_util import ravel_pytree

    x0, unravel = ravel_pytree(params)
    x0 = np.asarray(x0, np.float64)

    vg = jax.jit(jax.value_and_grad(lambda flat: loss_fn(unravel(flat))))
    history = []

    def fun(flat):
        v, g = vg(jnp.asarray(flat, jnp.float32))
        history.append(float(v))
        return float(v), np.asarray(g, np.float64)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9})
    if log_every:
        print(f"  [lbfgs] {res.nit} iters  loss {float(res.fun):.6f}  ({res.message})")
    return unravel(jnp.asarray(res.x, jnp.float32)), history


FIT_BACKENDS: dict[str, Callable] = {
    "adam": _fit_adam,
    "lbfgs": _fit_lbfgs,
}
