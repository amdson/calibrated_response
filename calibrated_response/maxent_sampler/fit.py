"""Stochastic fit driver for sample-native losses.

``tn.backends._fit_adam`` optimises a deterministic ``loss(params)`` built on a
*fixed* latent batch.  That makes the loss a pure function (nice for L-BFGS),
but a flexible sampler can **overfit the batch**: it learns to place exactly
those ``z_i`` in high-likelihood configurations while behaving arbitrarily
between them — the Monte-Carlo estimate stays great, the true expectation
collapses (fresh-sample entropy of a flow can be catastrophically below its
train-batch estimate).

The fix is the standard one: redraw ``z`` every step.  ``fit_adam_stochastic``
drives the two-argument ``loss(params, key)`` that ``constraint_loss`` returns,
splitting a fresh PRNG key per step, so there is no fixed batch to overfit —
the optimiser only ever wins by improving the true expectation.
``value_and_grad`` is jitted once; the key is a traced argument.

Optimiser knobs
---------------
Every Adam hyper-parameter is exposed, not just ``lr``.  All of them reach the
builder unchanged, since ``DistributionBuilder.build`` / ``.fit`` and
``SamplerModel.optimize`` forward ``**kw`` straight through::

    b.build(target_variable="anomaly", steps=3000, lr=2e-3,
            b2=0.9999, eps=1e-12, grad_clip=1.0)

The two that matter most here are ``b2`` and ``eps``, because these losses are
*noisy by construction* (a fresh latent batch every step) and some of the
signal is genuinely faint — a weakly-constrained region of the joint can be
worth only ~1e-2 nats, well under the per-step Monte-Carlo spread of the
entropy estimate itself:

``b1`` / ``b2``
    Longer averaging (``b1=0.95``, ``b2=0.9999``) trades adaptation speed for
    variance reduction.  When the per-step gradient SNR is below 1, the useful
    signal only accumulates over many steps, and that is exactly what the
    moment estimates are for.  Raising ``b2`` is usually the cheaper half of
    the trade — it smooths the *scale* without lagging the direction.
``eps``
    Adam steps ``mu_hat / (sqrt(nu_hat) + eps)``.  When a coordinate's
    gradients are uniformly tiny, ``sqrt(nu_hat)`` shrinks toward ``eps`` and
    the default ``1e-8`` starts to dominate the denominator — Adam's scale
    invariance quietly stops holding and the faint direction is damped instead
    of amplified.  Dropping to ``1e-12`` restores it.
``grad_clip``
    Global-norm clip, applied before Adam.  Pass ``None`` or ``0`` to disable.
``weight_decay``
    Non-zero switches ``optax.adam`` -> ``optax.adamw`` (decoupled decay).

``lr`` accepts an optax *schedule* (any ``callable(step) -> lr``) as well as a
float::

    import optax
    b.build(steps=6000,
            lr=optax.warmup_cosine_decay_schedule(1e-4, 2e-3, 500, 6000))

and ``optimizer=`` takes a fully-built ``optax.GradientTransformation``,
bypassing every knob above when you want something the arguments do not cover
(Nesterov momentum, Lion, per-layer masking, gradient accumulation, ...)::

    b.build(steps=3000, optimizer=optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(2e-3, nesterov=True)))
"""

from __future__ import annotations

import jax


def _build_optimizer(lr, grad_clip, b1, b2, eps, eps_root, weight_decay):
    """``optax`` chain: optional global-norm clip, then Adam(W)."""
    import optax

    if weight_decay:
        core = optax.adamw(lr, b1=b1, b2=b2, eps=eps, eps_root=eps_root,
                           weight_decay=weight_decay)
    else:
        core = optax.adam(lr, b1=b1, b2=b2, eps=eps, eps_root=eps_root)
    if not grad_clip:                      # None or 0 -> no clipping
        return core
    return optax.chain(optax.clip_by_global_norm(grad_clip), core)


def fit_adam_stochastic(loss_fn, params, steps=1500, lr=1e-3, grad_clip=5.0,
                        seed=0, log_every=0, *, b1=0.9, b2=0.999, eps=1e-8,
                        eps_root=0.0, weight_decay=0.0, optimizer=None):
    """Adam over a stochastic ``loss_fn(params, key)``, fresh key each step.

    Same contract as ``tn.backends._fit_adam`` otherwise; the returned history
    holds the (noisy) per-step minibatch losses.

    Parameters
    ----------
    steps, seed, log_every
        Iteration count, PRNG seed for the latent stream, print interval.
    lr
        Learning rate: a float, or any optax schedule ``callable(step) -> lr``.
    grad_clip
        Global-norm clip applied before Adam; ``None``/``0`` disables it.
    b1, b2, eps, eps_root, weight_decay
        Adam hyper-parameters — see the module docstring for which ones matter
        for these deliberately-noisy losses.  ``weight_decay != 0`` selects
        ``adamw``.
    optimizer
        A ready-made ``optax.GradientTransformation``.  When given it is used
        verbatim and *all* of ``lr``/``grad_clip``/``b*``/``eps*``/
        ``weight_decay`` are ignored.
    """
    import inspect

    import jax.numpy as jnp
    import optax

    opt = optimizer if optimizer is not None else _build_optimizer(
        lr, grad_clip, b1, b2, eps, eps_root, weight_decay)
    state = opt.init(params)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    # A loss may take (params, key, step) instead of (params, key) -- e.g.
    # constraint_loss(beta_schedule=...) threads the step into a tempering
    # schedule.  The step is passed as a traced f32 scalar (a Python int
    # would recompile every iteration).
    pass_step = len(inspect.signature(loss_fn).parameters) >= 3
    key = jax.random.PRNGKey(seed)
    history = []
    for it in range(steps):
        key, sub = jax.random.split(key)
        if pass_step:
            loss, g = vg(params, sub, jnp.asarray(it, jnp.float32))
        else:
            loss, g = vg(params, sub)
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  [adam*] step {it:5d}  loss {float(loss):.6f}")
    return params, history
