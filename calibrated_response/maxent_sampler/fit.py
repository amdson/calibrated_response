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
"""

from __future__ import annotations

import jax


def fit_adam_stochastic(loss_fn, params, steps=1500, lr=1e-3, grad_clip=5.0,
                        seed=0, log_every=0):
    """Adam over a stochastic ``loss_fn(params, key)``, fresh key each step.

    Same contract as ``tn.backends._fit_adam`` otherwise; the returned history
    holds the (noisy) per-step minibatch losses.
    """
    import optax

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    state = opt.init(params)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    key = jax.random.PRNGKey(seed)
    history = []
    for it in range(steps):
        key, sub = jax.random.split(key)
        loss, g = vg(params, sub)
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  [adam*] step {it:5d}  loss {float(loss):.6f}")
    return params, history
