"""Generic gradient-descent training loop (spec §6 optimizer).

The loop is deliberately uncommitted to any particular objective: the caller
passes ``loss_fn(params) -> scalar`` (built from circuit queries + regularizers,
or anything else differentiable), and this minimises it with Adam. Returns the
trained pytree and a per-step loss history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import optax


@dataclass
class TrainConfig:
    steps: int = 800
    lr: float = 1e-2
    grad_clip: Optional[float] = 5.0
    log_every: int = 0          # 0 = silent


def train(loss_fn: Callable, params, config: TrainConfig = TrainConfig(),
          optimizer: Optional[optax.GradientTransformation] = None):
    """Minimise ``loss_fn(params)`` with Adam (+ optional global-norm clipping)."""
    if optimizer is None:
        chain = []
        if config.grad_clip:
            chain.append(optax.clip_by_global_norm(config.grad_clip))
        chain.append(optax.adam(config.lr))
        optimizer = optax.chain(*chain)

    opt_state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def step(params, opt_state):
        loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    history = []
    for it in range(config.steps):
        params, opt_state, loss = step(params, opt_state)
        history.append(float(loss))
        if config.log_every and (it % config.log_every == 0 or it == config.steps - 1):
            print(f"  step {it:5d}  loss {float(loss):.6f}")
    return params, history
