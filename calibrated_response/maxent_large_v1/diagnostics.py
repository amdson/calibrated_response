from __future__ import annotations

import jax.numpy as jnp


def max_moment_error(target_moments: dict[int, jnp.ndarray], model_moments: dict[int, jnp.ndarray]) -> float:
    if not target_moments:
        return 0.0
    values = []
    for factor_id, target in target_moments.items():
        model = model_moments.get(factor_id)
        if model is None:
            continue
        values.append(float(jnp.max(jnp.abs(target - model))))
    return max(values) if values else 0.0


def mean_moment_error(target_moments: dict[int, jnp.ndarray], model_moments: dict[int, jnp.ndarray]) -> float:
    if not target_moments:
        return 0.0
    values = []
    for factor_id, target in target_moments.items():
        model = model_moments.get(factor_id)
        if model is None:
            continue
        values.append(float(jnp.mean(jnp.abs(target - model))))
    return float(sum(values) / len(values)) if values else 0.0
