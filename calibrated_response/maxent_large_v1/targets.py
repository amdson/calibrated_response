from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from calibrated_response.maxent_large_v1.graph import FactorGraph


@dataclass(frozen=True)
class TargetMoments:
    by_factor_id: dict[int, jnp.ndarray]

def _factor_moment_from_states(states: jnp.ndarray, var_ids: tuple[int, ...], shape: tuple[int, ...]) -> jnp.ndarray:
    sub = np.asarray(states[:, list(var_ids)], dtype=np.int64)
    if sub.ndim == 1:
        sub = sub[:, None]

    strides = []
    prod = 1
    for dim in reversed(shape[1:]):
        prod *= dim
        strides.append(prod)
    strides = [1] if len(shape) == 1 else list(reversed(strides)) + [1]

    flat = np.zeros((sub.shape[0],), dtype=np.int64)
    for i, stride in enumerate(strides):
        flat += sub[:, i] * stride

    counts = np.bincount(flat, minlength=int(np.prod(shape))).astype(np.float32)
    total = counts.sum()
    if total <= 0:
        counts[:] = 1.0 / counts.size
    else:
        counts /= total
    return jnp.asarray(counts.reshape(shape))


def estimate_moments(graph: FactorGraph, states: jnp.ndarray) -> dict[int, jnp.ndarray]:
    moments: dict[int, jnp.ndarray] = {}
    for factor in graph.factors:
        moments[factor.id] = _factor_moment_from_states(states, factor.var_ids, factor.table_shape)
    return moments

def build_targets_from_marginals(graph: FactorGraph, marginal_tables: dict[int, np.ndarray | jnp.ndarray]) -> TargetMoments:
    by_factor: dict[int, jnp.ndarray] = {}
    for factor in graph.factors:
        if factor.id in marginal_tables:
            table = jnp.asarray(marginal_tables[factor.id], dtype=jnp.float32)
            norm = jnp.sum(table)
            by_factor[factor.id] = table / jnp.maximum(norm, 1e-8)
    return TargetMoments(by_factor_id=by_factor)

def build_targets_from_samples(graph: FactorGraph, sample_states: jnp.ndarray) -> TargetMoments:
    return TargetMoments(by_factor_id=estimate_moments(graph, sample_states))