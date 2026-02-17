from __future__ import annotations

import jax.numpy as jnp

from calibrated_response.maxent_large_v1.graph import Factor, FactorGraph


def _factor_energy(factor: Factor, state: jnp.ndarray) -> jnp.ndarray:
    idx = tuple(int(state[var_id]) for var_id in factor.var_ids)
    return factor.theta[idx]


def energy(graph: FactorGraph, state: jnp.ndarray) -> jnp.ndarray:
    total = 0.0
    for factor in graph.factors:
        total = total + _factor_energy(factor, state)
    return jnp.asarray(total)


def local_energy_delta(graph: FactorGraph, state: jnp.ndarray, var_id: int, proposed_value: int) -> jnp.ndarray:
    old_value = int(state[var_id])
    if old_value == int(proposed_value):
        return jnp.asarray(0.0)

    old_energy = 0.0
    new_energy = 0.0
    for factor_idx in graph.var_to_factors[var_id]:
        factor = graph.factors[factor_idx]
        old_idx = tuple(int(state[v]) for v in factor.var_ids)
        old_energy += factor.theta[old_idx]

        new_assign = [int(state[v]) for v in factor.var_ids]
        pos = factor.var_ids.index(var_id)
        new_assign[pos] = int(proposed_value)
        new_energy += factor.theta[tuple(new_assign)]

    return jnp.asarray(new_energy - old_energy)
