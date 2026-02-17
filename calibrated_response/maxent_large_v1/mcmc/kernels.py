from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from calibrated_response.maxent_large_v1.graph import FactorGraph


def _factor_score_given(graph: FactorGraph, state: jnp.ndarray, factor_idx: int, var_id: int, candidate: int) -> jnp.ndarray:
    factor = graph.factors[factor_idx]
    assignment = [int(state[v]) for v in factor.var_ids]
    pos = factor.var_ids.index(var_id)
    assignment[pos] = int(candidate)
    return factor.theta[tuple(assignment)]


def gibbs_update(graph: FactorGraph, state: jnp.ndarray, var_id: int, rng_key: jax.Array, temperature: float = 1.0) -> jnp.ndarray:
    n_states = graph.cardinality(var_id)
    logits = []
    for candidate in range(n_states):
        score = 0.0
        for factor_idx in graph.var_to_factors[var_id]:
            score += _factor_score_given(graph, state, factor_idx, var_id, candidate)
        logits.append(score)

    logits_arr = jnp.asarray(logits, dtype=jnp.float32) / max(float(temperature), 1e-8)
    sampled = jax.random.categorical(rng_key, logits_arr)
    return state.at[var_id].set(sampled)


def gibbs_sweep(
    graph: FactorGraph,
    state: jnp.ndarray,
    rng_key: jax.Array,
    temperature: float = 1.0,
    order: Sequence[int] | None = None,
) -> jnp.ndarray:
    variable_order = list(order) if order is not None else list(range(graph.n_variables()))
    keys = jax.random.split(rng_key, len(variable_order))
    current = state
    for key, var_id in zip(keys, variable_order):
        current = gibbs_update(graph, current, var_id, key, temperature=temperature)
    return current
