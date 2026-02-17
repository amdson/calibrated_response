from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from calibrated_response.maxent_large_v1.graph import FactorGraph
from calibrated_response.maxent_large_v1.mcmc.kernels import gibbs_sweep


@dataclass(frozen=True)
class PersistentBuffer:
    states: jnp.ndarray
    rng_key: jax.Array

    @classmethod
    def initialize(cls, graph: FactorGraph, num_chains: int, rng_key: jax.Array) -> "PersistentBuffer":
        num_chains = max(1, int(num_chains))
        n_vars = graph.n_variables()
        states = jnp.zeros((num_chains, n_vars), dtype=jnp.int32)

        split = jax.random.split(rng_key, num_chains * n_vars + 1)
        key_out = split[0]
        ptr = 1
        for c in range(num_chains):
            for v in range(n_vars):
                states = states.at[c, v].set(
                    jax.random.randint(split[ptr], shape=(), minval=0, maxval=graph.cardinality(v), dtype=jnp.int32)
                )
                ptr += 1

        return cls(states=states, rng_key=key_out)

    def sample_step(
        self,
        graph: FactorGraph,
        n_sweeps: int,
        temperature: float = 1.0,
    ) -> "PersistentBuffer":
        sweeps = max(1, int(n_sweeps))
        n_chains = int(self.states.shape[0])
        keys = jax.random.split(self.rng_key, sweeps * n_chains + 1)
        key_out = keys[0]
        ptr = 1

        states = self.states
        for _ in range(sweeps):
            next_states = []
            for chain_idx in range(n_chains):
                next_states.append(gibbs_sweep(graph, states[chain_idx], keys[ptr], temperature=temperature))
                ptr += 1
            states = jnp.stack(next_states, axis=0)

        return PersistentBuffer(states=states, rng_key=key_out)

    def get_states(self) -> jnp.ndarray:
        return self.states
