from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from calibrated_response.maxent_large_v1.diagnostics import max_moment_error, mean_moment_error
from calibrated_response.maxent_large_v1.graph import FactorGraph
from calibrated_response.maxent_large_v1.mcmc.buffer import PersistentBuffer
from calibrated_response.maxent_large_v1.targets import TargetMoments, estimate_moments


@dataclass(frozen=True)
class SMMConfig:
    max_bins: int = 21
    num_chains: int = 64
    num_iterations: int = 200
    mcmc_steps_per_iteration: int = 3
    learning_rate: float = 0.5
    optimizer: Literal["sgd", "adam"] = "adam"
    l2_regularization: float = 1e-4
    seed: int = 0
    temperature: float = 1.0
    grad_clip: float = 5.0

def _clip_grad(grad: jnp.ndarray, grad_clip: float) -> jnp.ndarray:
    return jnp.clip(grad, -abs(float(grad_clip)), abs(float(grad_clip)))

def fit_smm(
    graph: FactorGraph,
    targets: TargetMoments,
    config: SMMConfig,
    skipped_constraints: list[str] | None = None,
) -> tuple[FactorGraph, dict, PersistentBuffer]:
    rng = jax.random.PRNGKey(config.seed)
    buffer = PersistentBuffer.initialize(graph, num_chains=config.num_chains, rng_key=rng)

    history = {
        "iteration": [],
        "max_error": [],
        "mean_error": [],
        "runtime_sec": [],
        "skipped_constraints": list(skipped_constraints or []),
    }

    # Adam state by factor id
    m: dict[int, jnp.ndarray] = {}
    v: dict[int, jnp.ndarray] = {}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    current_graph = graph

    model_moments = estimate_moments(current_graph, buffer.get_states())

    for iteration in range(1, config.num_iterations + 1):
        t0 = time.time()
        buffer = buffer.sample_step(
            current_graph,
            n_sweeps=config.mcmc_steps_per_iteration,
            temperature=config.temperature,
        )

        model_moments = estimate_moments(current_graph, buffer.get_states())

        theta_updates: dict[int, jnp.ndarray] = {}
        for factor in current_graph.factors:
            target = targets.by_factor_id.get(factor.id)
            if target is None:
                continue
            model = model_moments[factor.id]
            grad = target - model - config.l2_regularization * factor.theta
            grad = _clip_grad(grad, config.grad_clip)

            if config.optimizer == "adam":
                m_prev = m.get(factor.id, jnp.zeros_like(grad))
                v_prev = v.get(factor.id, jnp.zeros_like(grad))
                m_new = beta1 * m_prev + (1.0 - beta1) * grad
                v_new = beta2 * v_prev + (1.0 - beta2) * (grad * grad)
                m_hat = m_new / (1.0 - beta1 ** iteration)
                v_hat = v_new / (1.0 - beta2 ** iteration)
                step = config.learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
                m[factor.id] = m_new
                v[factor.id] = v_new
            else:
                step = config.learning_rate * grad

            theta_updates[factor.id] = factor.theta + step

        if theta_updates:
            current_graph = current_graph.with_updated_thetas(theta_updates)

        max_err = max_moment_error(targets.by_factor_id, model_moments)
        mean_err = mean_moment_error(targets.by_factor_id, model_moments)

        history["iteration"].append(iteration)
        history["max_error"].append(max_err)
        history["mean_error"].append(mean_err)
        history["runtime_sec"].append(time.time() - t0)

    history["final_model_moments"] = model_moments
    history["final_target_moments"] = targets.by_factor_id

    return current_graph, history, buffer
