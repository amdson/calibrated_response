"""Tests for calibrated_response.maxent_large.mcmc."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrated_response.models.variable import Variable, ContinuousVariable, BinaryVariable
from calibrated_response.maxent_large.mcmc import (
    HMCConfig,
    PersistentBuffer,
    _reflect_with_momentum,
    _hmc_step,
    _leapfrog,
    advance_buffer,
    build_hmc_step_fn,
)


class TestReflectWithMomentum:
    def test_in_range(self):
        q = jnp.array([0.3, 0.7])
        p = jnp.array([1.0, -1.0])
        q_r, p_r = _reflect_with_momentum(q, p)
        np.testing.assert_allclose(q_r, q, atol=1e-7)
        np.testing.assert_allclose(p_r, p, atol=1e-7)

    def test_above_one(self):
        q = jnp.array([1.2])
        p = jnp.array([1.0])
        q_r, p_r = _reflect_with_momentum(q, p)
        assert 0.0 <= float(q_r[0]) <= 1.0
        np.testing.assert_allclose(q_r, jnp.array([0.8]), atol=1e-6)
        # Momentum should flip
        np.testing.assert_allclose(p_r, jnp.array([-1.0]), atol=1e-6)

    def test_below_zero(self):
        q = jnp.array([-0.3])
        p = jnp.array([-1.0])
        q_r, p_r = _reflect_with_momentum(q, p)
        assert 0.0 <= float(q_r[0]) <= 1.0
        np.testing.assert_allclose(q_r, jnp.array([0.3]), atol=1e-6)
        # Momentum should flip (mod maps -0.3 → 1.7, which is >1, so reflected)
        np.testing.assert_allclose(p_r, jnp.array([1.0]), atol=1e-6)


class TestPersistentBuffer:
    def test_initialize(self):
        key = jax.random.PRNGKey(0)
        buf = PersistentBuffer.initialize(n_chains=16, rng_key=key, n_vars=3)
        assert buf.states.shape == (16, 3)
        assert float(buf.states.min()) >= 0.0
        assert float(buf.states.max()) <= 1.0

    def test_advance_returns_new_buffer(self):
        key = jax.random.PRNGKey(42)
        buf = PersistentBuffer.initialize(n_chains=8, rng_key=key, n_vars=2)

        # Simple quadratic energy centered at 0.5, signature: (theta, x)
        def energy(theta, x):
            return 5.0 * jnp.sum((x - 0.5) ** 2)

        grad_energy = jax.grad(energy, argnums=1)
        theta = jnp.zeros(1)

        config = HMCConfig(step_size=0.02, num_leapfrog_steps=5, adapt_step_size=False)
        step_fn = build_hmc_step_fn(energy, grad_energy)
        buf2 = advance_buffer(buf, theta=theta, n_steps=3, hmc_config=config, step_fn=step_fn)

        assert buf2.states.shape == buf.states.shape
        # States should have changed
        assert not jnp.allclose(buf.states, buf2.states)

    def test_samples_converge_to_known_mean(self):
        """HMC on a quadratic energy ∝ (x - 0.6)² should produce mean ≈ 0.6."""
        key = jax.random.PRNGKey(7)
        buf = PersistentBuffer.initialize(n_chains=64, rng_key=key, step_size=0.02, n_vars=1)

        center = 0.6
        def energy(theta, x):
            return 10.0 * jnp.sum((x - center) ** 2)

        grad_energy = jax.grad(energy, argnums=1)
        theta = jnp.zeros(1)
        config = HMCConfig(step_size=0.02, num_leapfrog_steps=8, adapt_step_size=True)

        step_fn = build_hmc_step_fn(energy, grad_energy)
        for _ in range(50):
            buf = advance_buffer(buf, theta=theta, n_steps=1, hmc_config=config, step_fn=step_fn)

        mean = float(buf.states.mean())
        assert abs(mean - center) < 0.1, f"Expected mean ≈ {center}, got {mean}"

    def test_states_stay_in_bounds(self):
        key = jax.random.PRNGKey(99)
        buf = PersistentBuffer.initialize(n_chains=32, rng_key=key, n_vars=3)

        # Energy that pushes toward the boundary
        def energy(theta, x):
            return -5.0 * jnp.sum(x)

        grad_energy = jax.grad(energy, argnums=1)
        theta = jnp.zeros(1)
        config = HMCConfig(step_size=0.05, num_leapfrog_steps=10, adapt_step_size=False)

        step_fn = build_hmc_step_fn(energy, grad_energy)
        for _ in range(20):
            buf = advance_buffer(buf, theta=theta, n_steps=1, hmc_config=config, step_fn=step_fn)

        assert float(buf.states.min()) >= 0.0
        assert float(buf.states.max()) <= 1.0
