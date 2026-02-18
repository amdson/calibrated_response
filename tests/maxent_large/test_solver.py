"""Tests for calibrated_response.maxent_large.maxent_solver."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from calibrated_response.maxent_large.features import (
    MomentFeature,
    SoftThresholdFeature,
)
from calibrated_response.maxent_large.maxent_solver import JAXSolverConfig, MaxEntSolver
from calibrated_response.models.variable import ContinuousVariable


def _cont(name: str = "x") -> ContinuousVariable:
    return ContinuousVariable(name=name, description=name, lower_bound=0.0, upper_bound=1.0)

class TestMaxEntSolverBuildSolve:
    """Basic build/solve round-trip."""

    def test_single_mean_constraint(self):
        """Single variable with E[X]=0.7 → chain mean should be ≈ 0.7."""
        config = JAXSolverConfig(
            num_chains=64,
            num_iterations=150,
            mcmc_steps_per_iteration=3,
            learning_rate=0.05,
            hmc_step_size=0.02,
            hmc_leapfrog_steps=8,
            seed=12,
        )
        solver = MaxEntSolver(config)
        specs = [MomentFeature(var_idx=0, order=1)]
        targets = jnp.array([0.7])

        solver.build(var_specs=[_cont()], feature_specs=specs, feature_targets=targets)
        theta, info = solver.solve()

        chain_mean = float(info["chain_states"][:, 0].mean())
        assert abs(chain_mean - 0.7) < 0.15, f"Expected ≈0.7, got {chain_mean}"

    def test_two_features(self):
        """Two independent variables with mean constraints."""
        config = JAXSolverConfig(
            num_chains=64,
            num_iterations=200,
            mcmc_steps_per_iteration=3,
            learning_rate=0.05,
            hmc_step_size=0.02,
            hmc_leapfrog_steps=8,
            seed=0,
        )
        solver = MaxEntSolver(config)
        specs = [
            MomentFeature(var_idx=0, order=1),
            MomentFeature(var_idx=1, order=1),
        ]
        targets = jnp.array([0.3, 0.8])

        solver.build(var_specs=[_cont("a"), _cont("b")], feature_specs=specs, feature_targets=targets)
        theta, info = solver.solve()

        means = info["chain_states"].mean(axis=0)
        assert abs(means[0] - 0.3) < 0.15
        assert abs(means[1] - 0.8) < 0.15

    def test_threshold_constraint(self):
        """P(X > 0.5) ≈ 0.8 — most mass above 0.5."""
        config = JAXSolverConfig(
            num_chains=64,
            num_iterations=200,
            mcmc_steps_per_iteration=3,
            learning_rate=0.05,
            hmc_step_size=0.02,
            hmc_leapfrog_steps=8,
            seed=5,
        )
        solver = MaxEntSolver(config)
        specs = [SoftThresholdFeature(var_idx=0, threshold=0.5, direction="greater")]
        targets = jnp.array([0.8])

        solver.build(var_specs=[_cont()], feature_specs=specs, feature_targets=targets)
        theta, info = solver.solve()

        frac_above = float((info["chain_states"][:, 0] > 0.5).mean())
        assert frac_above > 0.55, f"Expected >55% above 0.5, got {frac_above:.2f}"

    def test_info_keys(self):
        solver = MaxEntSolver(JAXSolverConfig(num_iterations=5, num_chains=8))
        solver.build(var_specs=[_cont()], feature_specs=[MomentFeature(var_idx=0)], feature_targets=jnp.array([0.5]))
        _, info = solver.solve()
        assert "history" in info
        assert "chain_states" in info
        assert "theta" in info
        assert "converged" in info
