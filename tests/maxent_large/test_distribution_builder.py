"""End-to-end tests for calibrated_response.maxent_large.distribution_builder."""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_response.maxent_large.distribution_builder import DistributionBuilder
from calibrated_response.maxent_large.maxent_solver import JAXSolverConfig
from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.models.query import (
    ConditionalExpectationEstimate,
    ConditionalProbabilityEstimate,
    EqualityProposition,
    ExpectationEstimate,
    InequalityProposition,
    ProbabilityEstimate,
)
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable


# Shared lightweight config for fast tests
_FAST_CFG = JAXSolverConfig(
    num_chains=32,
    num_iterations=80,
    mcmc_steps_per_iteration=2,
    learning_rate=0.05,
    hmc_step_size=0.02,
    hmc_leapfrog_steps=6,
    max_bins=10,
    seed=42,
)


class TestDistributionBuilderSmoke:
    """Basic smoke tests mirroring the v1 smoke test."""

    def test_all_estimate_types(self):
        variables = [
            ContinuousVariable(name="x", description="Target", lower_bound=0.0, upper_bound=1.0, unit="u"),
            BinaryVariable(name="rain", description="Whether it rains"),
        ]

        estimates = [
            ProbabilityEstimate(
                id="p_x_gt",
                proposition=InequalityProposition(variable="x", variable_type="continuous", threshold=0.4, is_lower_bound=True),
                probability=0.7,
            ),
            ExpectationEstimate(id="e_x", variable="x", expected_value=0.55),
            ConditionalProbabilityEstimate(
                id="cp_rain",
                proposition=InequalityProposition(variable="x", variable_type="continuous", threshold=0.6, is_lower_bound=True),
                conditions=[EqualityProposition(variable="rain", variable_type="binary", value=True)],
                probability=0.75,
            ),
            ConditionalExpectationEstimate(
                id="ce_rain_false",
                variable="x",
                conditions=[EqualityProposition(variable="rain", variable_type="binary", value=False)],
                expected_value=0.2,
            ),
        ]

        builder = DistributionBuilder(variables=variables, estimates=estimates, solver_config=_FAST_CFG)
        distribution, info = builder.build(target_variable="x")

        assert len(distribution.bin_probabilities) == 10
        assert np.isclose(sum(distribution.bin_probabilities), 1.0, atol=1e-6)
        assert info["target_variable"] == "x"
        assert info["n_estimates"] == 4

        all_marginals = builder.get_all_marginals(info)
        assert set(all_marginals.keys()) == {"x", "rain"}
        for name, dist in all_marginals.items():
            assert np.isclose(sum(dist.bin_probabilities), 1.0, atol=1e-6)


class TestExpectation:
    """Expectation matching on original-domain variables."""

    def test_mean_near_target(self):
        variables = [
            ContinuousVariable(name="temp", description="Temperature", lower_bound=0.0, upper_bound=100.0),
        ]
        estimates = [
            ExpectationEstimate(id="e_temp", variable="temp", expected_value=70.0),
        ]

        config = JAXSolverConfig(
            num_chains=64,
            num_iterations=150,
            mcmc_steps_per_iteration=3,
            learning_rate=0.05,
            hmc_step_size=0.02,
            hmc_leapfrog_steps=8,
            max_bins=10,
            seed=7,
        )

        builder = DistributionBuilder(variables=variables, estimates=estimates, solver_config=config)
        dist, info = builder.build(target_variable="temp")

        # Compute approximate mean from histogram
        edges = np.array(dist.bin_edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        probs = np.array(dist.bin_probabilities)
        approx_mean = float(np.dot(centers, probs))

        assert abs(approx_mean - 70.0) < 15.0, f"Expected ≈70, got {approx_mean:.1f}"


class TestNoEstimates:
    def test_returns_uniform(self):
        variables = [ContinuousVariable(name="x", description="X", lower_bound=0.0, upper_bound=1.0)]
        builder = DistributionBuilder(variables=variables, estimates=[], solver_config=_FAST_CFG)
        dist, info = builder.build()
        assert "No features" in info.get("message", "")
        probs = np.array(dist.bin_probabilities)
        np.testing.assert_allclose(probs, probs[0], atol=1e-6)


class TestUnknownVariableSkipped:
    def test_unknown_variable_skipped(self):
        variables = [ContinuousVariable(name="x", description="X", lower_bound=0, upper_bound=1)]
        estimates = [
            ExpectationEstimate(id="e_missing", variable="zzz", expected_value=0.5),
        ]
        builder = DistributionBuilder(variables=variables, estimates=estimates, solver_config=_FAST_CFG)
        assert len(builder.skipped) == 1
        assert "zzz" in builder.skipped[0]
