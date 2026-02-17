from __future__ import annotations

import numpy as np

from calibrated_response.maxent_large_v1.distribution_builder import DistributionBuilder
from calibrated_response.maxent_large_v1.training import SMMConfig
from calibrated_response.models.query import (
    ConditionalExpectationEstimate,
    ConditionalProbabilityEstimate,
    EqualityProposition,
    ExpectationEstimate,
    InequalityProposition,
    ProbabilityEstimate,
)
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable


def test_maxent_large_distribution_builder_smoke() -> None:
    variables = [
        ContinuousVariable(
            name="x",
            description="Target",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="u",
        ),
        BinaryVariable(name="rain", description="Whether it rains"),
    ]

    estimates = [
        ProbabilityEstimate(
            id="p_x_gt",
            proposition=InequalityProposition(
                variable="x",
                variable_type="continuous",
                threshold=0.4,
                is_lower_bound=True,
            ),
            probability=0.7,
        ),
        ExpectationEstimate(id="e_x", variable="x", expected_value=0.55),
        ConditionalProbabilityEstimate(
            id="cp_rain",
            proposition=InequalityProposition(
                variable="x",
                variable_type="continuous",
                threshold=0.6,
                is_lower_bound=True,
            ),
            conditions=[
                EqualityProposition(variable="rain", variable_type="binary", value=True)
            ],
            probability=0.75,
        ),
        ConditionalExpectationEstimate(
            id="ce_supported",
            variable="x",
            conditions=[
                EqualityProposition(variable="rain", variable_type="binary", value=False)
            ],
            expected_value=0.2,
        ),
    ]

    builder = DistributionBuilder(
        variables=variables,
        estimates=estimates,
        solver_config=SMMConfig(num_iterations=15, num_chains=32, mcmc_steps_per_iteration=2, max_bins=12, seed=7),
    )

    distribution, info = builder.build(target_variable="x")

    assert len(distribution.bin_probabilities) == 12
    assert np.isclose(sum(distribution.bin_probabilities), 1.0, atol=1e-6)
    assert info["target_variable"] == "x"
    assert info["n_estimates"] == 4
    assert not any("conditional expectation unsupported" in msg for msg in info["skipped_constraints"])

    all_marginals = builder.get_all_marginals(info)
    assert set(all_marginals.keys()) == {"x", "rain"}
