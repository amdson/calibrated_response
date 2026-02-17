from __future__ import annotations

import numpy as np

from calibrated_response.maxent_large_v1.distribution_builder import DistributionBuilder
from calibrated_response.maxent_large_v1.training import SMMConfig
from calibrated_response.models.query import (
    ConditionalProbabilityEstimate,
    EqualityProposition,
    InequalityProposition,
)
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable


def test_two_condition_probability_is_supported() -> None:
    variables = [
        ContinuousVariable(
            name="x",
            description="Target",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="u",
        ),
        BinaryVariable(name="rain", description="Rain"),
        BinaryVariable(name="holiday", description="Holiday"),
    ]

    estimates = [
        ConditionalProbabilityEstimate(
            id="cp_two",
            proposition=InequalityProposition(
                variable="x",
                variable_type="continuous",
                threshold=0.7,
                is_lower_bound=True,
            ),
            conditions=[
                EqualityProposition(variable="rain", variable_type="binary", value=True),
                EqualityProposition(variable="holiday", variable_type="binary", value=True),
            ],
            probability=0.8,
        )
    ]

    builder = DistributionBuilder(
        variables=variables,
        estimates=estimates,
        solver_config=SMMConfig(
            num_iterations=8,
            num_chains=24,
            mcmc_steps_per_iteration=2,
            max_bins=10,
            seed=11,
        ),
    )

    distribution, info = builder.build(target_variable="x")

    assert len(distribution.bin_probabilities) == 10
    assert np.isclose(sum(distribution.bin_probabilities), 1.0, atol=1e-6)
    assert not any("only one-condition" in msg for msg in info["skipped_constraints"])
    assert info["n_factors"] >= 4  # unary factors + at least one higher-order conditional factor

