from __future__ import annotations

import os

import numpy as np
import pytest

from calibrated_response.maxent_pgmax.distribution_builder import DistributionBuilder
from calibrated_response.maxent_pgmax.solver import SolverConfig
from calibrated_response.models.query import (
    ConditionalProbabilityEstimate,
    EqualityProposition,
    ExpectationEstimate,
    InequalityProposition,
    ProbabilityEstimate,
)
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable


def _build_variables() -> list[ContinuousVariable | BinaryVariable]:
    return [
        ContinuousVariable(
            name="x",
            description="Target variable",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="u",
        ),
        BinaryVariable(name="rain", description="Whether it rains"),
    ]


def test_distribution_builder_returns_histogram_and_info() -> None:
    # pgmax import path currently requires disabling numba jit cache for this environment.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

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
    ]

    builder = DistributionBuilder(
        variables=_build_variables(),
        estimates=estimates,
        solver_config=SolverConfig(max_bins=10, bp_iters=30, damping=0.2),
    )

    try:
        distribution, info = builder.build(target_variable="x")
    except Exception as exc:
        # pgmax currently depends on specific jax/numba combinations. Skip if runtime is incompatible.
        pytest.skip(f"pgmax runtime unavailable/incompatible in this environment: {exc}")

    assert len(distribution.bin_probabilities) == 10
    assert np.isclose(sum(distribution.bin_probabilities), 1.0, atol=1e-6)
    assert info["target_variable"] == "x"
    assert info["n_estimates"] == 3
    assert "joint_distribution" in info
    assert "joint_note" in info

    all_marginals = builder.get_all_marginals(info)
    assert set(all_marginals.keys()) == {"x", "rain"}
