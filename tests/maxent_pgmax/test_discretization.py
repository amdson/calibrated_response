from __future__ import annotations

import numpy as np

from calibrated_response.maxent_pgmax.discretization import DomainDiscretizer
from calibrated_response.models.query import EqualityProposition, InequalityProposition
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable


def test_discretizer_builds_continuous_and_binary_buckets() -> None:
    variables = [
        ContinuousVariable(
            name="temperature",
            description="Daily high temperature",
            lower_bound=30.0,
            upper_bound=110.0,
            unit="F",
        ),
        BinaryVariable(name="is_raining", description="Whether it rains"),
    ]

    discretizer = DomainDiscretizer(variables, max_bins=10)

    temp_buckets = discretizer.buckets("temperature")
    rain_buckets = discretizer.buckets("is_raining")

    assert temp_buckets.n_states == 10
    assert np.isclose(temp_buckets.bin_edges[0], 30.0)
    assert np.isclose(temp_buckets.bin_edges[-1], 110.0)

    assert rain_buckets.n_states == 2
    assert np.allclose(rain_buckets.bin_edges, np.array([0.0, 0.5, 1.0]))


def test_proposition_masks_for_inequality_and_binary_equality() -> None:
    variables = [
        ContinuousVariable(
            name="x",
            description="Continuous variable",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="u",
        ),
        BinaryVariable(name="b", description="Binary variable"),
    ]
    discretizer = DomainDiscretizer(variables, max_bins=4)

    gt_mask = discretizer.proposition_mask(
        InequalityProposition(
            variable="x",
            variable_type="continuous",
            threshold=0.5,
            is_lower_bound=True,
        )
    )
    lt_mask = discretizer.proposition_mask(
        InequalityProposition(
            variable="x",
            variable_type="continuous",
            threshold=0.5,
            is_lower_bound=False,
        )
    )

    true_mask = discretizer.proposition_mask(
        EqualityProposition(
            variable="b",
            variable_type="binary",
            value=True,
        )
    )
    false_mask = discretizer.proposition_mask(
        EqualityProposition(
            variable="b",
            variable_type="binary",
            value=False,
        )
    )

    assert gt_mask.dtype == np.bool_
    assert lt_mask.dtype == np.bool_
    assert np.any(gt_mask)
    assert np.any(lt_mask)
    assert np.all(~(gt_mask & lt_mask))

    assert np.array_equal(true_mask, np.array([False, True]))
    assert np.array_equal(false_mask, np.array([True, False]))
