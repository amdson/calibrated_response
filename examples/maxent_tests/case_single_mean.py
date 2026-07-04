"""Test case: single variable with a non-trivial mean constraint.

MATHEMATICALLY OBVIOUS.

A single continuous variable X on [0, 100] with only a mean constraint
E[X] = 25.  The maximum-entropy distribution on a bounded interval with a
fixed mean is the truncated exponential family:

    p(x) ∝ exp(λx),  x ∈ [0, 100]

where λ < 0 (since 25 < 50 = midpoint) tilts mass toward 0.  The key
prediction is simply:

    E[X] ≈ 25

regardless of the exact shape of the learned distribution.  The distribution
is *not* symmetric (skewed toward the lower end), so std > 0 and mean ≠ 50.

There is no correlation to check because there is only one variable.
"""

import numpy as np

from calibrated_response.models.variable import ContinuousVariable
from calibrated_response.models.query import ExpectationEstimate

# ── Metadata ──────────────────────────────────────────────────────────────────

NAME = "single_mean"

DESCRIPTION = (
    "Single variable X on [0, 100] with only E[X] = 25 specified.\n"
    "MaxEnt tilts toward the lower end; the recovered mean must equal 25."
)

MATH_DESCRIPTION = (
    "The maximum-entropy distribution on [0, 100] with E[X] = 25 is a\n"
    "truncated exponential p(x) ∝ exp(λx) with λ < 0.  The solver must\n"
    "place the mean at exactly 25.  No correlation is expected (one variable)."
)

EXPECTED_RESULTS = {
    "E[X]": 25.0,
}

# ── Problem specification ──────────────────────────────────────────────────────

variables = [
    ContinuousVariable(
        name="X",
        description="Test variable with specified mean",
        lower_bound=0.0,
        upper_bound=100.0,
        unit="units",
    ),
]

estimates = [
    ExpectationEstimate(
        id="mean_X",
        variable="X",
        expected_value=25.0,
    ),
]

# ── Verification ───────────────────────────────────────────────────────────────

def check_results(samples_original: np.ndarray, variable_names: list) -> list:
    """Check recovered distribution against analytic expectations.

    Parameters
    ----------
    samples_original : (N, D) array in original variable domains.
    variable_names   : list of variable names (column order matches variables).

    Returns
    -------
    List of result dicts with keys: name, value, expected, tolerance, passed.
    """
    idx = variable_names.index("X")
    x = samples_original[:, idx]

    mean_x = float(np.mean(x))

    return [
        {
            "name": "E[X]",
            "value": round(mean_x, 2),
            "expected": 25.0,
            "tolerance": 4.0,
            "passed": abs(mean_x - 25.0) < 4.0,
            "note": "Mean must match the single specified constraint.",
        },
        {
            "name": "X is biased low (mean < 50)",
            "value": round(mean_x, 2),
            "expected": "< 50",
            "tolerance": None,
            "passed": mean_x < 50.0,
            "note": "Mean constraint below midpoint must shift distribution downward.",
        },
    ]
