"""Test case: two variables with only independent marginal constraints.

MATHEMATICALLY OBVIOUS.

Two continuous variables X ∈ [0, 100] and Y ∈ [0, 100] with only separate
mean constraints (no joint constraint).  By the MaxEnt principle, the joint
distribution maximising entropy subject to only *marginal* moment constraints
is the *product* of the marginals:

    p*(x, y) = p*(x) · p*(y)

This is the formal justification for statistical independence in the absence
of any cross-variable information.  The predictions are:

    E[X] ≈ 25        (specified)
    E[Y] ≈ 75        (specified)
    Corr(X, Y) ≈ 0   (no joint information → MaxEnt enforces independence)

The zero-correlation result is the mathematically obvious consequence.  It
also makes this a useful *sanity check*: if the solver spuriously introduces
correlation between unrelated variables the test will fail.
"""

import numpy as np

from calibrated_response.models.variable import ContinuousVariable
from calibrated_response.models.query import ExpectationEstimate

# ── Metadata ──────────────────────────────────────────────────────────────────

NAME = "independent_marginals"

DESCRIPTION = (
    "Two variables X ∈ [0, 100] and Y ∈ [0, 100] with only marginal mean\n"
    "constraints E[X]=25 and E[Y]=75 and no joint constraints.\n"
    "MaxEnt → product distribution → Corr(X,Y) ≈ 0."
)

MATH_DESCRIPTION = (
    "MaxEnt with only marginal constraints factorises: p*(x,y) = p*(x)·p*(y).\n"
    "Expected: E[X] = 25, E[Y] = 75, Pearson correlation ≈ 0."
)

EXPECTED_RESULTS = {
    "E[X]": 25.0,
    "E[Y]": 75.0,
    "Corr(X,Y)": 0.0,
}

# ── Problem specification ──────────────────────────────────────────────────────

variables = [
    ContinuousVariable(
        name="X",
        description="First independent variable",
        lower_bound=0.0,
        upper_bound=100.0,
        unit="units",
    ),
    ContinuousVariable(
        name="Y",
        description="Second independent variable",
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
    ExpectationEstimate(
        id="mean_Y",
        variable="Y",
        expected_value=75.0,
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
    xi = variable_names.index("X")
    yi = variable_names.index("Y")

    x = samples_original[:, xi]
    y = samples_original[:, yi]

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))

    # Pearson correlation
    corr = float(np.corrcoef(x, y)[0, 1])

    return [
        {
            "name": "E[X]",
            "value": round(mean_x, 2),
            "expected": 25.0,
            "tolerance": 4.0,
            "passed": abs(mean_x - 25.0) < 4.0,
            "note": "Mean of X must match its marginal constraint.",
        },
        {
            "name": "E[Y]",
            "value": round(mean_y, 2),
            "expected": 75.0,
            "tolerance": 4.0,
            "passed": abs(mean_y - 75.0) < 4.0,
            "note": "Mean of Y must match its marginal constraint.",
        },
        {
            "name": "Corr(X, Y)",
            "value": round(corr, 3),
            "expected": 0.0,
            "tolerance": 0.15,
            "passed": abs(corr) < 0.15,
            "note": (
                "No joint constraints → MaxEnt gives independent marginals. "
                "Non-zero correlation would indicate solver artefact."
            ),
        },
    ]
