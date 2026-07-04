"""Test case: law of total expectation with a binary conditioning variable.

A binary variable Z (True/False) conditions a continuous variable X ∈ [0, 100].
The specified constraints are:

    P(Z = True)       = 0.30
    E[X | Z = True]   = 80
    E[X | Z = False]  = 20

By the law of total expectation the *unconditional* mean is determined exactly:

    E[X] = P(Z=True)·E[X|Z=True] + P(Z=False)·E[X|Z=False]
         = 0.30·80 + 0.70·20
         = 24 + 14 = 38

This is the main analytically-obvious prediction: no matter what shape the
conditional distributions take, E[X] must equal 38.

Additionally, the conditional means should be reproduced:

    E[X | Z=True]  ≈ 80   (samples where Z > 0.5)
    E[X | Z=False] ≈ 20   (samples where Z ≤ 0.5)

The result is a bimodal marginal for X with one mode near 20 and another
near 80.  The two modes are separated by the binary label Z.

Note on binary variables in samples_original:
    Binary variables are stored in [0, 1]; values > 0.5 represent True.
"""

import numpy as np

from calibrated_response.models.variable import ContinuousVariable, BinaryVariable
from calibrated_response.models.query import (
    ProbabilityEstimate,
    ConditionalExpectationEstimate,
    EqualityProposition,
)

# ── Metadata ──────────────────────────────────────────────────────────────────

NAME = "conditional_expectation"

DESCRIPTION = (
    "Binary Z with P(Z=True)=0.30, and continuous X on [0,100] with\n"
    "E[X|Z=True]=80 and E[X|Z=False]=20.\n"
    "Law of total expectation → E[X] = 0.30·80 + 0.70·20 = 38."
)

MATH_DESCRIPTION = (
    "E[X] = P(Z=T)·E[X|Z=T] + P(Z=F)·E[X|Z=F] = 0.30·80 + 0.70·20 = 38.\n"
    "Marginal of X should be bimodal with modes near 20 and 80."
)

EXPECTED_RESULTS = {
    "P(Z = True)": 0.30,
    "E[X | Z = True]": 80.0,
    "E[X | Z = False]": 20.0,
    "E[X]": 38.0,
}

# ── Problem specification ──────────────────────────────────────────────────────

variables = [
    BinaryVariable(
        name="Z",
        description="Binary conditioning variable (True/False)",
    ),
    ContinuousVariable(
        name="X",
        description="Continuous outcome variable",
        lower_bound=0.0,
        upper_bound=100.0,
        unit="units",
    ),
]

estimates = [
    # Marginal probability of Z = True
    ProbabilityEstimate(
        id="Z_prior",
        proposition=EqualityProposition(variable="Z", variable_type="binary", value=True),
        probability=0.30,
    ),
    # Conditional mean of X given Z = True
    ConditionalExpectationEstimate(
        id="E_X_given_Z_true",
        variable="X",
        conditions=[EqualityProposition(variable="Z", variable_type="binary", value=True)],
        expected_value=80.0,
    ),
    # Conditional mean of X given Z = False
    ConditionalExpectationEstimate(
        id="E_X_given_Z_false",
        variable="X",
        conditions=[EqualityProposition(variable="Z", variable_type="binary", value=False)],
        expected_value=20.0,
    ),
]

# ── Verification ───────────────────────────────────────────────────────────────

# Analytic prediction from law of total expectation
_P_Z_TRUE = 0.30
_E_X_GIVEN_TRUE = 80.0
_E_X_GIVEN_FALSE = 20.0
_E_X = _P_Z_TRUE * _E_X_GIVEN_TRUE + (1 - _P_Z_TRUE) * _E_X_GIVEN_FALSE  # = 38.0


def check_results(samples_original: np.ndarray, variable_names: list) -> list:
    """Check recovered distribution against analytic expectations.

    Parameters
    ----------
    samples_original : (N, D) array.
        Binary variables are in [0, 1] (>0.5 ≡ True).
        Continuous variables are in their original domains.
    variable_names   : list of variable names (column order matches variables).

    Returns
    -------
    List of result dicts with keys: name, value, expected, tolerance, passed.
    """
    zi = variable_names.index("Z")
    xi = variable_names.index("X")

    z = samples_original[:, zi]   # [0, 1]; > 0.5 → True
    x = samples_original[:, xi]   # original domain [0, 100]

    z_true_mask = z > 0.5
    z_false_mask = ~z_true_mask

    p_z_true = float(np.mean(z_true_mask))
    e_x = float(np.mean(x))

    results = [
        {
            "name": "P(Z = True)",
            "value": round(p_z_true, 3),
            "expected": _P_Z_TRUE,
            "tolerance": 0.05,
            "passed": abs(p_z_true - _P_Z_TRUE) < 0.05,
            "note": "Directly specified marginal probability of Z.",
        },
        {
            "name": "E[X]  (law of total expectation)",
            "value": round(e_x, 2),
            "expected": _E_X,
            "tolerance": 5.0,
            "passed": abs(e_x - _E_X) < 5.0,
            "note": f"Analytic: {_P_Z_TRUE}·{_E_X_GIVEN_TRUE} + {1-_P_Z_TRUE}·{_E_X_GIVEN_FALSE} = {_E_X}.",
        },
    ]

    # Conditional means (only check if enough samples in each group)
    if z_true_mask.sum() >= 30:
        e_x_true = float(np.mean(x[z_true_mask]))
        results.append({
            "name": "E[X | Z = True]",
            "value": round(e_x_true, 2),
            "expected": _E_X_GIVEN_TRUE,
            "tolerance": 8.0,
            "passed": abs(e_x_true - _E_X_GIVEN_TRUE) < 8.0,
            "note": "Directly specified conditional mean.",
        })
    else:
        results.append({
            "name": "E[X | Z = True]",
            "value": None,
            "expected": _E_X_GIVEN_TRUE,
            "tolerance": 8.0,
            "passed": False,
            "note": f"Too few Z=True samples ({z_true_mask.sum()}) to estimate conditional mean.",
        })

    if z_false_mask.sum() >= 30:
        e_x_false = float(np.mean(x[z_false_mask]))
        results.append({
            "name": "E[X | Z = False]",
            "value": round(e_x_false, 2),
            "expected": _E_X_GIVEN_FALSE,
            "tolerance": 8.0,
            "passed": abs(e_x_false - _E_X_GIVEN_FALSE) < 8.0,
            "note": "Directly specified conditional mean.",
        })
    else:
        results.append({
            "name": "E[X | Z = False]",
            "value": None,
            "expected": _E_X_GIVEN_FALSE,
            "tolerance": 8.0,
            "passed": False,
            "note": f"Too few Z=False samples ({z_false_mask.sum()}) to estimate conditional mean.",
        })

    return results
