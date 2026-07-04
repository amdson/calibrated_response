"""Test case: Markov chain with analytically computable marginals.

Three variables A → B → C on [0, 100] are linked by conditional probability
estimates.  The marginal probability of each variable exceeding 50 can be
computed exactly by the law of total probability:

    P(A > 50) = 0.70   (specified directly)

    P(B > 50) = P(B>50 | A>50)·P(A>50) + P(B>50 | A<50)·P(A<50)
              = 0.80·0.70 + 0.10·0.30
              = 0.56 + 0.03 = 0.59

    P(C > 50) = P(C>50 | B>50)·P(B>50) + P(C>50 | B<50)·P(B<50)
              = 0.80·0.59 + 0.10·0.41
              = 0.472 + 0.041 ≈ 0.513

The conditional probabilities also imply a positive Pearson correlation
between all pairs (A,B), (B,C), and (A,C).

Note: the complementary conditionals use is_lower_bound=False to specify
      P(B < 50 | A < 50) = 0.90, which implicitly sets P(B > 50 | A < 50) = 0.10.
"""

import numpy as np

from calibrated_response.models.variable import ContinuousVariable
from calibrated_response.models.query import (
    ProbabilityEstimate,
    ConditionalProbabilityEstimate,
    InequalityProposition,
)

# ── Metadata ──────────────────────────────────────────────────────────────────

NAME = "chain_propagation"

DESCRIPTION = (
    "Three variables A, B, C on [0, 100] in a Markov chain.\n"
    "Conditional probabilities allow analytic computation of marginals via\n"
    "the law of total probability."
)

MATH_DESCRIPTION = (
    "P(A>50)=0.70 (given).\n"
    "P(B>50) = 0.80·0.70 + 0.10·0.30 = 0.59.\n"
    "P(C>50) = 0.80·0.59 + 0.10·0.41 ≈ 0.513.\n"
    "All pairs should be positively correlated."
)

EXPECTED_RESULTS = {
    "P(A > 50)": 0.70,
    "P(B > 50)": 0.59,
    "P(C > 50)": 0.513,
}

# ── Problem specification ──────────────────────────────────────────────────────

variables = [
    ContinuousVariable(name="A", description="Chain variable A", lower_bound=0.0, upper_bound=100.0, unit="%"),
    ContinuousVariable(name="B", description="Chain variable B", lower_bound=0.0, upper_bound=100.0, unit="%"),
    ContinuousVariable(name="C", description="Chain variable C", lower_bound=0.0, upper_bound=100.0, unit="%"),
]

estimates = [
    # Marginal prior on A
    ProbabilityEstimate(
        id="A_prior",
        proposition=InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=True),
        probability=0.70,
    ),
    # P(B > 50 | A > 50) = 0.80
    ConditionalProbabilityEstimate(
        id="B_given_A_high",
        proposition=InequalityProposition(variable="B", variable_type="continuous", threshold=50.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=True)],
        probability=0.80,
    ),
    # P(B < 50 | A < 50) = 0.90  →  P(B > 50 | A < 50) = 0.10
    ConditionalProbabilityEstimate(
        id="B_given_A_low",
        proposition=InequalityProposition(variable="B", variable_type="continuous", threshold=50.0, is_lower_bound=False),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=False)],
        probability=0.90,
    ),
    # P(C > 50 | B > 50) = 0.80
    ConditionalProbabilityEstimate(
        id="C_given_B_high",
        proposition=InequalityProposition(variable="C", variable_type="continuous", threshold=50.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="B", variable_type="continuous", threshold=50.0, is_lower_bound=True)],
        probability=0.80,
    ),
    # P(C < 50 | B < 50) = 0.90  →  P(C > 50 | B < 50) = 0.10
    ConditionalProbabilityEstimate(
        id="C_given_B_low",
        proposition=InequalityProposition(variable="C", variable_type="continuous", threshold=50.0, is_lower_bound=False),
        conditions=[InequalityProposition(variable="B", variable_type="continuous", threshold=50.0, is_lower_bound=False)],
        probability=0.90,
    ),
]

# ── Verification ───────────────────────────────────────────────────────────────

# Analytic expectations (from law of total probability)
_P_A = 0.70
_P_B = 0.80 * _P_A + 0.10 * (1 - _P_A)   # = 0.59
_P_C = 0.80 * _P_B + 0.10 * (1 - _P_B)   # ≈ 0.513


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
    ai = variable_names.index("A")
    bi = variable_names.index("B")
    ci = variable_names.index("C")

    a = samples_original[:, ai]
    b = samples_original[:, bi]
    c = samples_original[:, ci]

    p_a = float(np.mean(a > 50))
    p_b = float(np.mean(b > 50))
    p_c = float(np.mean(c > 50))

    corr_ab = float(np.corrcoef(a, b)[0, 1])
    corr_bc = float(np.corrcoef(b, c)[0, 1])

    tol = 0.06   # allow ±6 pp error on marginal probabilities

    return [
        {
            "name": "P(A > 50)",
            "value": round(p_a, 3),
            "expected": round(_P_A, 3),
            "tolerance": tol,
            "passed": abs(p_a - _P_A) < tol,
            "note": "Directly specified marginal prior on A.",
        },
        {
            "name": "P(B > 50)",
            "value": round(p_b, 3),
            "expected": round(_P_B, 3),
            "tolerance": tol,
            "passed": abs(p_b - _P_B) < tol,
            "note": "Analytic: 0.80·P(A>50) + 0.10·P(A<50) = 0.59.",
        },
        {
            "name": "P(C > 50)",
            "value": round(p_c, 3),
            "expected": round(_P_C, 3),
            "tolerance": tol,
            "passed": abs(p_c - _P_C) < tol,
            "note": "Analytic: 0.80·P(B>50) + 0.10·P(B<50) ≈ 0.513.",
        },
        {
            "name": "Corr(A, B) > 0",
            "value": round(corr_ab, 3),
            "expected": "> 0",
            "tolerance": None,
            "passed": corr_ab > 0,
            "note": "A high → B high, so positive correlation is expected.",
        },
        {
            "name": "Corr(B, C) > 0",
            "value": round(corr_bc, 3),
            "expected": "> 0",
            "tolerance": None,
            "passed": corr_bc > 0,
            "note": "B high → C high, so positive correlation is expected.",
        },
    ]
