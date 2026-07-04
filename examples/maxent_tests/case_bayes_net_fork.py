"""Test case: fork Bayes net — A is a common cause of B and C.

Structure:   A → B
             A → C   (B and C are d-separated given A)

Each node's conditional distribution is pinned by *two* threshold constraints
(at different probability levels), mimicking a two-row CPT for each child.
This gives enough constraints that the MaxEnt joint closely approximates the
true Bayes net joint.

Variables
---------
A ∈ [0, 100]   "root driver"  (e.g. market conditions)
B ∈ [0, 100]   "effect 1"     (e.g. sales)
C ∈ [0, 100]   "effect 2"     (e.g. hiring)

Conditional probability table for B | A
-----------------------------------------
               A > 50          A < 50
  B > 60       0.72            0.12
  B > 25       0.95            0.45

Conditional probability table for C | A
-----------------------------------------
               A > 50          A < 50
  C > 55       0.75            0.10
  C > 20       0.95            0.50

With P(A > 50) = 0.40, the law of total probability gives:

  P(B > 60) = 0.72·0.40 + 0.12·0.60 = 0.360
  P(B > 25) = 0.95·0.40 + 0.45·0.60 = 0.650
  P(C > 55) = 0.75·0.40 + 0.10·0.60 = 0.360
  P(C > 20) = 0.95·0.40 + 0.50·0.60 = 0.680

Key Bayes-net property (checked qualitatively)
-----------------------------------------------
B and C are conditionally independent given A, so:

  Corr(B, C | A > 50) ≈ 0   (within the "A high" stratum)
  Corr(B, C | A < 50) ≈ 0   (within the "A low" stratum)

Yet the *marginal* correlation is positive:

  Corr(B, C) > 0

because A is a common cause — both B and C are high when A is high and low
when A is low.  This is the classic "explaining away" signature.  If the
solver correctly captures the Bayes net structure, the within-stratum
correlations should be noticeably smaller than the overall correlation.
"""

import numpy as np

from calibrated_response.models.variable import ContinuousVariable
from calibrated_response.models.query import (
    ProbabilityEstimate,
    ConditionalProbabilityEstimate,
    InequalityProposition,
)

# ── Metadata ──────────────────────────────────────────────────────────────────

NAME = "bayes_net_fork"

DESCRIPTION = (
    "Fork Bayes net: A → B and A → C.\n"
    "Two CPT rows per child tightly constrain each conditional distribution.\n"
    "Marginals are analytically computable; B and C should be marginally\n"
    "correlated but (approximately) conditionally independent given A."
)

MATH_DESCRIPTION = (
    "P(A>50)=0.40.  "
    "P(B>60)=0.72·0.40+0.12·0.60=0.360.  "
    "P(B>25)=0.95·0.40+0.45·0.60=0.650.\n"
    "P(C>55)=0.75·0.40+0.10·0.60=0.360.  "
    "P(C>20)=0.95·0.40+0.50·0.60=0.680.\n"
    "Corr(B,C)>0 overall (common cause); Corr(B,C|A) < Corr(B,C)."
)

EXPECTED_RESULTS = {
    "P(A > 50)": 0.40,
    "P(B > 60)": 0.360,
    "P(B > 25)": 0.650,
    "P(C > 55)": 0.360,
    "P(C > 20)": 0.680,
    "Corr(B, C)": "> 0",
    "Corr(B, C | A > 50)": "< Corr(B, C)",
}

# ── Problem specification ──────────────────────────────────────────────────────

variables = [
    ContinuousVariable(name="A", description="Root driver (e.g. market conditions)", lower_bound=0.0, upper_bound=100.0, unit="units"),
    ContinuousVariable(name="B", description="Effect 1 (e.g. sales)",                lower_bound=0.0, upper_bound=100.0, unit="units"),
    ContinuousVariable(name="C", description="Effect 2 (e.g. hiring)",               lower_bound=0.0, upper_bound=100.0, unit="units"),
]

estimates = [
    # ── Root node A ──────────────────────────────────────────────────────────
    ProbabilityEstimate(
        id="A_prior",
        proposition=InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=True),
        probability=0.40,
    ),

    # ── B | A  (upper row of CPT: high threshold) ────────────────────────────
    ConditionalProbabilityEstimate(
        id="B_high_given_A_high",
        proposition=InequalityProposition(variable="B", variable_type="continuous", threshold=60.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=True)],
        probability=0.72,
    ),
    ConditionalProbabilityEstimate(
        id="B_high_given_A_low",
        proposition=InequalityProposition(variable="B", variable_type="continuous", threshold=60.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=False)],
        probability=0.12,
    ),

    # ── B | A  (lower row of CPT: low threshold) ─────────────────────────────
    ConditionalProbabilityEstimate(
        id="B_mid_given_A_high",
        proposition=InequalityProposition(variable="B", variable_type="continuous", threshold=25.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=True)],
        probability=0.95,
    ),
    ConditionalProbabilityEstimate(
        id="B_mid_given_A_low",
        proposition=InequalityProposition(variable="B", variable_type="continuous", threshold=25.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=False)],
        probability=0.45,
    ),

    # ── C | A  (upper row of CPT: high threshold) ────────────────────────────
    ConditionalProbabilityEstimate(
        id="C_high_given_A_high",
        proposition=InequalityProposition(variable="C", variable_type="continuous", threshold=55.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=True)],
        probability=0.75,
    ),
    ConditionalProbabilityEstimate(
        id="C_high_given_A_low",
        proposition=InequalityProposition(variable="C", variable_type="continuous", threshold=55.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=False)],
        probability=0.10,
    ),

    # ── C | A  (lower row of CPT: low threshold) ─────────────────────────────
    ConditionalProbabilityEstimate(
        id="C_mid_given_A_high",
        proposition=InequalityProposition(variable="C", variable_type="continuous", threshold=20.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=True)],
        probability=0.95,
    ),
    ConditionalProbabilityEstimate(
        id="C_mid_given_A_low",
        proposition=InequalityProposition(variable="C", variable_type="continuous", threshold=20.0, is_lower_bound=True),
        conditions=[InequalityProposition(variable="A", variable_type="continuous", threshold=50.0, is_lower_bound=False)],
        probability=0.50,
    ),
]

# ── Analytic expectations ──────────────────────────────────────────────────────

_P_A_HIGH = 0.40
_P_A_LOW  = 1 - _P_A_HIGH

_P_B60 = 0.72 * _P_A_HIGH + 0.12 * _P_A_LOW   # = 0.360
_P_B25 = 0.95 * _P_A_HIGH + 0.45 * _P_A_LOW   # = 0.650
_P_C55 = 0.75 * _P_A_HIGH + 0.10 * _P_A_LOW   # = 0.360
_P_C20 = 0.95 * _P_A_HIGH + 0.50 * _P_A_LOW   # = 0.680

# ── Verification ───────────────────────────────────────────────────────────────

def check_results(samples_original: np.ndarray, variable_names: list) -> list:
    """Check recovered distribution against analytic Bayes net expectations.

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

    p_a_high = float(np.mean(a > 50))
    p_b60    = float(np.mean(b > 60))
    p_b25    = float(np.mean(b > 25))
    p_c55    = float(np.mean(c > 55))
    p_c20    = float(np.mean(c > 20))

    corr_bc          = float(np.corrcoef(b, c)[0, 1])

    # Within-stratum (conditional) correlations
    mask_a_high = a > 50
    mask_a_low  = a <= 50

    if mask_a_high.sum() >= 30:
        corr_bc_given_high = float(np.corrcoef(b[mask_a_high], c[mask_a_high])[0, 1])
    else:
        corr_bc_given_high = np.nan

    if mask_a_low.sum() >= 30:
        corr_bc_given_low  = float(np.corrcoef(b[mask_a_low],  c[mask_a_low])[0, 1])
    else:
        corr_bc_given_low = np.nan

    tol_prob = 0.07   # ±7 pp on marginal probabilities

    results = [
        {
            "name": "P(A > 50)",
            "value": round(p_a_high, 3),
            "expected": _P_A_HIGH,
            "tolerance": tol_prob,
            "passed": abs(p_a_high - _P_A_HIGH) < tol_prob,
            "note": "Directly specified root node prior.",
        },
        {
            "name": "P(B > 60)  [analytic: 0.360]",
            "value": round(p_b60, 3),
            "expected": round(_P_B60, 3),
            "tolerance": tol_prob,
            "passed": abs(p_b60 - _P_B60) < tol_prob,
            "note": "Law of total prob: 0.72·P(A>50) + 0.12·P(A<50).",
        },
        {
            "name": "P(B > 25)  [analytic: 0.650]",
            "value": round(p_b25, 3),
            "expected": round(_P_B25, 3),
            "tolerance": tol_prob,
            "passed": abs(p_b25 - _P_B25) < tol_prob,
            "note": "Law of total prob: 0.95·P(A>50) + 0.45·P(A<50).",
        },
        {
            "name": "P(C > 55)  [analytic: 0.360]",
            "value": round(p_c55, 3),
            "expected": round(_P_C55, 3),
            "tolerance": tol_prob,
            "passed": abs(p_c55 - _P_C55) < tol_prob,
            "note": "Law of total prob: 0.75·P(A>50) + 0.10·P(A<50).",
        },
        {
            "name": "P(C > 20)  [analytic: 0.680]",
            "value": round(p_c20, 3),
            "expected": round(_P_C20, 3),
            "tolerance": tol_prob,
            "passed": abs(p_c20 - _P_C20) < tol_prob,
            "note": "Law of total prob: 0.95·P(A>50) + 0.50·P(A<50).",
        },
        {
            "name": "Corr(B, C) > 0  [common cause]",
            "value": round(corr_bc, 3),
            "expected": "> 0",
            "tolerance": None,
            "passed": corr_bc > 0.05,
            "note": "A is a common cause of B and C → marginal positive correlation.",
        },
        {
            "name": "Corr(B,C|A>50) < Corr(B,C)  [conditional indep.]",
            "value": round(corr_bc_given_high, 3) if not np.isnan(corr_bc_given_high) else None,
            "expected": f"< {round(corr_bc, 3)}",
            "tolerance": None,
            "passed": (
                (not np.isnan(corr_bc_given_high))
                and corr_bc_given_high < corr_bc
            ),
            "note": (
                "In a fork Bayes net, conditioning on A makes B and C less correlated. "
                "Perfect conditional independence would give 0."
            ),
        },
        {
            "name": "Corr(B,C|A<50) < Corr(B,C)  [conditional indep.]",
            "value": round(corr_bc_given_low, 3) if not np.isnan(corr_bc_given_low) else None,
            "expected": f"< {round(corr_bc, 3)}",
            "tolerance": None,
            "passed": (
                (not np.isnan(corr_bc_given_low))
                and corr_bc_given_low < corr_bc
            ),
            "note": (
                "In a fork Bayes net, conditioning on A makes B and C less correlated."
            ),
        },
    ]

    return results
