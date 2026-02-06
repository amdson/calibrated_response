from calibrated_response.models.variable import ContinuousVariable, BinaryVariable, VariableType
from calibrated_response.models.query import (EqualityProposition, InequalityProposition, 
                                              ProbabilityEstimate, ExpectationEstimate, 
                                              ConditionalProbabilityEstimate, ConditionalExpectationEstimate)

demo_continuous_var = ContinuousVariable(
    name="daily_high_temp",
    description="The daily high temperature in degrees Fahrenheit",
    lower_bound=30.0,
    upper_bound=100.0,
    unit="degrees F"
) 

demo_binary_var = BinaryVariable(
    name="is_raining",
    description="Whether it is currently raining"
)

# ============================================================================
# Proposition Examples
# ============================================================================

# EqualityProposition: For binary variables (X = value)
demo_equality_prop = EqualityProposition(
    variable="is_raining",
    value=True
)

# InequalityProposition: For continuous variables (X > threshold or X < threshold)
demo_inequality_prop_greater = InequalityProposition(
    variable="daily_high_temp",
    variable_type=VariableType.CONTINUOUS,
    threshold=80.0,
    greater=True  # daily_high_temp > 80
)

demo_inequality_prop_less = InequalityProposition(
    variable="daily_high_temp",
    variable_type=VariableType.CONTINUOUS,
    threshold=50.0,
    greater=False  # daily_high_temp < 50
)

# ============================================================================
# Estimate Examples
# ============================================================================

# ProbabilityEstimate: P(X) - probability of a proposition
demo_probability_estimate = ProbabilityEstimate(
    id="prob_rain",
    proposition=demo_equality_prop,
    probability=0.3  # P(is_raining = True) = 0.3
)

demo_probability_estimate_threshold = ProbabilityEstimate(
    id="prob_hot_day",
    proposition=demo_inequality_prop_greater,
    probability=0.45  # P(daily_high_temp > 80) = 0.45
)

# ExpectationEstimate: E[X] - expected value of a variable
demo_expectation_estimate = ExpectationEstimate(
    id="expected_temp",
    variable="daily_high_temp",
    expected_value=72.5  # E[daily_high_temp] = 72.5
)

# ConditionalProbabilityEstimate: P(X | condition)
demo_conditional_prob_estimate = ConditionalProbabilityEstimate(
    id="prob_hot_given_no_rain",
    proposition=demo_inequality_prop_greater,  # daily_high_temp > 80
    conditions=[EqualityProposition(variable="is_raining", value=False)],
    probability=0.6  # P(daily_high_temp > 80 | is_raining = False) = 0.6
)

# ConditionalExpectationEstimate: E[X | condition]
demo_conditional_exp_estimate = ConditionalExpectationEstimate(
    id="expected_temp_given_rain",
    variable="daily_high_temp",
    conditions=[EqualityProposition(variable="is_raining", value=True)],
    expected_value=65.0  # E[daily_high_temp | is_raining = True] = 65.0
)