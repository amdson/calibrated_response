from calibrated_response.models.variable import (DiscreteVariable, ContinuousVariable, BinaryVariable,
                                                 VariableType)
from calibrated_response.models.query import (
    EqualityProposition, InequalityProposition,
    ProbabilityEstimate, ExpectationEstimate,
    ConditionalProbabilityEstimate, ConditionalExpectationEstimate,
)
import json
from calibrated_response.models.demo import (
    demo_continuous_var,
    demo_binary_var,
    demo_equality_prop,
    demo_inequality_prop_greater,
    demo_inequality_prop_less,
    demo_probability_estimate,
    demo_expectation_estimate,
    demo_conditional_prob_estimate,
    demo_conditional_exp_estimate,)

"""Prompt templates for LLM generation tasks."""
# Pre-compute JSON strings for use in prompts (avoids f-string format specifier issues)
demo_continuous_var_json = json.dumps(demo_continuous_var.model_dump(), indent=4).replace('{', '{{').replace('}', '}}')
demo_binary_var_json = json.dumps(demo_binary_var.model_dump(), indent=4).replace('{', '{{').replace('}', '}}')
demo_prob_json = json.dumps(demo_probability_estimate.model_dump(), indent=6).replace('{', '{{').replace('}', '}}')
demo_exp_json = json.dumps(demo_expectation_estimate.model_dump(), indent=6).replace('{', '{{').replace('}', '}}')
demo_cond_prob_json = json.dumps(demo_conditional_prob_estimate.model_dump(), indent=6).replace('{', '{{').replace('}', '}}')
demo_cond_exp_json = json.dumps(demo_conditional_exp_estimate.model_dump(), indent=6).replace('{', '{{').replace('}', '}}')

PROMPTS = {
    "variable_generation": {
        "system": """You are an expert forecaster helping to decompose complex prediction questions.
Your task is to identify relevant variables that could influence the answer to a forecasting question.
Focus on variables that are:
1. Measurable or estimable
2. Potentially influential on the outcome
3. Not perfectly correlated with each other
4. A mix of easy-to-know facts and uncertain quantities

For continuous variables, always provide reasonable lower and upper bounds for the plausible range and units. 

CRITICAL RULES:
- Variable names must be short (2-4 words, no spaces preferred, use underscores)
- Variable descriptions should be clear and concise
- Variable types must be binary or continuous
- The first variable should be a literal answer to the main question (e.g., "Will it rain tomorrow?" -> variable: "will_rain_tomorrow", type: binary)
""",
        
        "user": """Question to forecast: {question}

Identify {n_variables} relevant variables that could help predict the answer to this question.
For each variable, provide:
- A short name (2-4 words, no spaces preferred, use underscores)
- A description of what it represents  
- Whether it's binary (yes/no), continuous (numeric), or discrete (categories)
- An estimated importance from 0 to 1
- For CONTINUOUS variables: lower_bound, upper_bound (plausible range), and unit (e.g., "inches", "dollars", "people")
- For BINARY variables: yes_label and no_label (e.g., "raining", "not raining")

Respond in JSON format. E.g.:
{{{{
  "variables": [
    {demo_continuous_var_json},
    {demo_binary_var_json}
  ]
}}}}""".format(question="{question}", n_variables="{n_variables}", 
            demo_continuous_var_json=demo_continuous_var_json, 
            demo_binary_var_json=demo_binary_var_json)
    }, 

    "estimate_generation": {
        "system": """You are an expert forecaster providing calibrated probability and expectation estimates.
Your task is to generate structured estimates about relationships between variables for a forecasting problem.

You MUST generate four types of estimates:

1. PROBABILITY ESTIMATES: P(proposition) - probability that a proposition is true
   - For binary variables: P(X = true) or P(X = false)
   - For continuous variables: P(X > threshold) or P(X < threshold)
   - Probabilities must be between 0 and 1

2. EXPECTATION ESTIMATES: E[X] - expected value of a continuous variable
   - Must be within the variable's plausible range
   - Only for continuous variables

3. CONDITIONAL PROBABILITY ESTIMATES: P(proposition | conditions) 
   - Probability of a proposition given one or more conditions
   - Conditions can be on different variables than the target
   - Use conditions that provide meaningful information

4. CONDITIONAL EXPECTATION ESTIMATES: E[X | conditions]
   - Expected value of a variable given conditions on OTHER variables
   - Shows how expectations change under different scenarios

CRITICAL RULES:
- All variable names must EXACTLY match the provided variable names
- All thresholds must be within the variable's plausible range
- Probabilities must be between 0 and 1
- Expected values must be realistic given the variable's range
- Condition variables must be DIFFERENT from the target variable
- Include a good mix of all four estimate types
- Estimates should be calibrated and reflect genuine uncertainty""",
        
        "user": """Question to forecast: {question}

AVAILABLE VARIABLES:
{variables}

Generate {num_estimates} estimates that capture the joint distribution over these variables. 
Make sure your estimates are plausible based on your knowledge and common sense. 
Include a mix of probability, expectation, conditional probability, and conditional expectation estimates, 
and include one direct prediction of the main question as a probability or expectation estimate. 


IMPORTANT:
- All variable names must EXACTLY match the variable names above
- Use EqualityProposition for binary variables (value is true or false)
- Use InequalityProposition for continuous variables (threshold with greater=true or greater=false)
- variable_type must be "binary" for EqualityProposition and "continuous" for InequalityProposition
- Condition variables must be different from the target

Respond with a JSON object. E.g. :
{{{{ 
  "estimates": [
    {demo_prob_json},
    {demo_exp_json},
    {demo_cond_prob_json},
    {demo_cond_exp_json}
  ]
}}}}""".format(question="{question}", variables="{variables}", num_estimates="{num_estimates}",
            demo_prob_json=demo_prob_json, demo_exp_json=demo_exp_json,
            demo_cond_prob_json=demo_cond_prob_json, demo_cond_exp_json=demo_cond_exp_json)
    },
    
    "natural_estimate_generation": {
        "system": """You are an expert forecaster providing calibrated probability and expectation estimates.
Output estimates using concise mathematical notation with brief reasoning.

ESTIMATE FORMATS:
- Probability: P(variable > threshold) = value  or  P(variable = True) = value
- Expectation: E[variable] = value
- Conditional Probability: P(variable > threshold | condition) = value
- Conditional Expectation: E[variable | condition] = value

RULES:
- Use P(...) for probabilities, E[...] for expectations
- Use parentheses () for P, square brackets [] for E
- Conditions come after | (pipe symbol)
- Multiple conditions separated by commas: P(X > 5 | Y = True, Z > 10) = 0.3
- Probabilities must be between 0 and 1
- For binary variables use = True or = False
- For continuous variables use > or < with thresholds
- Variable names must exactly match those provided
- Include a brief "logic" explanation for each estimate""",
        
        "user": """Question to forecast: {question}

AVAILABLE VARIABLES:
{variables}

Generate {num_estimates} estimates. Include a mix of:
- Unconditional probabilities: P(var > threshold) = value
- Unconditional expectations: E[var] = value  
- Conditional probabilities: P(var > threshold | other_var > value) = prob
- Conditional expectations: E[var | other_var = True] = value

For each estimate, include a brief "logic" field explaining your reasoning. 

IMPORTANT:
- All variable names must EXACTLY match the variable names above
- Use > or < for continuous variables, = True/False for binary variables
- Include a variety of estimates that capture different relationships between the variables, not just direct predictions of the main question.

Respond with JSON:
{{
  "estimates": [
    {{"logic": "Based on current market trends and growth rate", "expression": "P(variable_name > 50.0) = 0.3"}},
    {{"logic": "Historical average adjusted for inflation", "expression": "E[variable_name] = 75.0"}},
    {{"logic": "Strong correlation observed between these variables", "expression": "P(var1 > 10 | var2 = True) = 0.6"}},
    {{"logic": "Conditional mean shifts upward when var2 is high", "expression": "E[var1 | var2 > 5.0] = 25.0"}}
  ]
}}"""
    },
}


def format_variables_for_prompt(variables: list[dict]) -> str:
    """Format a list of variables for inclusion in a prompt.
    
    Includes variable name, type, description, and plausible range if available.
    """
    lines = []
    for i, var in enumerate(variables, 1):
        name = var.get('name', f'Variable {i}')
        desc = var.get('description', 'No description')
        vtype = var.get('type', 'unknown')
        
        # Build range string if bounds are available
        lower = var.get('lower_bound')
        upper = var.get('upper_bound')
        unit = var.get('unit', '')
        
        range_str = ""
        if lower is not None or upper is not None:
            lower_str = str(lower) if lower is not None else "?"
            upper_str = str(upper) if upper is not None else "?"
            range_str = f" [range: {lower_str} to {upper_str}"
            if unit:
                range_str += f" {unit}"
            range_str += "]"
        elif vtype == 'binary':
            range_str = " [values: 0 or 1]"
        
        lines.append(f"{i}. {name} ({vtype}): {desc}{range_str}")
    return "\n".join(lines)
