from calibrated_response.models.variable import (DiscreteVariable, ContinuousVariable, BinaryVariable,
                                                 demo_binary_var, demo_continuous_var)
import json

"""Prompt templates for LLM generation tasks."""

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
For binary variables, provide clear yes/no labels in the description.""",
        
        "user": f"""Question to forecast: {{question}}

Identify {{n_variables}} relevant variables that could help predict the answer to this question.
For each variable, provide:
- A short name (2-4 words, no spaces preferred, use underscores)
- A description of what it represents  
- Whether it's binary (yes/no), continuous (numeric), or discrete (categories)
- An estimated importance from 0 to 1
- For CONTINUOUS variables: lower_bound, upper_bound (plausible range), and unit (e.g., "inches", "dollars", "people")
- For BINARY variables: yes_label and no_label (e.g., "raining", "not raining")

Respond in JSON format:
{
  "variables": [
    {json.dumps(demo_continuous_var.model_dump())},
    {json.dumps(demo_binary_var.model_dump())}
  ]
}"""
    },  
    
    "query_generation": {
        "system": """You are an expert forecaster designing queries to elicit probability distributions.
Your task is to generate specific, answerable questions about variables related to a forecasting problem.

You MUST generate exactly four types of queries that map to maximum entropy constraints:

1. PROBABILITY queries: Ask for the probability of a variable exceeding (or being below) a specific threshold
   - Example: "What is the probability that Snow Depth exceeds 6 inches?"
   - Example: "What is the probability that Stock Return is below -5%?"
   - Use thresholds within the variable's plausible range

2. EXPECTATION queries: Ask for the expected/mean value of a variable
   - Example: "What is the expected value of Snow Depth?"

3. CONDITIONAL PROBABILITY queries: Ask for a threshold probability given conditions on OTHER variables
   - Conditions use actual threshold values (not percentiles)
   - Example: "Given that Yesterday's Snowfall was above 3 inches, what is the probability that Snow Depth exceeds 10 inches?"
   - Example: "Given that Temperature is below 25°F, what is the probability that Snow Depth exceeds 8 inches?"

4. CONDITIONAL EXPECTATION queries: Ask for expected value given threshold conditions on OTHER variables
   - Example: "Given that Yesterday's Snowfall was above 2 inches and Temperature is below 30°F, what is the expected Snow Depth?"

CRITICAL RULES:
- target_variable MUST be one of the exact variable names provided (case-sensitive)
- condition variables MUST be different from the target_variable
- condition variables MUST be exact variable names from the list provided
- ALL threshold values MUST be within the plausible range of the variable
- Use a variety of thresholds spread across each variable's range
- Include a mix of all four query types for diverse information
""",
        "user": """Main forecasting question: {question}

AVAILABLE VARIABLES (use these exact names, respecting the given ranges):
{variables}

Generate {n_queries} queries to help estimate the answer. 
Include a mix of probability, expectation, conditional_probability, and conditional_expectation queries.

IMPORTANT: 
- All target_variable values MUST match one of the variable names above EXACTLY
- All condition variable names MUST match one of the variable names above EXACTLY  
- All threshold values MUST be within the variable's plausible range
- exceeds=true means P(X > threshold), exceeds=false means P(X <= threshold)
- is_upper_bound=true means "variable <= threshold", is_upper_bound=false means "variable > threshold"

Respond with a JSON object containing four arrays:
{{
  "probability_queries": [
    {{
      "text": "What is the probability that VariableName exceeds 10?",
      "target_variable": "VariableName",
      "threshold": 10.0,
      "exceeds": true,
      "informativeness": 0.8
    }}
  ],
  "expectation_queries": [
    {{
      "text": "What is the expected value of VariableName?",
      "target_variable": "VariableName",
      "informativeness": 0.7
    }}
  ],
  "conditional_probability_queries": [
    {{
      "text": "Given that VarA is above 5, what is the probability that VarB exceeds 15?",
      "target_variable": "VarB",
      "threshold": 15.0,
      "exceeds": true,
      "conditions": [
        {{"variable": "VarA", "threshold": 5.0, "is_upper_bound": false}}
      ],
      "informativeness": 0.9
    }}
  ],
  "conditional_expectation_queries": [
    {{
      "text": "Given that VarA is above 5, what is the expected value of VarB?",
      "target_variable": "VarB",
      "conditions": [
        {{"variable": "VarA", "threshold": 5.0, "is_upper_bound": false}}
      ],
      "informativeness": 0.85
    }}
  ]
}}"""
    },

    
    "probability_query": {
        "system": """You are an expert forecaster providing calibrated probability estimates.
Give your best estimate as a probability between 0 and 1.
Be specific and quantitative. Explain your reasoning briefly, then state your probability estimate.
Format your final answer as: "My probability estimate: X%" where X is a number between 0 and 100.""",
        
        "user": """{query}

Consider relevant factors and provide your probability estimate."""
    },
    
    "quantile_query": {
        "system": """You are an expert forecaster providing calibrated numeric estimates.
Give your best estimate for the requested quantity.
Be specific and quantitative. Explain your reasoning briefly, then state your numeric estimate.
Format your final answer as: "My estimate: [number]" with appropriate units if applicable.""",
        
        "user": """{query}

Consider relevant factors and provide your numeric estimate."""
    },
    
    "conditional_probability": {
        "system": """You are an expert forecaster providing calibrated conditional probability estimates.
Given a specific condition, estimate how it affects the probability of the outcome.
Be specific and quantitative. Explain your reasoning briefly, then state your probability estimate.
Format your final answer as: "My conditional probability estimate: X%" where X is a number between 0 and 100.""",
        
        "user": """Main question: {main_question}

Condition: {condition}

Query: {query}

Assuming the condition is true, provide your probability estimate."""
    },
    
    "threshold_probability": {
        "system": """You are an expert forecaster estimating the probability that a quantity exceeds a threshold.
Think about the distribution of possible values and estimate what fraction would exceed the threshold.
Be specific and quantitative. Explain your reasoning briefly, then state your probability estimate.
Format your final answer as: "Probability of exceeding threshold: X%" where X is a number between 0 and 100.""",
        
        "user": """Question context: {context}

What is the probability that {variable} will be {direction} {threshold}?

Consider the range of plausible values and provide your probability estimate."""
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
