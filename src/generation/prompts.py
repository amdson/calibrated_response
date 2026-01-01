"""Prompt templates for LLM generation tasks."""

PROMPTS = {
    "variable_generation": {
        "system": """You are an expert forecaster helping to decompose complex prediction questions.
Your task is to identify relevant variables that could influence the answer to a forecasting question.
Focus on variables that are:
1. Measurable or estimable
2. Potentially influential on the outcome
3. Not perfectly correlated with each other
4. A mix of easy-to-know facts and uncertain quantities""",
        
        "user": """Question to forecast: {question}

Identify {n_variables} relevant variables that could help predict the answer to this question.
For each variable, provide:
- A short name (2-4 words)
- A description of what it represents
- Why it's relevant to the question
- Whether it's binary (yes/no), continuous (numeric), or discrete (categories)
- An estimated importance from 0 to 1

Respond in JSON format:
{{
  "variables": [
    {{
      "name": "variable name",
      "description": "what this variable represents",
      "relevance": "why this matters for the prediction",
      "type": "binary|continuous|discrete",
      "importance": 0.8
    }}
  ]
}}"""
    },
    
    "query_generation": {
        "system": """You are an expert forecaster designing queries to elicit probability distributions.
Your task is to generate specific, answerable questions about variables related to a forecasting problem.
Design queries that:
1. Can be answered with a probability or numeric value
2. Would provide useful information for the main prediction
3. Cover different aspects of uncertainty
4. Include both marginal and conditional queries when useful""",
        
        "user": """Main forecasting question: {question}

Relevant variables:
{variables}

Generate {n_queries} specific queries to help estimate the answer to the main question.
Each query should ask about one of the variables or relationships between them.
Include a mix of:
- Direct probability questions (P(X > threshold))
- Conditional questions (P(X | Y))
- Expectation/median questions

Respond in JSON format:
{{
  "queries": [
    {{
      "id": "q1",
      "text": "natural language query",
      "target_variable": "variable name",
      "query_type": "threshold|conditional|marginal|quantile|expectation",
      "threshold": null,
      "condition_variable": null,
      "condition_text": null,
      "informativeness": 0.8
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
    """Format a list of variables for inclusion in a prompt."""
    lines = []
    for i, var in enumerate(variables, 1):
        name = var.get('name', f'Variable {i}')
        desc = var.get('description', 'No description')
        vtype = var.get('type', 'unknown')
        lines.append(f"{i}. {name} ({vtype}): {desc}")
    return "\n".join(lines)
