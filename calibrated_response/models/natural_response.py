import re
import uuid
from typing import Literal, Union, List
from pydantic import BaseModel, Field

from calibrated_response.models.query import (
    Proposition, PropositionUnion, EqualityProposition, InequalityProposition,
    Estimate, EstimateUnion, ProbabilityEstimate, ExpectationEstimate, 
    ConditionalExpectationEstimate, ConditionalProbabilityEstimate)

class NaturalEstimate(BaseModel):
    """
    The interface for the LLM. 
    It only asks for a strictly formatted string, minimizing token overhead.
    """
    logic: str = Field(..., description="Concise natural language explanation of the estimate.")
    # Regex breakdown:
    # ^([PE])             -> Starts with P or E (Group 1)
    # [\[\(]              -> Opening bracket [ or (
    # (.+?)               -> The main term (Variable or Proposition) (Group 2)
    # (?: \|\ (.+?))?     -> Optional: " | " followed by conditions (Group 3)
    # [\]\)]              -> Closing bracket ] or )
    # \s*=\s* -> Equals sign with optional whitespace
    # (.+)$               -> The result value (Group 4)
    expression: str = Field(
        ..., 
        pattern=r"^([PE])[\[\(](.+?)(?: \| (.+?))?[\]\)]\s*=\s*(.+)$",
        description="Format: 'P(A > 10 | B = True) = 0.5' or 'E[Cost | Tax = True] = 100.0'",
        examples=["E[battery_cost | growth > 40.0] = 125.0"]
    )

    def convert(self) -> EstimateUnion:
        """Helper method to convert this natural response into your structured format."""
        return parse_natural_syntax(self.expression)


class NaturalEstimateList(BaseModel):
    """A list of natural language estimates for LLM generation."""
    estimates: List[NaturalEstimate] = Field(
        default_factory=list,
        description="List of estimates in natural language format"
    )
    
    def convert_all(self) -> List[EstimateUnion]:
        """Convert all natural estimates to structured format."""
        l = []
        for est in self.estimates:
            try:
                l.append(est.convert())
            except Exception as e:
                print(f"Error converting estimate '{est.expression}': {e}")
        return l


# ==========================================
# 2. The Converter Logic
# ==========================================

def _parse_proposition(prop_str: str) -> PropositionUnion:
    """Parses a string like 'X > 5' or 'Y = True' into a Proposition object."""
    # Regex to find operator and split
    # Matches: variable name, operator (>=, <=, >, <, =), value
    match = re.match(r"(.+?)\s*(>=|<=|>|<|=)\s*(.+)", prop_str.strip())
    
    if not match:
        raise ValueError(f"Could not parse proposition: {prop_str}")
        
    var_name, operator, val_str = match.groups()
    var_name = var_name.strip()
    val_str = val_str.strip()

    # Helper to parse values
    def parse_value(v: str) -> Union[bool, float, str]:
        if v.lower() == 'true' or v.lower() == '1': return True
        if v.lower() == 'false' or v.lower() == '0': return False
        try: return float(v)
        except ValueError: return v  # Return as string if not float/bool

    value = parse_value(val_str)

    if operator == "=":
        # Construct EqualityProposition
        assert isinstance(value, (bool, str)), "EqualityProposition value must be bool or str"
        return EqualityProposition(
            variable=var_name,
            value=value
        )
    else:
        # Construct InequalityProposition
        assert isinstance(value, (int, float)), "InequalityProposition value must be numeric"
        return InequalityProposition(
            variable=var_name,
            threshold=float(value), # Inequalities require floats in your model
            is_lower_bound=(not operator in ["<", "<="])
        )

def parse_natural_syntax(expression: str) -> EstimateUnion:
    """
    Converts a natural language math string into the complex EstimateUnion structure.
    """
    # 1. Regex Extraction
    pattern = r"^([PE])[\[\(](.+?)(?: \| (.+?))?[\]\)]\s*=\s*(.+)$"
    match = re.match(pattern, expression.strip())
    
    if not match:
        raise ValueError(f"Invalid syntax: {expression}")

    type_char, main_term, condition_str, value_str = match.groups()
    
    # 2. Parse Value
    est_value = float(value_str)
    
    # 3. Parse Conditions (if any)
    conditions = []
    if condition_str:
        # Split by comma if multiple conditions exist (e.g. "A=1, B>2")
        cond_parts = [c.strip() for c in condition_str.split(',')]
        conditions = [_parse_proposition(c) for c in cond_parts]

    # 4. Generate a unique ID (required by your base class)
    est_id = f"{type_char}_{main_term}_{'_'.join([c.variable[:5] for c in conditions])}"

    # 5. Construct the specific Object
    if type_char == 'P':
        # Probability: Main term is a Proposition (e.g., P(X > 5))
        main_prop = _parse_proposition(main_term)
        
        if conditions:
            return ConditionalProbabilityEstimate(
                id=est_id,
                proposition=main_prop,
                conditions=conditions,
                probability=est_value
            )
        else:
            return ProbabilityEstimate(
                id=est_id,
                proposition=main_prop,
                probability=est_value
            )

    elif type_char == 'E':
        # Expectation: Main term is just a Variable (e.g., E[Cost])
        variable_name = main_term.strip()
        
        if conditions:
            return ConditionalExpectationEstimate(
                id=est_id,
                variable=variable_name,
                conditions=conditions,
                expected_value=est_value
            )
        else:
            return ExpectationEstimate(
                id=est_id,
                variable=variable_name,
                expected_value=est_value
            )
            
    raise ValueError(f"Unknown estimate type: {type_char}")