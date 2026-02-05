"""Query types for eliciting distributional information."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from calibrated_response.models.variable import Variable, BinaryVariable, ContinuousVariable, VariableType

#A proposition is a statement about a variable. 
class PropositionType(str, Enum):
    """Type of condition in a conditional query."""
    EQUALITY = "equality"       # X = value
    INEQUALITY = "inequality"   # X > value or X < value

class Proposition(BaseModel):
    """A condition applied to a variable in a conditional query."""
    model_config = ConfigDict(frozen=True)
    proposition_type: PropositionType = Field(..., description="Type of proposition")
    variable: str = Field(..., description="Name of the variable")
    variable_type: VariableType = Field(..., description="Type of the variable")

class EqualityProposition(Proposition):
    """Equality condition: X = value."""
    proposition_type: PropositionType = Field(default=PropositionType.EQUALITY, frozen=True, description="Equality proposition for binary variables")
    variable_type: VariableType = VariableType.BINARY
    value: bool | str = Field(..., description="Value for equality condition")

class InequalityProposition(Proposition):
    """Inequality condition: X > value or X < value."""
    proposition_type: PropositionType = Field(default=PropositionType.INEQUALITY, frozen=True, description="Inequality proposition for continuous or discrete variables")
    variable_type: VariableType = Field(..., description="Type of the variable (continuous or discrete)")
    threshold: float = Field(..., description="Threshold value for inequality")
    greater: bool = Field(..., description="True for X > threshold, False for X < threshold")

class EstimateType(str, Enum):
    """Type of likelihood estimate."""
    PROBABILITY = "probability"     # P(X)
    EXPECTATION = "expectation"     # E[X]
    CONDITIONAL_PROBABILITY = "conditional_probability"   # P(X | condition)
    CONDITIONAL_EXPECTATION = "conditional_expectation"   # E[X | condition]
    
class Estimate(BaseModel):
    """Base class for likelihood estimates."""
    id: str = Field(..., description="Unique identifier for the estimate")
    estimate_type: EstimateType = Field(..., description="Type of likelihood estimate")

class ProbabilityEstimate(Estimate):
    """Estimate for probability P(X)."""
    estimate_type: EstimateType = Field(default=EstimateType.PROBABILITY, frozen=True)
    proposition: Proposition = Field(..., description="Proposition defining the event")
    probability: float = Field(..., description="Estimated probability value")

class ExpectationEstimate(Estimate):
    """Estimate for expectation E[X]."""
    estimate_type: EstimateType = Field(default=EstimateType.EXPECTATION, frozen=True)
    variable: str = Field(..., description="Name of the variable")
    expected_value: float = Field(..., description="Estimated expected value")

class ConditionalProbabilityEstimate(Estimate):
    """Estimate for conditional probability P(X | condition)."""
    estimate_type: EstimateType = Field(default=EstimateType.CONDITIONAL_PROBABILITY, frozen=True)
    proposition: Proposition = Field(..., description="Proposition defining the event")
    conditions: list[Proposition] = Field(default=[], description="Condition propositions")
    probability: float = Field(..., description="Estimated conditional probability value")

class ConditionalExpectationEstimate(Estimate):
    """Estimate for conditional expectation E[X | condition]."""
    estimate_type: EstimateType = Field(default=EstimateType.CONDITIONAL_EXPECTATION, frozen=True)
    variable: str = Field(..., description="Name of the variable")
    conditions: list[Proposition] = Field(default=[], description="Condition propositions")
    expected_value: float = Field(..., description="Estimated conditional expected value")

class QueryType(str, Enum):
    """Type of distributional query."""
    BOOLEAN = "boolean"         # P(X) for binary variables
    THRESHOLD = "threshold"         # P(X >= t) or P(X <= t)
    EXPECTATION = "expectation"     # E[X] or E[X | Y]
    CONDITIONAL = "conditional"     # P(X | condition)

class Query(BaseModel):
    """Base class for distributional queries.
    
    Queries are specific questions asked to the LLM to elicit 
    distributional information about variables.
    """
    
    id: str = Field(..., description="Unique identifier for the query")
    query_type: QueryType = Field(..., description="Type of query")
    
    # The natural language form of the query
    text: str = Field(..., description="Natural language query text")
    
    # Variables involved
    target_variable: str = Field(..., description="Name of the variable being queried")


class BooleanQuery(Query):
    """Query for the marginal distribution of a variable.
    
    For binary variables: "What is the probability that X?"
    For continuous: Converted to threshold or quantile queries.
    """
    
    query_type: QueryType = Field(default=QueryType.BOOLEAN, frozen=True)

    def to_prompt(self) -> str:
        """Generate the natural language prompt."""
        return f"What is the probability that {self.target_variable} is true?"

class ThresholdQuery(Query):
    """Query for probability of exceeding/falling below a threshold.
    
    Example: "What is the probability that more than 50,000 people 
    will take the train tomorrow?"
    """
    
    query_type: QueryType = Field(default=QueryType.THRESHOLD, frozen=True)
    threshold: float = Field(..., description="The threshold value")
    direction: str = Field(
        "greater",
        description="'greater' for P(X > t), 'less' for P(X < t)"
    )
    
    def to_prompt(self) -> str:
        """Generate the natural language prompt."""
        if self.direction == "greater":
            return f"What is the probability that {self.target_variable} will be greater than {self.threshold}?"
        else:
            return f"What is the probability that {self.target_variable} will be less than {self.threshold}?"

class ExpectationQuery(Query):
    """Query for expected value of a variable.
    
    Example: "What is the expected number of train riders tomorrow?"
    """
    
    query_type: QueryType = Field(default=QueryType.EXPECTATION, frozen=True)
    
    def to_prompt(self) -> str:
        """Generate the natural language prompt."""
        return f"What is the expected value of {self.target_variable}?"
    

class ConditionalQuery(Query):
    """Query for conditional distribution P(X | conditions).
    
    Examples: "If it rains tomorrow, what is the probability that more
    than 50,000 people will take the train?" 

    "If it rains tomorrow and the temperature is below 60F, what is the expected number of train riders?"
    """
    
    query_type: QueryType = Field(default=QueryType.CONDITIONAL, frozen=True)
    base_query_type: QueryType = Field(..., description="Type of the base query (threshold or expectation)")
    
    # The conditioning information
    condition_variable: str = Field(..., description="Variable being conditioned on")
    condition_value: Any = Field(..., description="Value of the conditioning variable")
    condition_text: str = Field(..., description="Natural language condition")
    
    # For threshold conditionals
    threshold: Optional[float] = Field(None, description="Threshold if querying P(X > t | Y)")
    threshold_direction: str = Field("greater", description="Direction for threshold")

    def to_prompt(self) -> str:
        """Generate the natural language prompt."""
        if self.threshold is not None:
            direction_word = "greater" if self.threshold_direction == "greater" else "less"
            return (
                f"Given that {self.condition_text}, what is the probability that "
                f"{self.target_variable} will be {direction_word} than {self.threshold}?"
            )
        else:
            return f"Given that {self.condition_text}, what is the expected value of {self.target_variable}?"

