"""Query types for eliciting distributional information."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from calibrated_response.models.variable import Variable


class QueryType(str, Enum):
    """Type of distributional query."""
    MARGINAL = "marginal"           # P(X)
    CONDITIONAL = "conditional"     # P(X | Y)
    THRESHOLD = "threshold"         # P(X > t) or P(X < t)
    EXPECTATION = "expectation"     # E[X] or E[X | Y]
    QUANTILE = "quantile"           # Median, quartiles, etc.


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
    
    # Query metadata
    estimated_informativeness: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Estimated information gain from this query"
    )
    
    def to_prompt(self) -> str:
        """Convert query to a prompt suitable for the LLM."""
        return self.text


class MarginalQuery(Query):
    """Query for the marginal distribution of a variable.
    
    For binary variables: "What is the probability that X?"
    For continuous: Converted to threshold or quantile queries.
    """
    
    query_type: QueryType = Field(default=QueryType.MARGINAL, frozen=True)


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


class ConditionalQuery(Query):
    """Query for conditional distribution P(X | condition).
    
    Example: "If it rains tomorrow, what is the probability that more
    than 50,000 people will take the train?"
    """
    
    query_type: QueryType = Field(default=QueryType.CONDITIONAL, frozen=True)
    
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


class QuantileQuery(Query):
    """Query for quantiles of a distribution.
    
    Example: "What is the median number of people who will take the train?"
    """
    
    query_type: QueryType = Field(default=QueryType.QUANTILE, frozen=True)
    
    quantile: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="The quantile to query (0.5 = median)"
    )
    
    # For conditional quantiles
    condition_text: Optional[str] = Field(None, description="Conditioning statement if any")
    
    def to_prompt(self) -> str:
        """Generate the natural language prompt."""
        quantile_name = {
            0.25: "25th percentile",
            0.5: "median",
            0.75: "75th percentile",
            0.1: "10th percentile",
            0.9: "90th percentile",
        }.get(self.quantile, f"{self.quantile*100:.0f}th percentile")
        
        if self.condition_text:
            return f"Given that {self.condition_text}, what is the {quantile_name} of {self.target_variable}?"
        return f"What is the {quantile_name} of {self.target_variable}?"


class ExpectationQuery(Query):
    """Query for expected value of a variable.
    
    Example: "What is the expected number of train riders tomorrow?"
    """
    
    query_type: QueryType = Field(default=QueryType.EXPECTATION, frozen=True)
    
    # For conditional expectations
    condition_text: Optional[str] = Field(None, description="Conditioning statement if any")
    
    def to_prompt(self) -> str:
        """Generate the natural language prompt."""
        if self.condition_text:
            return f"Given that {self.condition_text}, what is the expected value of {self.target_variable}?"
        return f"What is the expected value of {self.target_variable}?"


class QueryData(BaseModel):
    """Raw query data from LLM generation."""
    id: str = Field(..., description="Unique identifier for the query")
    text: str = Field(..., description="Natural language query text")
    target_variable: str = Field(..., description="Name of the variable being queried")
    query_type: str = Field(..., description="Type: threshold, conditional, marginal, quantile, expectation")
    threshold: Optional[float] = Field(None, description="Threshold value if applicable")
    threshold_direction: Optional[str] = Field(None, description="Direction: greater or less")
    condition_variable: Optional[str] = Field(None, description="Variable being conditioned on")
    condition_text: Optional[str] = Field(None, description="Natural language condition")
    informativeness: float = Field(0.5, description="Estimated information gain")


class QueryList(BaseModel):
    """List of queries from LLM generation."""
    queries: list[QueryData] = Field(..., description="List of generated queries")


class QueryResult(BaseModel):
    """Result of a distributional query."""
    
    query_id: str = Field(..., description="ID of the query this answers")
    
    # The answer
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Probability answer (for threshold/marginal queries)"
    )
    value: Optional[float] = Field(
        None,
        description="Value answer (for quantile/expectation queries)"
    )
    
    # Confidence in the answer
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="LLM's confidence in this answer"
    )
    
    # Raw response for debugging
    raw_response: Optional[str] = Field(None, description="Raw LLM response text")
    
    def get_answer(self) -> float:
        """Get the numeric answer."""
        if self.probability is not None:
            return self.probability
        if self.value is not None:
            return self.value
        raise ValueError("No answer available")
