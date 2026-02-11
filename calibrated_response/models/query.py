"""Query types for eliciting distributional information."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from calibrated_response.models.variable import VariableType

class Proposition(BaseModel):
    """A condition applied to a variable in a conditional query."""
    model_config = ConfigDict(frozen=True)
    proposition_type: Literal['equality', 'inequality']
    variable: str = Field(..., description="Name of the variable")
    variable_type: Literal["binary"] = Field(..., description="Type of the variable")

    def to_query_proposition(self) -> str:
        """Get a string representation suitable for queries."""
        raise NotImplementedError("Subclasses must implement to_query_proposition()")

class EqualityProposition(Proposition):
    """Equality condition: X = value."""
    proposition_type: Literal["equality"] = "equality"
    variable_type: Literal["binary"] = "binary"
    value: bool | str = Field(..., description="Value for equality condition")

    def to_query_proposition(self) -> str:
        """Get a string representation suitable for queries."""
        return f"{self.variable} = {self.value}"

class InequalityProposition(Proposition):
    """Inequality condition: X > value or X < value."""
    proposition_type: Literal["inequality"] = "inequality"
    variable_type: Literal["continuous", "discrete"] = "continuous"
    threshold: float = Field(..., description="Threshold value for inequality")
    is_lower_bound: bool = Field(..., description="True for X > threshold, False for X < threshold")

    def to_query_proposition(self) -> str:
        """Get a string representation suitable for queries."""
        operator = "<" if self.is_lower_bound else ">"
        return f"{self.variable} {operator} {self.threshold}"

# Discriminated union for propositions - Pydantic will use proposition_type to determine subclass
PropositionUnion = Annotated[
    Union[EqualityProposition, InequalityProposition],
    Field(discriminator='proposition_type')
]

class Estimate(BaseModel):
    """Base class for likelihood estimates."""
    model_config = ConfigDict(frozen=True)
    id: str = Field(..., description="Unique identifier for the estimate")
    estimate_type: Literal["probability", "expectation", "conditional_probability", "conditional_expectation"] = Field(..., description="Type of likelihood estimate")

    def to_query_estimate(self) -> str:
        """Get a string representation suitable for queries."""
        raise NotImplementedError("Subclasses must implement to_query_estimate()")

class ProbabilityEstimate(Estimate):
    """Estimate for probability P(X)."""
    estimate_type: Literal["probability"] = "probability"
    proposition: PropositionUnion = Field(..., description="Proposition defining the event")
    probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability value")

    def to_query_estimate(self) -> str:
        return f"P({self.proposition.to_query_proposition()}) = {self.probability}"

class ExpectationEstimate(Estimate):
    """Estimate for expectation E[X]."""
    estimate_type: Literal["expectation"] = "expectation"
    variable: str = Field(..., description="Name of the variable")
    expected_value: float = Field(..., description="Estimated expected value")

    def to_query_estimate(self) -> str:
        return f"E[{self.variable}] = {self.expected_value}"

class ConditionalProbabilityEstimate(Estimate):
    """Estimate for conditional probability P(X | condition)."""
    estimate_type: Literal["conditional_probability"] = "conditional_probability"
    proposition: PropositionUnion = Field(..., description="Proposition defining the event")
    conditions: list[PropositionUnion] = Field(default_factory=list, description="Condition propositions")
    probability: float = Field(..., ge=0.0, le=1.0, description="Estimated conditional probability value")

    def to_query_estimate(self) -> str:
        conditions_str = ", ".join([cond.to_query_proposition() for cond in self.conditions])
        return f"P({self.proposition.to_query_proposition()} | {conditions_str}) = {self.probability}"

class ConditionalExpectationEstimate(Estimate):
    """Estimate for conditional expectation E[X | condition]."""
    estimate_type: Literal["conditional_expectation"] = "conditional_expectation"
    variable: str = Field(..., description="Name of the variable")
    conditions: list[PropositionUnion] = Field(default_factory=list, description="Condition propositions")
    expected_value: float = Field(..., description="Estimated conditional expected value")

    def to_query_estimate(self) -> str:
        conditions_str = ", ".join([cond.to_query_proposition() for cond in self.conditions])
        return f"E[{self.variable} | {conditions_str}] = {self.expected_value}"
    
# Discriminated union for estimates
EstimateUnion = Annotated[
    Union[
        ProbabilityEstimate,
        ExpectationEstimate,
        ConditionalProbabilityEstimate,
        ConditionalExpectationEstimate
    ],
    Field(discriminator='estimate_type')
]

class EstimateList(BaseModel):
    """A list of estimates."""
    model_config = ConfigDict(frozen=True)
    estimates: list[EstimateUnion] = Field(default_factory=list, description="List of estimates")