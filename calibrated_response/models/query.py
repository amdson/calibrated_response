"""Query types for eliciting distributional information."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

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
        operator = ">" if self.is_lower_bound else "<"
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
    estimate_type: Literal["probability", "expectation", "conditional_probability", "conditional_expectation", "correlation", "equation"] = Field(..., description="Type of likelihood estimate")
    relation: Literal["eq", "le", "ge"] = Field(
        default="eq",
        description="How the estimated quantity relates to the stated value: "
                    "'eq' is a point belief (quantity = value, the default and "
                    "the only form in older elicitations), 'le'/'ge' are "
                    "one-sided bounds (quantity <= / >= value). Bounds map to a "
                    "hinge penalty in the solver — only the violating side is "
                    "constrained, leaving the quantity free on the other side.")
    sd: Optional[float] = Field(
        default=None,
        description="Optional belief width for THIS estimate, in the penalty's "
                    "native residual space (log-odds for probabilities under "
                    "the logit penalty, value units for expectations). None "
                    "means use the solver's global default. Set by "
                    "repeat-collapse to encode agreement/disagreement across "
                    "repeated elicitations of the same quantity.")

    def to_query_estimate(self) -> str:
        """Get a string representation suitable for queries."""
        raise NotImplementedError("Subclasses must implement to_query_estimate()")

    @property
    def _rel_op(self) -> str:
        """Rendering of ``relation`` as an operator for ``to_query_estimate``."""
        return {"eq": "=", "le": "<", "ge": ">"}[self.relation]

class ProbabilityEstimate(Estimate):
    """Estimate for probability P(X)."""
    estimate_type: Literal["probability"] = "probability"
    proposition: PropositionUnion = Field(..., description="Proposition defining the event")
    probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability value")

    def to_query_estimate(self) -> str:
        return f"P({self.proposition.to_query_proposition()}) {self._rel_op} {self.probability}"

class ExpectationEstimate(Estimate):
    """Estimate for expectation E[X]."""
    estimate_type: Literal["expectation"] = "expectation"
    variable: str = Field(..., description="Name of the variable")
    expected_value: float = Field(..., description="Estimated expected value")

    def to_query_estimate(self) -> str:
        return f"E[{self.variable}] {self._rel_op} {self.expected_value}"

class ConditionalProbabilityEstimate(Estimate):
    """Estimate for conditional probability P(X | condition)."""
    estimate_type: Literal["conditional_probability"] = "conditional_probability"
    proposition: PropositionUnion = Field(..., description="Proposition defining the event")
    conditions: list[PropositionUnion] = Field(default_factory=list, description="Condition propositions")
    probability: float = Field(..., ge=0.0, le=1.0, description="Estimated conditional probability value")

    def to_query_estimate(self) -> str:
        conditions_str = ", ".join([cond.to_query_proposition() for cond in self.conditions])
        return f"P({self.proposition.to_query_proposition()} | {conditions_str}) {self._rel_op} {self.probability}"

class ConditionalExpectationEstimate(Estimate):
    """Estimate for conditional expectation E[X | condition]."""
    estimate_type: Literal["conditional_expectation"] = "conditional_expectation"
    variable: str = Field(..., description="Name of the variable")
    conditions: list[PropositionUnion] = Field(default_factory=list, description="Condition propositions")
    expected_value: float = Field(..., description="Estimated conditional expected value")

    def to_query_estimate(self) -> str:
        conditions_str = ", ".join([cond.to_query_proposition() for cond in self.conditions])
        return f"E[{self.variable} | {conditions_str}] {self._rel_op} {self.expected_value}"
    
class CorrelationEstimate(Estimate):
    """Estimate for the Pearson correlation Corr(X, Y).

    Scale-free dependence information: unlike conditional estimates it does
    not hinge on a conditioning event being sampled often enough, which makes
    it a cheap way for the LLM to couple variables. Binary variables are
    treated as their 0/1 indicator."""
    estimate_type: Literal["correlation"] = "correlation"
    variable_a: str = Field(..., description="Name of the first variable")
    variable_b: str = Field(..., description="Name of the second variable")
    correlation: float = Field(..., ge=-1.0, le=1.0,
                               description="Estimated Pearson correlation")

    def to_query_estimate(self) -> str:
        return f"Corr({self.variable_a}, {self.variable_b}) {self._rel_op} {self.correlation}"


class EquationEstimate(Estimate):
    """A structural relation ``lhs <op> rhs`` over the variables.

    ``rhs`` is an arithmetic expression (``+ - * / **``, constants, variable
    names, ``abs``/``min``/``max``, and the threshold indicator ``ind(e > c)``);
    see :mod:`calibrated_response.maxent_sampler.equations`.  Everything is a
    statement about the residual ``r = lhs - (rhs)``, and ``relation`` picks
    which statement:

    * ``"eq"`` — an equation ``lhs = rhs``.  With ``noise_sd=None`` it is a
      deterministic identity (``r -> 0`` for every sample): Fermi links the
      target is a function of, ``total = a*x + b*y``, ``diff = end - start``.
      With a positive ``noise_sd`` it is additive Gaussian noise
      ``lhs = rhs + N(0, noise_sd)`` — the solver matches the residual's mean
      (0) and variance (``noise_sd**2``) and maxent completes it to Gaussian
      noise ~independent of the rhs.
    * ``"ge"`` / ``"le"`` — an *inequality* ``lhs > rhs`` / ``lhs < rhs``: a
      one-sided constraint on the residual (``r >= 0`` / ``r <= 0``) carried by
      a hinge penalty, so only the violating side costs anything.  This is a
      support restriction rather than a moment match: maxent then spreads
      freely over the allowed region instead of hugging a line.  Here
      ``noise_sd`` sets how *soft* the boundary is (the scale over which
      violations become expensive) rather than a residual spread; ``None``
      falls back to the solver's deterministic tolerance, i.e. a sharp edge.
      Use for bounds a quantity must respect: ``revenue > costs``,
      ``peak_2030 < capacity ~ N(0, 5)``.
    """
    estimate_type: Literal["equation"] = "equation"
    lhs: str = Field(..., description="Left-hand-side variable name")
    rhs: str = Field(..., description="Right-hand-side arithmetic expression over the variables")
    noise_sd: Optional[float] = Field(
        default=None, ge=0.0,
        description="For an equation: additive Gaussian noise sd, None (or 0) = "
                    "deterministic identity. For an inequality: the softness of "
                    "the boundary, None = the solver's sharp default tolerance")

    def to_query_estimate(self) -> str:
        base = f"{self.lhs} {self._rel_op} {self.rhs}"
        return base if not self.noise_sd else f"{base} ~ N(0, {self.noise_sd})"


# Discriminated union for estimates
EstimateUnion = Annotated[
    Union[
        ProbabilityEstimate,
        ExpectationEstimate,
        ConditionalProbabilityEstimate,
        ConditionalExpectationEstimate,
        CorrelationEstimate,
        EquationEstimate
    ],
    Field(discriminator='estimate_type')
]

class EstimateList(BaseModel):
    """A list of estimates."""
    model_config = ConfigDict(frozen=True)
    estimates: list[EstimateUnion] = Field(default_factory=list, description="List of estimates")