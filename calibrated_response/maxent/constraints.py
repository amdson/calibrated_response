"""Constraint representations for maximum entropy optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from calibrated_response.models.variable import (Variable, 
                                                 BinaryVariable, ContinuousVariable, DiscreteVariable)



class ConstraintType(str, Enum):
    """Type of constraint."""
    PROBABILITY = "probability"  # P(x1 <= X < x2) = p
    MEAN = "mean"                # E[X] = mu
    VARIANCE = "variance"        # Var(X) = sigma^2
    THRESHOLD = "threshold"        # P(X <= x) = p
    CONDITIONALMEAN = "conditional_mean"  # E[X | conditions] = mu
    CONDITIONALTHRESHOLD = "conditional_threshold"  # P(X <= x | conditions) = p



class Constraint(BaseModel, ABC):
    """Abstract base class for distributional constraints."""
    
    class Config:
        arbitrary_types_allowed = True
    
    id: str = Field(..., description="Unique identifier for this constraint")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    target_variable: Variable = Field(..., description="The variable for which the mean is defined")
    
    # Confidence/weight for soft constraint handling
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this constraint (for soft constraints)"
    )

class ProbabilityConstraint(Constraint):
    """Constraint on probability mass in a region: P(a <= X <= b) = p."""
    
    constraint_type: ConstraintType = Field(default=ConstraintType.PROBABILITY, frozen=True)
    
    lower_bound: float = Field(..., description="Lower bound of region")
    upper_bound: float = Field(..., description="Upper bound of region")
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Target probability mass in region"
    )
    
    def evaluate(self, distribution: np.ndarray, bin_edges: np.ndarray) -> float:
        """Compute P(lower <= X <= upper) for the distribution."""
        total_prob = 0.0
        for i, (p, left, right) in enumerate(zip(
            distribution,
            bin_edges[:-1],
            bin_edges[1:]
        )):
            # Compute overlap of bin with [lower, upper]
            overlap_left = max(left, self.lower_bound)
            overlap_right = min(right, self.upper_bound)
            
            if overlap_right > overlap_left:
                # Fraction of bin that overlaps
                bin_width = right - left
                if bin_width > 0:
                    overlap_frac = (overlap_right - overlap_left) / bin_width
                    total_prob += p * overlap_frac
        
        return total_prob
    
    def target_value(self) -> float:
        return self.probability
    
    @classmethod
    def from_threshold(
        cls,
        threshold: float,
        probability: float,
        direction: str = "greater",
        domain_min: float = float('-inf'),
        domain_max: float = float('inf'),
        **kwargs,
    ) -> ProbabilityConstraint:
        """Create a constraint from a threshold query result.
        
        Args:
            threshold: The threshold value
            probability: P(X > threshold) or P(X <= threshold)
            direction: "greater" for P(X > t), "less" for P(X <= t)
            domain_min: Minimum of the domain
            domain_max: Maximum of the domain
        """
        if direction == "greater":
            # P(X > threshold) = probability
            return cls(
                lower_bound=threshold,
                upper_bound=domain_max,
                probability=probability,
                **kwargs,
            )
        else:
            # P(X <= threshold) = probability
            return cls(
                lower_bound=domain_min,
                upper_bound=threshold,
                probability=probability,
                **kwargs,
            )

class MeanConstraint(Constraint):
    """Constraint on expected value: E[X] = mu."""
    
    constraint_type: ConstraintType = Field(default=ConstraintType.MEAN, frozen=True)
    
    mean: float = Field(..., description="Target mean value")
    
    def evaluate(self, distribution: np.ndarray, bin_edges: np.ndarray) -> float:
        """Compute E[X] for the distribution."""
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean = np.sum(distribution * bin_centers)
        assert isinstance(mean, float)
        return mean
    
    def target_value(self) -> float:
        return self.mean


class VarianceConstraint(Constraint):
    """Constraint on variance: Var(X) = sigma^2."""
    
    constraint_type: ConstraintType = Field(default=ConstraintType.VARIANCE, frozen=True)
    
    variance: float = Field(..., ge=0, description="Target variance")
    
    def evaluate(self, distribution: np.ndarray, bin_edges: np.ndarray) -> float:
        """Compute Var(X) for the distribution."""
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean = np.sum(distribution * bin_centers)
        # assert isinstance(mean, float)
        variance = np.sum(distribution * (bin_centers - mean) ** 2)
        assert isinstance(variance, float)
        return variance
    
    def target_value(self) -> float:
        return self.variance


class ThresholdConstraint(Constraint):
    """Constraint P(X <= t) = p."""
    
    constraint_type: ConstraintType = Field(default=ConstraintType.THRESHOLD, frozen=True)
    
    threshold: float = Field(
        ...,
        description="Threshold value t in P(X <= t) = p"
    )
    probability: float = Field(..., description="Target threshold probability p")
    
    def target_value(self) -> float:
        return self.threshold
    

class ConditionalConstraint(Constraint):

    condition_variables : List[Variable] = Field(default_factory=list, description="List of condition variables")
    condition_values : List[float] = Field(default_factory=list, description="List of thresholds for condition variables")
    is_lower_bound: List[bool] = Field(default_factory=list, description="If True, condition is X > threshold (value is lower bound); if False, condition is X <= threshold (value is upper bound)")

class ConditionalThresholdConstraint(ConditionalConstraint):
    """Constraint on threshold: P(X <= t) = p, subject to arbitrary conditions on thresholds of variables."""
    
    constraint_type: ConstraintType = Field(default=ConstraintType.CONDITIONALTHRESHOLD, frozen=True)
    
    threshold: float = Field(
        ...,
        description="Threshold value t in P(X <= t) = p"
    )
    probability: float = Field(..., description="Target threshold probability p")

    def target_value(self) -> float:
        return self.threshold
    
class ConditionalMeanConstraint(ConditionalConstraint):
    """Constraint on mean: E[X] = mu, subject to arbitrary conditions on thresholds of variables."""
    
    constraint_type: ConstraintType = Field(default=ConstraintType.CONDITIONALMEAN, frozen=True)
    value: float = Field(..., description="Target mean value")

    def target_value(self) -> float:
        return self.value

# Union type for all constraints
ConstraintUnion = ProbabilityConstraint | MeanConstraint | VarianceConstraint | ThresholdConstraint | ConditionalThresholdConstraint | ConditionalMeanConstraint
    
def to_bins(var: Variable, max_bins=5, normalized: bool = False) -> np.ndarray:
    """Create bin edges for a variable.
    
    Args:
        var: Variable object
        max_bins: Maximum number of bins
        normalized: If True, always use [0, 1] domain regardless of variable's actual domain
    
    Returns:
        Array of bin edges
    """
    if isinstance(var, BinaryVariable):
        return np.array([0, 0.5, 1])
    if isinstance(var, ContinuousVariable):
        if normalized:
            # Use [0, 1] domain for normalized mode
            return np.linspace(0.0, 1.0, max_bins + 1)
        else:
            # Use variable's actual domain
            lower, upper = var.get_domain()
            return np.linspace(lower, upper, max_bins + 1)
    if isinstance(var, DiscreteVariable):
        # For discrete variables, bins correspond to categories
        return np.arange(len(var.categories) + 1)
    else:
        raise ValueError(f"Unsupported variable type: {type(var)}")

class ConstraintSet(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    variables: list[Variable] = Field(default_factory=list)
    constraint_variables: list[Variable] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

# class ConstraintSet(BaseModel):
#     """Collection of constraints for a maximum entropy problem."""
    
#     class Config:
#         arbitrary_types_allowed = True
    
#     constraints: list[Constraint] = Field(default_factory=list)
    
#     # Domain specification
#     domain_min: float = Field(0.0, description="Minimum of the domain")
#     domain_max: float = Field(100.0, description="Maximum of the domain")
#     n_bins: int = Field(50, description="Number of bins for discretization")
    
#     def add(self, constraint: Constraint) -> None:
#         """Add a constraint to the set."""
#         self.constraints.append(constraint)
    
#     def add_probability_constraint(
#         self,
#         lower: float,
#         upper: float,
#         probability: float,
#         confidence: float = 1.0,
#         source_query_id: Optional[str] = None,
#     ) -> None:
#         """Add a probability constraint."""
#         self.add(ProbabilityConstraint(
#             id=f"prob_{len(self.constraints)}",
#             lower_bound=lower,
#             upper_bound=upper,
#             probability=probability,
#             confidence=confidence,
#             source_query_id=source_query_id,
#         ))
    
#     def add_threshold_constraint(
#         self,
#         threshold: float,
#         probability: float,
#         direction: str = "greater",
#         confidence: float = 1.0,
#         source_query_id: Optional[str] = None,
#     ) -> None:
#         """Add a threshold constraint."""
#         constraint = ProbabilityConstraint.from_threshold(
#             threshold=threshold,
#             probability=probability,
#             direction=direction,
#             domain_min=self.domain_min,
#             domain_max=self.domain_max,
#             id=f"thresh_{len(self.constraints)}",
#             confidence=confidence,
#             source_query_id=source_query_id,
#         )
#         self.add(constraint)
    
#     def add_mean_constraint(
#         self,
#         mean: float,
#         confidence: float = 1.0,
#         source_query_id: Optional[str] = None,
#     ) -> None:
#         """Add a mean constraint."""
#         self.add(MeanConstraint(
#             id=f"mean_{len(self.constraints)}",
#             mean=mean,
#             confidence=confidence,
#             source_query_id=source_query_id,
#         ))
    
#     def add_quantile_constraint(
#         self,
#         quantile: float,
#         value: float,
#         confidence: float = 1.0,
#         source_query_id: Optional[str] = None,
#     ) -> None:
#         """Add a quantile constraint."""
#         self.add(ThresholdConstraint(
#             id=f"quantile_{len(self.constraints)}",
#             quantile=quantile,
#             value=value,
#             confidence=confidence,
#             source_query_id=source_query_id,
#         ))
    
#     @property
#     def bin_edges(self) -> np.ndarray:
#         """Get bin edges for the domain."""
#         return np.linspace(self.domain_min, self.domain_max, self.n_bins + 1)
    
#     def total_violation(self, distribution: np.ndarray) -> float:
#         """Compute total weighted violation across all constraints."""
#         edges = self.bin_edges
#         return sum(c.weighted_violation(distribution, edges) for c in self.constraints)
    
#     def constraint_violations(self, distribution: np.ndarray) -> dict[str, float]:
#         """Get individual constraint violations."""
#         edges = self.bin_edges
#         return {c.id: c.violation(distribution, edges) for c in self.constraints}
