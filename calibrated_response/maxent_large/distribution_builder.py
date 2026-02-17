"""Build distributions from variable and estimate lists using MaxEnt."""

from __future__ import annotations

from typing import Optional, Union, Sequence
import uuid
import copy

import numpy as np

from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.models.query import (
    EstimateUnion,
    ProbabilityEstimate,
    ExpectationEstimate,
    ConditionalProbabilityEstimate,
    ConditionalExpectationEstimate,
    EqualityProposition,
    InequalityProposition,
)
from calibrated_response.models.variable import Variable, ContinuousVariable, BinaryVariable
from calibrated_response.maxent.constraints import (
    Constraint,
    ProbabilityConstraint,
    MeanConstraint,
    ConditionalThresholdConstraint,
    ConditionalMeanConstraint,
    ConstraintUnion,
)
from calibrated_response.maxent_large.maxent_solver import (
    MaxEntSolver,
    JAXSolverConfig,
)
    
class DistributionBuilder:
    """Build probability distributions from variables and estimates using multivariate MaxEnt."""
    
    def __init__(
        self,
        variables: Sequence[Variable],
        estimates: Sequence[EstimateUnion],
        solver_config: Optional[JAXSolverConfig] = None,
    ):
        """Initialize the distribution builder.
        
        Args:
            variables: List of Variable objects defining the joint distribution
            estimates: List of Estimate objects (probability, expectation, conditional variants)
            solver_config: Configuration for the JAX MaxEnt solver
        """
        self.variables = variables
        self.estimates = estimates
        self.var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
        self.var_name_to_var = {v.name: v for v in variables}

        self.solver_config = solver_config or JAXSolverConfig()
        self.solver = MaxEntSolver(self.solver_config)
        # Convert estimates to constraints (in original domain)
        self.constraints = self._build_constraints()
        
        # Create normalized constraints for solver (all on [0, 1])
        self.normalized_constraints = self._normalize_constraints(self.constraints)
    
    def _build_constraints(self) -> list[ConstraintUnion]:
        """Convert all estimates to MaxEnt constraints."""
        constraints = []
        for estimate in self.estimates:
            constraint = self._estimate_to_constraint(estimate)
            if constraint is not None:
                constraints.append(constraint)
        return constraints
    
    def _normalize_constraints(self, constraints: list[ConstraintUnion]) -> list[ConstraintUnion]:
        """Normalize all constraints to [0, 1] domain."""
        normalized = []
        for constraint in constraints:
            norm_constraint = self._normalize_single_constraint(constraint)
            if norm_constraint is not None:
                normalized.append(norm_constraint)
        return normalized
    
    def build(
        self,
        target_variable: Optional[str] = None,
    ) -> dict:
        """Build a distribution for the target variable.
        
        The solver operates on normalized [0, 1] domain for all variables.
        Results are denormalized back to the original variable domains.
        
        Args:
            target_variable: Name of the variable to extract marginal for.
                           If None, returns the first variable marked as target,
                           or the first variable if none are marked.
            
        Returns:
            Tuple of (distribution, diagnostics)
        """
        target_var = self._get_target_variable(target_variable)
        
        # Handle no constraints case
        if len(self.constraints) == 0:
            distribution = self._get_uniform_distribution(target_var)
            info = {'message': 'No constraints, returning uniform distribution'}
            return distribution, info
        
        # Solve using multivariate MaxEnt with NORMALIZED constraints
        # The solver assumes all variables are on [0, 1]
        p_joint, bin_edges_list_normalized, info = self.solver.solve(
            variables=self.variables,
            constraints=self.normalized_constraints,
        )
        
        # Denormalize bin edges back to original domains
        bin_edges_list = self.normalizer.denormalize_bin_edges(bin_edges_list_normalized)
        
        # Extract marginal for the target variable
        target_idx = self.var_name_to_idx[target_var.name]
        marginal = extract_marginal(p_joint, target_idx)
        bin_edges = bin_edges_list[target_idx]
        
        # Add diagnostics
        info['joint_distribution'] = p_joint
        info['bin_edges_list'] = bin_edges_list  # Denormalized
        info['bin_edges_list_normalized'] = bin_edges_list_normalized  # Keep normalized for reference
        info['target_variable'] = target_var.name
        info['n_constraints'] = len(self.constraints)
        info['n_estimates'] = len(self.estimates)
        info['domains'] = {v.name: self.normalizer.get_domain(v.name) for v in self.variables}
        
        return distribution, info