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
from calibrated_response.maxent.multivariate_solver import (
    MultivariateMaxEntSolver,
    JAXSolverConfig,
    extract_marginal,
)

class DomainNormalizer:
    """Handles normalization of variables to [0, 1] domain and denormalization of results."""
    
    def __init__(self, variables: Sequence[Variable]):
        """Initialize with variable list to extract domains.
        
        Args:
            variables: List of Variable objects
        """
        self.variables = variables
        self.var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
        
        # Store original domains: (min, max) for each variable
        self.domains: list[tuple[float, float]] = []
        for var in variables:
            if isinstance(var, ContinuousVariable):
                self.domains.append(var.get_domain())
            elif isinstance(var, BinaryVariable):
                self.domains.append((0.0, 1.0))
            else:
                self.domains.append((0.0, 1.0))
    
    def get_domain(self, var_name: str) -> tuple[float, float]:
        """Get the original domain for a variable."""
        idx = self.var_name_to_idx.get(var_name)
        if idx is None:
            raise ValueError(f"Variable '{var_name}' not found.")
        return self.domains[idx]
    
    def normalize_value(self, value: float, var_name: str) -> float:
        """Normalize a value from original domain to [0, 1]."""
        lo, hi = self.get_domain(var_name)
        if hi == lo:
            return 0.5
        return (value - lo) / (hi - lo)
    
    def denormalize_value(self, value: float, var_name: str) -> float:
        """Denormalize a value from [0, 1] to original domain."""
        lo, hi = self.get_domain(var_name)
        return lo + value * (hi - lo)
    
    def normalize_value_by_idx(self, value: float, idx: int) -> float:
        """Normalize a value from original domain to [0, 1] by variable index."""
        lo, hi = self.domains[idx]
        if hi == lo:
            return 0.5
        return (value - lo) / (hi - lo)
    
    def denormalize_value_by_idx(self, value: float, idx: int) -> float:
        """Denormalize a value from [0, 1] to original domain by variable index."""
        lo, hi = self.domains[idx]
        return lo + value * (hi - lo)
    
    def denormalize_bin_edges(self, bin_edges_list: list[np.ndarray]) -> list[np.ndarray]:
        """Denormalize all bin edges from [0, 1] to original domains."""
        result = []
        for idx, edges in enumerate(bin_edges_list):
            lo, hi = self.domains[idx]
            denorm_edges = lo + edges * (hi - lo)
            result.append(denorm_edges)
        return result
    
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
        self.solver = MultivariateMaxEntSolver(self.solver_config)
        
        # Initialize domain normalizer
        self.normalizer = DomainNormalizer(variables)
        
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
    
    def _normalize_single_constraint(self, constraint: ConstraintUnion) -> Optional[ConstraintUnion]:
        """Normalize a single constraint to [0, 1] domain."""
        target_var = constraint.target_variable
        var_name = target_var.name
        
        if isinstance(constraint, ProbabilityConstraint):
            # Normalize lower_bound and upper_bound
            norm_lower = self.normalizer.normalize_value(constraint.lower_bound, var_name)
            norm_upper = self.normalizer.normalize_value(constraint.upper_bound, var_name)
            
            return ProbabilityConstraint(
                id=constraint.id,
                target_variable=target_var,
                lower_bound=norm_lower,
                upper_bound=norm_upper,
                probability=constraint.probability,
                confidence=constraint.confidence,
            )
        
        elif isinstance(constraint, MeanConstraint):
            # Normalize the mean value
            norm_mean = self.normalizer.normalize_value(constraint.mean, var_name)
            
            return MeanConstraint(
                id=constraint.id,
                target_variable=target_var,
                mean=norm_mean,
                confidence=constraint.confidence,
            )
        
        elif isinstance(constraint, ConditionalThresholdConstraint):
            # Normalize the threshold value and condition values
            norm_threshold = self.normalizer.normalize_value(constraint.threshold, var_name)
            probability = constraint.probability
            
            # Normalize condition values
            norm_cond_values = []
            for cond_var, cond_val in zip(constraint.condition_variables, constraint.condition_values):
                norm_cond_values.append(
                    self.normalizer.normalize_value(cond_val, cond_var.name)
                )
            
            return ConditionalThresholdConstraint(
                id=constraint.id,
                target_variable=target_var,
                probability=probability,
                threshold=norm_threshold,
                condition_variables=constraint.condition_variables,
                condition_values=norm_cond_values,
                is_lower_bound=constraint.is_lower_bound,
                confidence=constraint.confidence,
            )
        
        elif isinstance(constraint, ConditionalMeanConstraint):
            # Normalize the mean value and condition values
            norm_value = self.normalizer.normalize_value(constraint.value, var_name)
            
            # Normalize condition values
            norm_cond_values = []
            for cond_var, cond_val in zip(constraint.condition_variables, constraint.condition_values):
                norm_cond_values.append(
                    self.normalizer.normalize_value(cond_val, cond_var.name)
                )
            
            return ConditionalMeanConstraint(
                id=constraint.id,
                target_variable=target_var,
                value=norm_value,
                condition_variables=constraint.condition_variables,
                condition_values=norm_cond_values,
                is_lower_bound=constraint.is_lower_bound,
                confidence=constraint.confidence,
            )
        
        # Unknown constraint type - throw error
        raise ValueError(f"Unknown constraint type: {type(constraint)}")
    
    def _estimate_to_constraint(self, estimate: EstimateUnion) -> Optional[Constraint]:
        """Convert a single estimate to a MaxEnt constraint."""
        constraint_id = f"c_{estimate.id}_{uuid.uuid4().hex[:6]}"
        
        if isinstance(estimate, ProbabilityEstimate):
            return self._probability_estimate_to_constraint(estimate, constraint_id)
        elif isinstance(estimate, ExpectationEstimate):
            return self._expectation_estimate_to_constraint(estimate, constraint_id)
        elif isinstance(estimate, ConditionalProbabilityEstimate):
            return self._conditional_probability_to_constraint(estimate, constraint_id)
        elif isinstance(estimate, ConditionalExpectationEstimate):
            return self._conditional_expectation_to_constraint(estimate, constraint_id)
        else:
            raise ValueError(f"Unknown estimate type: {type(estimate)}")
    
    def _probability_estimate_to_constraint(
        self,
        estimate: ProbabilityEstimate,
        constraint_id: str,
    ) -> Optional[ProbabilityConstraint]:
        """Convert ProbabilityEstimate to ProbabilityConstraint."""
        prop = estimate.proposition
        var_name = prop.variable
        
        if var_name not in self.var_name_to_var:
            return None
        
        target_var = self.var_name_to_var[var_name]
        
        # Get domain bounds
        if isinstance(target_var, ContinuousVariable):
            domain_min, domain_max = target_var.get_domain()
        elif isinstance(target_var, BinaryVariable):
            domain_min, domain_max = 0.0, 1.0
        else:
            domain_min, domain_max = 0.0, 1.0
        
        if isinstance(prop, InequalityProposition):
            # P(X > threshold) or P(X < threshold)
            threshold = prop.threshold
            if prop.is_lower_bound:
                # P(X > threshold)
                lower_bound = threshold
                upper_bound = domain_max
            else:
                # P(X < threshold)
                lower_bound = domain_min
                upper_bound = threshold

        elif isinstance(prop, EqualityProposition):
            # P(X = value) - for binary, map True -> upper half, False -> lower half
            if isinstance(prop.value, bool):
                if prop.value:
                    lower_bound = 0.5
                    upper_bound = 1.0
                else:
                    lower_bound = 0.0
                    upper_bound = 0.5
            else:
                # For discrete values, create a small interval
                val = float(prop.value) if isinstance(prop.value, (int, float)) else 0.5
                lower_bound = val - 0.01
                upper_bound = val + 0.01
        else:
            return None
        
        return ProbabilityConstraint(
            id=constraint_id,
            target_variable=target_var,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            probability=estimate.probability,
            confidence=1.0,
        )
    
    def _expectation_estimate_to_constraint(
        self,
        estimate: ExpectationEstimate,
        constraint_id: str,
    ) -> Optional[MeanConstraint]:
        """Convert ExpectationEstimate to MeanConstraint."""
        var_name = estimate.variable
        
        if var_name not in self.var_name_to_var:
            return None
        
        target_var = self.var_name_to_var[var_name]
        
        return MeanConstraint(
            id=constraint_id,
            target_variable=target_var,
            mean=estimate.expected_value,
            confidence=1.0,
        )
    
    def _conditional_probability_to_constraint(
        self,
        estimate: ConditionalProbabilityEstimate,
        constraint_id: str,
    ) -> Optional[ConditionalThresholdConstraint]:
        """Convert ConditionalProbabilityEstimate to ConditionalThresholdConstraint."""
        prop = estimate.proposition
        var_name = prop.variable
        
        if var_name not in self.var_name_to_var:
            return None
        
        target_var = self.var_name_to_var[var_name]
        
        # Parse conditions
        condition_vars = []
        condition_values = []
        is_lower_bound = []
        
        for cond in estimate.conditions:
            cond_var_name = cond.variable
            if cond_var_name not in self.var_name_to_var:
                continue
            
            cond_var = self.var_name_to_var[cond_var_name]
            condition_vars.append(cond_var)
            
            if isinstance(cond, InequalityProposition):
                condition_values.append(cond.threshold)
                is_lower_bound.append(cond.is_lower_bound)  # greater means NOT lower bound
            elif isinstance(cond, EqualityProposition):
                if isinstance(cond.value, bool):
                    condition_values.append(0.5)
                    is_lower_bound.append(cond.value)
                else:
                    #throw error for non-binary equality conditions since we can't handle them in threshold constraints
                    assert ValueError(f"Cannot handle non-binary equality condition for variable '{cond_var_name}' in conditional probability constraint.")
        
        if not condition_vars:
            return None
        
        # Get threshold from proposition
        if isinstance(prop, InequalityProposition):
            threshold = prop.threshold
            # P(X > threshold | cond) = prob means quantile at threshold = 1 - prob
            if prop.is_lower_bound:
                probability = 1.0 - estimate.probability
            else:
                probability = estimate.probability
        elif isinstance(prop, EqualityProposition):
            # For equality, use midpoint
            threshold = 0.5
            if isinstance(prop.value, bool):
                threshold = 0.5
                probability = 1 - estimate.probability if prop.value else estimate.probability #TODO check logic here
        else:
            return None
        
        return ConditionalThresholdConstraint(
            id=constraint_id,
            target_variable=target_var,
            probability=probability,
            threshold=threshold,
            condition_variables=condition_vars,
            condition_values=condition_values,
            is_lower_bound=is_lower_bound,
            confidence=1.0,
        )
    
    def _conditional_expectation_to_constraint(
        self,
        estimate: ConditionalExpectationEstimate,
        constraint_id: str,
    ) -> Optional[ConditionalMeanConstraint]:
        """Convert ConditionalExpectationEstimate to ConditionalMeanConstraint."""
        var_name = estimate.variable
        
        if var_name not in self.var_name_to_var:
            return None
        
        target_var = self.var_name_to_var[var_name]
        
        # Parse conditions
        condition_vars = []
        condition_values = []
        is_lower_bound = []
        
        for cond in estimate.conditions:
            cond_var_name = cond.variable
            if cond_var_name not in self.var_name_to_var:
                continue
            
            cond_var = self.var_name_to_var[cond_var_name]
            condition_vars.append(cond_var)
            
            if isinstance(cond, InequalityProposition):
                condition_values.append(cond.threshold)
                is_lower_bound.append(cond.is_lower_bound)
            elif isinstance(cond, EqualityProposition):
                if isinstance(cond.value, bool):
                    condition_values.append(0.5)
                    is_lower_bound.append(cond.value)
                else:
                    #throw error for non-binary equality conditions since we can't handle them in threshold constraints
                    assert ValueError(f"Cannot handle non-binary equality condition for variable '{cond_var_name}' in conditional probability constraint.")
        
        return ConditionalMeanConstraint(
            id=constraint_id,
            target_variable=target_var,
            value=estimate.expected_value,
            condition_variables=condition_vars,
            condition_values=condition_values,
            is_lower_bound=is_lower_bound,
            confidence=1.0,  # Lower confidence for conditional mean without proper conditions
        )
    
    def _get_target_variable(self, target_variable: Optional[str]) -> Variable:
        """Get the target variable for marginal extraction."""
        if target_variable is not None:
            if target_variable in self.var_name_to_var:
                return self.var_name_to_var[target_variable]
            raise ValueError(f"Variable '{target_variable}' not found")
        
        # Look for variable marked as target
        for v in self.variables:
            if getattr(v, 'is_target', False):
                return v
        
        # Default to first variable
        return self.variables[0]
    
    def _get_uniform_distribution(self, target_var: Variable) -> HistogramDistribution:
        """Create a uniform distribution for a variable in original domain."""
        n_bins = self.solver_config.max_bins
        
        if isinstance(target_var, ContinuousVariable):
            lower, upper = target_var.get_domain()
            bin_edges = np.linspace(lower, upper, n_bins + 1)
        elif isinstance(target_var, BinaryVariable):
            bin_edges = np.array([0.0, 0.5, 1.0])
            n_bins = 2
        else:
            bin_edges = np.arange(n_bins + 1, dtype=float)
        
        probs = np.ones(n_bins) / n_bins
        
        return HistogramDistribution(
            bin_edges=bin_edges.tolist(),
            bin_probabilities=probs.tolist(),
        )
    
    def build(
        self,
        target_variable: Optional[str] = None,
    ) -> tuple[HistogramDistribution, dict]:
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
        
        # Build histogram distribution (in original domain)
        distribution = HistogramDistribution(
            bin_edges=bin_edges.tolist(),
            bin_probabilities=marginal.tolist(),
        )
        
        # Add diagnostics
        info['joint_distribution'] = p_joint
        info['bin_edges_list'] = bin_edges_list  # Denormalized
        info['bin_edges_list_normalized'] = bin_edges_list_normalized  # Keep normalized for reference
        info['target_variable'] = target_var.name
        info['n_constraints'] = len(self.constraints)
        info['n_estimates'] = len(self.estimates)
        info['domains'] = {v.name: self.normalizer.get_domain(v.name) for v in self.variables}
        
        return distribution, info
    
    def get_all_marginals(self, info) -> dict[str, HistogramDistribution]:
        """Build marginal distributions for all variables.
        
        Uses pre-computed joint distribution and denormalized bin edges from info.
            
        Returns:
            Dictionary mapping variable names to their marginal distributions
        """
        p_joint, bin_edges_list = info['joint_distribution'], info['bin_edges_list']
        
        # Extract marginals for all variables
        marginals = {}
        for i, var in enumerate(self.variables):
            marginal = extract_marginal(p_joint, i)
            marginals[var.name] = HistogramDistribution(
                bin_edges=bin_edges_list[i].tolist(),
                bin_probabilities=marginal.tolist(),
            )
        
        return marginals
