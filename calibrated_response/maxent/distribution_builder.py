"""Build distributions from query results using MaxEnt."""

from __future__ import annotations

from typing import Optional
import uuid

import numpy as np

from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.models.query import Query, QueryResult, QueryType
from calibrated_response.models.variable import Variable, ContinuousVariable
from calibrated_response.maxent.constraints import (
    Constraint,
    ProbabilityConstraint,
    MeanConstraint,
    QuantileConstraint,
    ConditionalQuantileConstraint,
    ConditionalMeanConstraint,
)
from calibrated_response.maxent.multivariate_solver import (
    MultivariateMaxEntSolver,
    JAXSolverConfig,
    extract_marginal,
)


class DistributionBuilder:
    """Build probability distributions from query results using multivariate MaxEnt."""
    
    def __init__(
        self,
        variables: list[Variable],
        solver_config: Optional[JAXSolverConfig] = None,
    ):
        """Initialize the distribution builder.
        
        Args:
            variables: List of Variable objects defining the joint distribution
            solver_config: Configuration for the JAX MaxEnt solver
        """
        self.variables = variables
        self.var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
        self.solver_config = solver_config or JAXSolverConfig()
        self.solver = MultivariateMaxEntSolver(self.solver_config)
    
    def build(
        self,
        queries: list[Query],
        results: list[QueryResult],
        target_variable: Optional[str] = None,
    ) -> tuple[HistogramDistribution, dict]:
        """Build a distribution from query results.
        
        Args:
            queries: List of queries that were asked
            results: Corresponding results from the LLM
            target_variable: Name of the variable to extract marginal for.
                           If None, returns the first variable marked as target,
                           or the first variable if none are marked.
            
        Returns:
            Tuple of (distribution, diagnostics)
        """
        # Convert query results to constraints
        constraints = []
        for query, result in zip(queries, results):
            constraint = self._create_constraint(query, result)
            if constraint is not None:
                constraints.append(constraint)
        
        # Solve for maximum entropy distribution
        if len(constraints) == 0:
            # No constraints, return uniform distribution
            # Use default bins for the target variable
            target_var = self._get_target_variable(target_variable)
            target_idx = self.var_name_to_idx[target_var.name]
            n_bins = self.solver_config.max_bins
            
            if isinstance(target_var, ContinuousVariable):
                lower, upper = target_var.get_domain()
                bin_edges = np.linspace(lower, upper, n_bins + 1)
            else:
                bin_edges = np.arange(n_bins + 1, dtype=float)
            
            probs = np.ones(n_bins) / n_bins
            info = {'message': 'No constraints, returning uniform distribution'}
            
            distribution = HistogramDistribution(
                bin_edges=bin_edges.tolist(),
                bin_probabilities=probs.tolist(),
            )
            return distribution, info
        
        # Solve using multivariate MaxEnt
        p_joint, bin_edges_list, info = self.solver.solve(
            variables=self.variables,
            constraints=constraints,
        )
        
        # Extract marginal for the target variable
        target_var = self._get_target_variable(target_variable)
        target_idx = self.var_name_to_idx[target_var.name]
        marginal = extract_marginal(p_joint, target_idx)
        bin_edges = bin_edges_list[target_idx]
        
        # Build histogram distribution
        distribution = HistogramDistribution(
            bin_edges=bin_edges.tolist(),
            bin_probabilities=marginal.tolist(),
        )
        
        # Add joint distribution to info
        info['joint_distribution'] = p_joint
        info['bin_edges_list'] = bin_edges_list
        info['target_variable'] = target_var.name
        info['n_constraints'] = len(constraints)
        
        return distribution, info
    
    def _get_target_variable(self, target_variable: Optional[str]) -> Variable:
        """Get the target variable for marginal extraction."""
        if target_variable is not None:
            for v in self.variables:
                if v.name == target_variable:
                    return v
            raise ValueError(f"Variable '{target_variable}' not found")
        
        # Look for variable marked as target
        for v in self.variables:
            if getattr(v, 'is_target', False):
                return v
        
        # Default to first variable
        return self.variables[0]
    
    def _create_constraint(
        self,
        query: Query,
        result: QueryResult,
    ) -> Optional[Constraint]:
        """Create a constraint from a query result."""
        confidence = result.confidence
        constraint_id = f"c_{query.id}_{uuid.uuid4().hex[:6]}"
        
        # Get target variable
        target_var_name = query.target_variable
        if target_var_name not in self.var_name_to_idx:
            return None  # Skip if variable not found
        
        target_var = self.variables[self.var_name_to_idx[target_var_name]]
        
        if query.query_type == QueryType.THRESHOLD:
            # P(X > threshold) = probability
            if result.probability is not None:
                from calibrated_response.models.query import ThresholdQuery
                if isinstance(query, ThresholdQuery):
                    # Get domain bounds for the variable
                    if isinstance(target_var, ContinuousVariable):
                        domain_min, domain_max = target_var.get_domain()
                    else:
                        domain_min, domain_max = 0.0, 1.0
                    
                    return ProbabilityConstraint.from_threshold(
                        id=constraint_id,
                        target_variable=target_var,
                        threshold=query.threshold,
                        probability=result.probability,
                        direction=query.direction,
                        domain_min=domain_min,
                        domain_max=domain_max,
                        confidence=confidence,
                        source_query_id=query.id,
                    )
        
        elif query.query_type == QueryType.QUANTILE:
            # Q(p) = value - interpreted as P(X <= value) = quantile
            if result.value is not None:
                from calibrated_response.models.query import QuantileQuery
                if isinstance(query, QuantileQuery):
                    # Check if this is a conditional quantile
                    if query.condition_text:
                        # Parse condition from condition_text if possible
                        # For now, treat as regular quantile with lower confidence
                        return QuantileConstraint(
                            id=constraint_id,
                            target_variable=target_var,
                            quantile=query.quantile,
                            value=result.value,
                            confidence=confidence * 0.7,
                            source_query_id=query.id,
                        )
                    else:
                        return QuantileConstraint(
                            id=constraint_id,
                            target_variable=target_var,
                            quantile=query.quantile,
                            value=result.value,
                            confidence=confidence,
                            source_query_id=query.id,
                        )
        
        elif query.query_type == QueryType.EXPECTATION:
            # E[X] = value
            if result.value is not None:
                from calibrated_response.models.query import ExpectationQuery
                if isinstance(query, ExpectationQuery):
                    # Check if this is a conditional expectation
                    if query.condition_text:
                        # For conditional expectations, we need to parse the conditions
                        # This is a simplified handling - treat as regular mean with lower confidence
                        return MeanConstraint(
                            id=constraint_id,
                            target_variable=target_var,
                            mean=result.value,
                            confidence=confidence * 0.7,
                            source_query_id=query.id,
                        )
                    else:
                        return MeanConstraint(
                            id=constraint_id,
                            target_variable=target_var,
                            mean=result.value,
                            confidence=confidence,
                            source_query_id=query.id,
                        )
        
        elif query.query_type == QueryType.CONDITIONAL:
            # Conditional probability P(X > t | conditions)
            if result.probability is not None:
                from calibrated_response.models.query import ConditionalQuery
                if isinstance(query, ConditionalQuery) and query.threshold is not None:
                    # Get condition variable
                    cond_var_name = query.condition_variable
                    if cond_var_name in self.var_name_to_idx:
                        cond_var = self.variables[self.var_name_to_idx[cond_var_name]]
                        
                        # Determine condition bounds
                        cond_value = query.condition_value
                        if isinstance(cond_value, bool):
                            # Binary condition - treat as threshold at 0.5
                            cond_threshold = 0.5
                            is_lower_bound = cond_value  # True = above threshold
                        else:
                            cond_threshold = float(cond_value)
                            is_lower_bound = True  # Default: condition is "above threshold"
                        
                        # Get domain bounds
                        if isinstance(target_var, ContinuousVariable):
                            domain_min, domain_max = target_var.get_domain()
                        else:
                            domain_min, domain_max = 0.0, 1.0
                        
                        # Create conditional quantile constraint
                        # P(X > threshold | condition) = probability
                        # This means the quantile of X at threshold = 1 - probability
                        direction = getattr(query, 'threshold_direction', 'greater')
                        if direction == 'greater':
                            quantile_level = 1.0 - result.probability
                        else:
                            quantile_level = result.probability
                        
                        return ConditionalQuantileConstraint(
                            id=constraint_id,
                            target_variable=target_var,
                            quantile=quantile_level,
                            value=query.threshold,
                            condition_variables=[cond_var],
                            condition_values=[cond_threshold],
                            is_lower_bound=[not is_lower_bound],  # Flip for "above" condition
                            confidence=confidence,
                            source_query_id=query.id,
                        )
        
        elif query.query_type == QueryType.MARGINAL:
            # For binary variables, P(X = 1)
            if result.probability is not None:
                # Create probability constraint for upper half
                if isinstance(target_var, ContinuousVariable):
                    domain_min, domain_max = target_var.get_domain()
                    midpoint = (domain_min + domain_max) / 2
                else:
                    domain_min, domain_max = 0.0, 1.0
                    midpoint = 0.5
                
                return ProbabilityConstraint(
                    id=constraint_id,
                    target_variable=target_var,
                    lower_bound=midpoint,
                    upper_bound=domain_max,
                    probability=result.probability,
                    confidence=confidence * 0.5,
                    source_query_id=query.id,
                )
        
        return None
    
    def get_all_marginals(
        self,
        queries: list[Query],
        results: list[QueryResult],
    ) -> dict[str, HistogramDistribution]:
        """Build marginal distributions for all variables.
        
        Args:
            queries: List of queries that were asked
            results: Corresponding results from the LLM
            
        Returns:
            Dictionary mapping variable names to their marginal distributions
        """
        # Convert query results to constraints
        constraints = []
        for query, result in zip(queries, results):
            constraint = self._create_constraint(query, result)
            if constraint is not None:
                constraints.append(constraint)
        
        if len(constraints) == 0:
            # Return uniform distributions for all variables
            marginals = {}
            for var in self.variables:
                n_bins = self.solver_config.max_bins
                if isinstance(var, ContinuousVariable):
                    lower, upper = var.get_domain()
                    bin_edges = np.linspace(lower, upper, n_bins + 1)
                else:
                    bin_edges = np.arange(n_bins + 1, dtype=float)
                probs = np.ones(n_bins) / n_bins
                marginals[var.name] = HistogramDistribution(
                    bin_edges=bin_edges.tolist(),
                    bin_probabilities=probs.tolist(),
                )
            return marginals
        
        # Solve using multivariate MaxEnt
        p_joint, bin_edges_list, info = self.solver.solve(
            variables=self.variables,
            constraints=constraints,
        )
        
        # Extract marginals for all variables
        marginals = {}
        for i, var in enumerate(self.variables):
            marginal = extract_marginal(p_joint, i)
            marginals[var.name] = HistogramDistribution(
                bin_edges=bin_edges_list[i].tolist(),
                bin_probabilities=marginal.tolist(),
            )
        
        return marginals
