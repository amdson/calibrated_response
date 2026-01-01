"""Build distributions from query results using MaxEnt."""

from __future__ import annotations

from typing import Optional

import numpy as np

from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.models.query import Query, QueryResult, QueryType
from calibrated_response.maxent.constraints import ConstraintSet, ProbabilityConstraint
from calibrated_response.maxent.solver import MaxEntSolver, SolverConfig


class DistributionBuilder:
    """Build probability distributions from query results using MaxEnt."""
    
    def __init__(
        self,
        domain_min: float = 0.0,
        domain_max: float = 100.0,
        n_bins: int = 50,
        solver_config: Optional[SolverConfig] = None,
    ):
        """Initialize the distribution builder.
        
        Args:
            domain_min: Minimum value in the domain
            domain_max: Maximum value in the domain
            n_bins: Number of bins for discretization
            solver_config: Configuration for the MaxEnt solver
        """
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.n_bins = n_bins
        self.solver_config = solver_config or SolverConfig()
        self.solver = MaxEntSolver(self.solver_config)
    
    def build(
        self,
        queries: list[Query],
        results: list[QueryResult],
    ) -> tuple[HistogramDistribution, dict]:
        """Build a distribution from query results.
        
        Args:
            queries: List of queries that were asked
            results: Corresponding results from the LLM
            
        Returns:
            Tuple of (distribution, diagnostics)
        """
        # Create constraint set
        constraint_set = ConstraintSet(
            domain_min=self.domain_min,
            domain_max=self.domain_max,
            n_bins=self.n_bins,
        )
        
        # Convert query results to constraints
        for query, result in zip(queries, results):
            self._add_constraint(constraint_set, query, result)
        
        # Solve for maximum entropy distribution
        if len(constraint_set.constraints) == 0:
            # No constraints, return uniform distribution
            probs = np.ones(self.n_bins) / self.n_bins
            info = {'message': 'No constraints, returning uniform distribution'}
        else:
            probs, info = self.solver.solve(constraint_set)
        
        # Build histogram distribution
        bin_edges = np.linspace(self.domain_min, self.domain_max, self.n_bins + 1)
        distribution = HistogramDistribution(
            bin_edges=bin_edges.tolist(),
            bin_probabilities=probs.tolist(),
        )
        
        return distribution, info
    
    def build_iterative(
        self,
        queries: list[Query],
        results: list[QueryResult],
    ) -> tuple[HistogramDistribution, dict]:
        """Build a distribution iteratively, handling inconsistent constraints.
        
        Args:
            queries: List of queries that were asked
            results: Corresponding results from the LLM
            
        Returns:
            Tuple of (distribution, diagnostics)
        """
        constraint_set = ConstraintSet(
            domain_min=self.domain_min,
            domain_max=self.domain_max,
            n_bins=self.n_bins,
        )
        
        for query, result in zip(queries, results):
            self._add_constraint(constraint_set, query, result)
        
        if len(constraint_set.constraints) == 0:
            probs = np.ones(self.n_bins) / self.n_bins
            info = {'message': 'No constraints, returning uniform distribution'}
        else:
            probs, info = self.solver.solve_iterative(constraint_set)
        
        bin_edges = np.linspace(self.domain_min, self.domain_max, self.n_bins + 1)
        distribution = HistogramDistribution(
            bin_edges=bin_edges.tolist(),
            bin_probabilities=probs.tolist(),
        )
        
        return distribution, info
    
    def _add_constraint(
        self,
        constraint_set: ConstraintSet,
        query: Query,
        result: QueryResult,
    ) -> None:
        """Add a constraint from a query result."""
        confidence = result.confidence
        
        if query.query_type == QueryType.THRESHOLD:
            # P(X > threshold) = probability
            if result.probability is not None:
                from calibrated_response.models.query import ThresholdQuery
                if isinstance(query, ThresholdQuery):
                    constraint_set.add_threshold_constraint(
                        threshold=query.threshold,
                        probability=result.probability,
                        direction=query.direction,
                        confidence=confidence,
                        source_query_id=query.id,
                    )
        
        elif query.query_type == QueryType.QUANTILE:
            # Q(p) = value
            if result.value is not None:
                from calibrated_response.models.query import QuantileQuery
                if isinstance(query, QuantileQuery):
                    constraint_set.add_quantile_constraint(
                        quantile=query.quantile,
                        value=result.value,
                        confidence=confidence,
                        source_query_id=query.id,
                    )
        
        elif query.query_type == QueryType.EXPECTATION:
            # E[X] = value
            if result.value is not None:
                constraint_set.add_mean_constraint(
                    mean=result.value,
                    confidence=confidence,
                    source_query_id=query.id,
                )
        
        elif query.query_type == QueryType.MARGINAL:
            # For binary variables, this is P(X = 1)
            if result.probability is not None:
                # Interpret as probability in upper half of domain
                midpoint = (self.domain_min + self.domain_max) / 2
                constraint_set.add_threshold_constraint(
                    threshold=midpoint,
                    probability=result.probability,
                    direction="greater",
                    confidence=confidence * 0.5,  # Lower confidence for interpretation
                    source_query_id=query.id,
                )
        
        elif query.query_type == QueryType.CONDITIONAL:
            # Conditional constraints are more complex
            # For now, treat as soft probability constraint with lower confidence
            if result.probability is not None:
                from calibrated_response.models.query import ConditionalQuery
                if isinstance(query, ConditionalQuery) and query.threshold is not None:
                    constraint_set.add_threshold_constraint(
                        threshold=query.threshold,
                        probability=result.probability,
                        direction=query.threshold_direction,
                        confidence=confidence * 0.7,  # Lower confidence for conditionals
                        source_query_id=query.id,
                    )
    
    def update_domain(
        self,
        queries: list[Query],
        results: list[QueryResult],
    ) -> None:
        """Update domain bounds based on query results.
        
        Automatically adjusts domain based on quantile and expectation
        responses to ensure the domain covers the likely range.
        """
        values = []
        
        for query, result in zip(queries, results):
            if result.value is not None:
                values.append(result.value)
            
            if query.query_type == QueryType.THRESHOLD:
                from calibrated_response.models.query import ThresholdQuery
                if isinstance(query, ThresholdQuery):
                    values.append(query.threshold)
        
        if values:
            min_val = min(values)
            max_val = max(values)
            
            # Expand domain to cover values with some margin
            margin = (max_val - min_val) * 0.2 if max_val > min_val else abs(max_val) * 0.5
            
            self.domain_min = min(self.domain_min, min_val - margin)
            self.domain_max = max(self.domain_max, max_val + margin)
            
            # Ensure non-negative if all values are non-negative
            if all(v >= 0 for v in values):
                self.domain_min = max(0, self.domain_min)
