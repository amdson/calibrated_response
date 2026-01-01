"""Main prediction pipeline."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from calibrated_response.llm.base import LLMClient
from calibrated_response.generation import VariableGenerator, QueryGenerator, QueryAnswerer
from calibrated_response.maxent import DistributionBuilder, SolverConfig
from calibrated_response.models.distribution import HistogramDistribution


class Pipeline:
    """End-to-end prediction pipeline.
    
    Combines variable generation, query generation, query answering,
    and maximum entropy distribution building into a single interface.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        n_variables: int = 5,
        n_queries: int = 10,
        n_bins: int = 50,
        solver_config: Optional[SolverConfig] = None,
    ):
        """Initialize the pipeline.
        
        Args:
            llm_client: LLM client for generation
            n_variables: Number of variables to generate
            n_queries: Number of queries to generate
            n_bins: Number of bins for distribution discretization
            solver_config: Configuration for MaxEnt solver
        """
        self.llm_client = llm_client
        self.n_variables = n_variables
        self.n_queries = n_queries
        self.n_bins = n_bins
        
        # Initialize components
        self.variable_generator = VariableGenerator(llm_client)
        self.query_generator = QueryGenerator(llm_client)
        self.query_answerer = QueryAnswerer(llm_client)
        self.solver_config = solver_config or SolverConfig()
    
    def predict(
        self,
        question: str,
        domain_min: float = 0.0,
        domain_max: float = 100.0,
        context: Optional[str] = None,
    ) -> Tuple[HistogramDistribution, dict]:
        """Make a prediction for a forecasting question.
        
        Args:
            question: The forecasting question
            domain_min: Minimum value for the answer domain
            domain_max: Maximum value for the answer domain
            context: Optional additional context
            
        Returns:
            Tuple of (distribution, info_dict)
        """
        info = {
            'question': question,
            'domain': (domain_min, domain_max),
        }
        
        # Step 1: Generate relevant variables
        variables = self.variable_generator.generate(
            question=question,
            n_variables=self.n_variables,
            include_target=True,
        )
        info['n_variables'] = len(variables)
        info['variables'] = [
            {'name': v.name, 'description': v.description}
            for v in variables
        ]
        
        # Step 2: Generate queries
        queries = self.query_generator.generate(
            question=question,
            variables=variables,
            n_queries=self.n_queries,
            include_conditionals=True,
        )
        info['n_queries'] = len(queries)
        info['queries'] = [
            {'id': q.id, 'text': q.text, 'type': q.query_type.value}
            for q in queries
        ]
        
        # Step 3: Answer queries
        results = self.query_answerer.answer_batch(
            queries=queries,
            context=question,
        )
        info['query_results'] = [
            {
                'query_id': r.query_id,
                'probability': r.probability,
                'value': r.value,
                'confidence': r.confidence,
            }
            for r in results
        ]
        
        # Step 4: Build distribution
        builder = DistributionBuilder(
            domain_min=domain_min,
            domain_max=domain_max,
            n_bins=self.n_bins,
            solver_config=self.solver_config,
        )
        
        # Update domain based on query results
        builder.update_domain(queries, results)
        
        # Build the distribution
        distribution, solver_info = builder.build_iterative(queries, results)
        info['solver_info'] = solver_info
        
        return distribution, info
    
    def predict_with_variables(
        self,
        question: str,
        variables: list[dict],
        domain_min: float = 0.0,
        domain_max: float = 100.0,
    ) -> Tuple[HistogramDistribution, dict]:
        """Make a prediction with pre-specified variables.
        
        Args:
            question: The forecasting question
            variables: List of variable dicts with 'name' and 'description'
            domain_min: Minimum value for the answer domain
            domain_max: Maximum value for the answer domain
            
        Returns:
            Tuple of (distribution, info_dict)
        """
        from calibrated_response.models.variable import Variable, VariableType
        
        # Convert to Variable objects
        var_objects = []
        for v in variables:
            var_objects.append(Variable(
                name=v.get('name', 'unnamed'),
                description=v.get('description', ''),
                variable_type=VariableType(v.get('type', 'continuous')),
            ))
        
        info = {
            'question': question,
            'domain': (domain_min, domain_max),
            'n_variables': len(var_objects),
        }
        
        # Generate queries
        queries = self.query_generator.generate(
            question=question,
            variables=var_objects,
            n_queries=self.n_queries,
        )
        info['n_queries'] = len(queries)
        
        # Answer queries
        results = self.query_answerer.answer_batch(queries, context=question)
        
        # Build distribution
        builder = DistributionBuilder(
            domain_min=domain_min,
            domain_max=domain_max,
            n_bins=self.n_bins,
            solver_config=self.solver_config,
        )
        builder.update_domain(queries, results)
        distribution, solver_info = builder.build_iterative(queries, results)
        info['solver_info'] = solver_info
        
        return distribution, info
