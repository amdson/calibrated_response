"""Generate distributional queries from variables."""

from __future__ import annotations

import uuid
from typing import Optional

from calibrated_response.llm.base import LLMClient
from calibrated_response.models.variable import Variable, VariableType
from calibrated_response.models.query import (
    Query,
    QueryType,
    QueryList,
    MarginalQuery,
    ThresholdQuery,
    ConditionalQuery,
    QuantileQuery,
    ExpectationQuery,
)
from calibrated_response.generation.prompts import PROMPTS, format_variables_for_prompt


class QueryGenerator:
    """Generate distributional queries from variables."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client.
        
        Args:
            llm_client: Client for LLM queries
        """
        self.llm = llm_client
    
    def generate(
        self,
        question: str,
        variables: list[Variable],
        n_queries: int = 10,
        include_conditionals: bool = True,
    ) -> list[Query]:
        """Generate queries for the given variables.
        
        Args:
            question: The main forecasting question
            variables: List of relevant variables
            n_queries: Number of queries to generate
            include_conditionals: Whether to include conditional queries
            
        Returns:
            List of Query objects
        """
        prompts = PROMPTS["query_generation"]
        
        # Format variables for prompt
        var_list = [
            {
                "name": v.name,
                "description": v.description,
                "type": v.variable_type.value,
            }
            for v in variables
        ]
        variables_text = format_variables_for_prompt(var_list)
        
        user_prompt = prompts["user"].format(
            question=question,
            variables=variables_text,
            n_queries=n_queries,
        )
        
        try:
            result = self.llm.query_structured(
                prompt=user_prompt,
                response_model=QueryList,
                system_prompt=prompts["system"],
                temperature=0.7,
                max_tokens=1500+800*n_queries,
            )
            
            # Convert to Query objects
            queries = []
            for q_data in result.queries:
                query = self._create_query(q_data.model_dump(), question)
                if query:
                    queries.append(query)
        except Exception as e:
            # Fallback to basic queries
            queries = self._generate_basic_queries(question, variables, n_queries)
        
        # Ensure we have enough queries
        if len(queries) < n_queries:
            basic_queries = self._generate_basic_queries(
                question, variables, n_queries - len(queries)
            )
            queries.extend(basic_queries)
        
        return queries[:n_queries]
    
    def _create_query(self, data: dict, main_question: str) -> Optional[Query]:
        """Create a Query object from parsed data."""
        query_id = data.get("id", str(uuid.uuid4())[:8])
        text = data.get("text", "").strip()
        target_var = data.get("target_variable", "").strip()
        query_type_str = data.get("query_type", "marginal").lower()
        informativeness = data.get("informativeness", 0.5)
        
        if not text or not target_var:
            return None
        
        # Create appropriate query type
        if query_type_str == "threshold":
            threshold = data.get("threshold")
            direction = data.get("threshold_direction", "greater")
            if threshold is not None:
                return ThresholdQuery(
                    id=query_id,
                    text=text,
                    target_variable=target_var,
                    threshold=threshold,
                    direction=direction,
                    estimated_informativeness=informativeness,
                )
            else:
                return MarginalQuery(
                    id=query_id,
                    text=text,
                    target_variable=target_var,
                    estimated_informativeness=informativeness,
                )
        
        elif query_type_str == "conditional":
            condition_var = data.get("condition_variable")
            condition_text = data.get("condition_text", "")
            threshold = data.get("threshold")
            
            return ConditionalQuery(
                id=query_id,
                text=text,
                target_variable=target_var,
                condition_variable=condition_var or "",
                condition_value=True,  # Default to binary condition
                condition_text=condition_text,
                threshold=threshold,
                estimated_informativeness=informativeness,
            )
        
        elif query_type_str == "quantile":
            quantile = data.get("quantile", 0.5)
            return QuantileQuery(
                id=query_id,
                text=text,
                target_variable=target_var,
                quantile=quantile,
                estimated_informativeness=informativeness,
            )
        
        elif query_type_str == "expectation":
            return ExpectationQuery(
                id=query_id,
                text=text,
                target_variable=target_var,
                estimated_informativeness=informativeness,
            )
        
        else:  # marginal
            return MarginalQuery(
                id=query_id,
                text=text,
                target_variable=target_var,
                estimated_informativeness=informativeness,
            )
    
    def _generate_basic_queries(
        self,
        question: str,
        variables: list[Variable],
        n_queries: int,
    ) -> list[Query]:
        """Generate basic queries without LLM assistance."""
        queries = []
        query_idx = 0
        
        for var in variables:
            if len(queries) >= n_queries:
                break
            
            if var.is_target:
                # Generate quantile queries for target
                for q in [0.1, 0.5, 0.9]:
                    if len(queries) >= n_queries:
                        break
                    
                    quantile_name = {0.1: "10th percentile", 0.5: "median", 0.9: "90th percentile"}[q]
                    queries.append(QuantileQuery(
                        id=f"q{query_idx}",
                        text=f"What is the {quantile_name} estimate for: {question}",
                        target_variable=var.name,
                        quantile=q,
                        estimated_informativeness=0.7,
                    ))
                    query_idx += 1
            
            elif var.variable_type == VariableType.BINARY:
                # Generate marginal probability query
                queries.append(MarginalQuery(
                    id=f"q{query_idx}",
                    text=f"What is the probability that {var.description}?",
                    target_variable=var.name,
                    estimated_informativeness=var.estimated_importance,
                ))
                query_idx += 1
            
            elif var.variable_type == VariableType.CONTINUOUS:
                # Generate expectation query
                queries.append(ExpectationQuery(
                    id=f"q{query_idx}",
                    text=f"What is the expected value of {var.description}?",
                    target_variable=var.name,
                    estimated_informativeness=var.estimated_importance,
                ))
                query_idx += 1
        
        return queries
