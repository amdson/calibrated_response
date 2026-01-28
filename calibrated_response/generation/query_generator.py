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
    # Constrained models for structured output
    ConstrainedQueryList,
    ConstraintQueryType,
    ProbabilityQuerySpec,
    ExpectationQuerySpec,
    ConditionalProbabilityQuerySpec,
    ConditionalExpectationQuerySpec,
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
        
        # Format variables for prompt - include names, types, and ranges
        var_names = [v.name for v in variables]
        var_list = self._format_variable_info(variables)
        variables_text = format_variables_for_prompt(var_list)
        
        user_prompt = prompts["user"].format(
            question=question,
            variables=variables_text,
            n_queries=n_queries,
        )
        
        try:
            result = self.llm.query_structured(
                prompt=user_prompt,
                response_model=ConstrainedQueryList,
                system_prompt=prompts["system"],
                temperature=0.7,
                max_tokens=1500+800*n_queries,
            )
            
            # Validate and convert to Query objects
            queries = self._convert_constrained_queries(result, var_names, question)
            
        except Exception as e:
            print(f"LLM query generation failed: {e}")
            # Fallback to basic queries
            queries = self._generate_basic_queries(question, variables, n_queries)
        
        # Ensure we have enough queries
        if len(queries) < n_queries:
            basic_queries = self._generate_basic_queries(
                question, variables, n_queries - len(queries)
            )
            queries.extend(basic_queries)
        
        return queries[:n_queries]
    
    def _format_variable_info(self, variables: list[Variable]) -> list[dict]:
        """Extract variable info including ranges for prompt formatting."""
        var_list = []
        for v in variables:
            info = {
                "name": v.name,
                "description": v.description,
                "type": v.variable_type.value,
            }
            # Add range info if available (for ContinuousVariable)
            lower = getattr(v, 'lower_bound', None)
            upper = getattr(v, 'upper_bound', None)
            unit = getattr(v, 'unit', None)
            if lower is not None:
                info['lower_bound'] = lower
            if upper is not None:
                info['upper_bound'] = upper
            if unit:
                info['unit'] = unit
            var_list.append(info)
        return var_list
    
    def _convert_constrained_queries(
        self,
        result: ConstrainedQueryList,
        valid_var_names: list[str],
        main_question: str,
    ) -> list[Query]:
        """Convert ConstrainedQueryList to Query objects, validating variable names."""
        queries = []
        query_idx = 0
        
        # Process probability queries (threshold queries)
        for q in result.probability_queries:
            if q.target_variable not in valid_var_names:
                continue  # Skip invalid variable references
            direction = "greater" if q.exceeds else "less"
            queries.append(ThresholdQuery(
                id=f"q{query_idx}",
                text=q.text,
                target_variable=q.target_variable,
                threshold=q.threshold,
                direction=direction,
                estimated_informativeness=q.informativeness,
            ))
            query_idx += 1
        
        # Process expectation queries
        for q in result.expectation_queries:
            if q.target_variable not in valid_var_names:
                continue
            queries.append(ExpectationQuery(
                id=f"q{query_idx}",
                text=q.text,
                target_variable=q.target_variable,
                estimated_informativeness=q.informativeness,
            ))
            query_idx += 1
        
        # Process conditional probability queries
        for q in result.conditional_probability_queries:
            if q.target_variable not in valid_var_names:
                continue
            # Validate condition variables
            valid_conditions = all(c.variable in valid_var_names for c in q.conditions)
            if not valid_conditions:
                continue
            # Build condition text from threshold conditions
            condition_parts = []
            for c in q.conditions:
                direction = "at most" if c.is_upper_bound else "above"
                condition_parts.append(f"{c.variable} is {direction} {c.threshold}")
            condition_text = " and ".join(condition_parts)
            
            # Use ConditionalQuery for conditional probability
            queries.append(ConditionalQuery(
                id=f"q{query_idx}",
                text=q.text,
                target_variable=q.target_variable,
                condition_variable=q.conditions[0].variable,  # Primary condition
                condition_value=q.conditions[0].threshold,
                condition_text=condition_text,
                threshold=q.threshold,
                threshold_direction="greater" if q.exceeds else "less",
                estimated_informativeness=q.informativeness,
            ))
            query_idx += 1
        
        # Process conditional expectation queries
        for q in result.conditional_expectation_queries:
            if q.target_variable not in valid_var_names:
                continue
            valid_conditions = all(c.variable in valid_var_names for c in q.conditions)
            if not valid_conditions:
                continue
            condition_parts = []
            for c in q.conditions:
                direction = "at most" if c.is_upper_bound else "above"
                condition_parts.append(f"{c.variable} is {direction} {c.threshold}")
            condition_text = " and ".join(condition_parts)
            
            queries.append(ExpectationQuery(
                id=f"q{query_idx}",
                text=q.text,
                target_variable=q.target_variable,
                condition_text=condition_text,
                estimated_informativeness=q.informativeness,
            ))
            query_idx += 1
        
        return queries
    
    def get_raw_constrained_queries(
        self,
        question: str,
        variables: list[Variable],
        n_queries: int = 10,
    ) -> ConstrainedQueryList:
        """Get raw constrained query specs without conversion to Query objects.
        
        This is useful for directly mapping to constraint types.
        
        Args:
            question: The main forecasting question
            variables: List of relevant variables
            n_queries: Number of queries to generate
            
        Returns:
            ConstrainedQueryList with raw query specifications
        """
        prompts = PROMPTS["query_generation"]
        
        var_list = self._format_variable_info(variables)
        variables_text = format_variables_for_prompt(var_list)
        
        user_prompt = prompts["user"].format(
            question=question,
            variables=variables_text,
            n_queries=n_queries,
        )
        
        result = self.llm.query_structured(
            prompt=user_prompt,
            response_model=ConstrainedQueryList,
            system_prompt=prompts["system"],
            temperature=0.7,
            max_tokens=1500+800*n_queries,
        )
        
        return result
    
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
