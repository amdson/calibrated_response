"""Generate relevant variables for a forecasting question."""

from __future__ import annotations

from typing import Optional, Sequence, Union

from calibrated_response.llm.base import LLMClient
from calibrated_response.models.variable import (
    Variable, 
    VariableList,
    BinaryVariable,
    ContinuousVariable,
)
from calibrated_response.models.query import (
    Proposition, EqualityProposition, InequalityProposition,
    Estimate, ProbabilityEstimate, ExpectationEstimate,
    ConditionalProbabilityEstimate, ConditionalExpectationEstimate,
    EstimateList,
)


from calibrated_response.generation.prompts import PROMPTS

class EstimateGenerator:
    """Generate estimates of relationships between variables for a forecasting question."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client.
        
        Args:
            llm_client: Client for LLM queries
        """
        self.llm = llm_client
    
    def generate(
        self,
        question: str,
        variables: Sequence[Variable],
        num_estimates: int = 15,
    ) -> Sequence[Estimate]:
        """Generate relevant variables for a forecasting question.
        
        Args:
            question: The forecasting question to decompose
            variables: List of variables to consider for estimates
            num_estimates: Number of estimates to generate
            
        Returns:
            List of typed Estimate objects (ProbabilityEstimate, ExpectationEstimate, etc.)
        """
        prompts = PROMPTS["estimate_generation"]
        
        user_prompt = prompts["user"].format(
            question=question,
            variables=variables,
            num_estimates=num_estimates,
        )

        try:
            result = self.llm.query_structured(
                prompt=user_prompt,
                response_model=EstimateList, 
                system_prompt=prompts["system"],
                temperature=0.7,
                max_tokens=1000*len(variables)+2500,
            )
            
        except Exception as e:
            raise ValueError(f"Failed to generate estimates: {e}")
        
        return result.estimates