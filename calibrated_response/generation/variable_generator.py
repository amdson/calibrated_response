"""Generate relevant variables for a forecasting question."""

from __future__ import annotations

from typing import Optional, Union

from calibrated_response.llm.base import LLMClient
from calibrated_response.models.variable import (
    Variable, 
    VariableList,
    BinaryVariable,
    ContinuousVariable,
)
from calibrated_response.generation.prompts import PROMPTS

class VariableGenerator:
    """Generate relevant variables for decomposing a forecasting question."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client.
        
        Args:
            llm_client: Client for LLM queries
        """
        self.llm = llm_client
    
    def generate(
        self,
        question: str,
        n_variables: int = 5,
    ) -> list[Union[BinaryVariable, ContinuousVariable]]:
        """Generate relevant variables for a forecasting question.
        
        Args:
            question: The forecasting question to decompose
            n_variables: Number of variables to generate
            include_target: Whether to include the target variable itself
            target_lower_bound: Lower bound for the target variable domain
            target_upper_bound: Upper bound for the target variable domain
            target_unit: Unit of measurement for the target variable
            
        Returns:
            List of typed Variable objects (BinaryVariable, ContinuousVariable, or DiscreteVariable)
        """
        prompts = PROMPTS["variable_generation"]
        
        user_prompt = prompts["user"].format(
            question=question,
            n_variables=n_variables,
        )

        try:
            result = self.llm.query_structured(
                prompt=user_prompt,
                response_model=VariableList, 
                system_prompt=prompts["system"],
                temperature=0.7,
                max_tokens=1000*n_variables+2500,
            )
            
        except Exception as e:
            raise ValueError(f"Failed to generate variables: {e}")
        
        return result.variables