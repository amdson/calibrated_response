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
        user_prompt, kwargs = self._build_request(question, n_variables)
        try:
            result = self.llm.query_structured(prompt=user_prompt, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to generate variables: {e}")
        return result.variables

    async def agenerate(
        self,
        question: str,
        n_variables: int = 5,
        max_tokens_scale: float = 1.0,
    ) -> list[Union[BinaryVariable, ContinuousVariable]]:
        """Async twin of :meth:`generate` (same prompt and parsing).

        ``max_tokens_scale`` multiplies the token budget — retry loops should
        escalate it, since reasoning models occasionally spend the whole
        budget thinking and return truncated JSON."""
        user_prompt, kwargs = self._build_request(question, n_variables)
        kwargs["max_tokens"] = int(kwargs["max_tokens"] * max_tokens_scale)
        try:
            result = await self.llm.aquery_structured(prompt=user_prompt, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to generate variables: {e}")
        return result.variables

    @staticmethod
    def _build_request(question: str, n_variables: int):
        prompts = PROMPTS["variable_generation"]
        user_prompt = prompts["user"].format(
            question=question,
            n_variables=n_variables,
        )
        kwargs = dict(
            response_model=VariableList,
            system_prompt=prompts["system"],
            temperature=0.7,
            # generous headroom: reasoning models spend their thinking budget
            # from max_tokens too, and a long think + truncated JSON fails the
            # whole call (billing is per token used, not per token allowed)
            max_tokens=2000 * n_variables + 8000,
        )
        return user_prompt, kwargs