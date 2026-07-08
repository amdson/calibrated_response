"""Generate estimates using natural language format for efficiency."""

from __future__ import annotations

from typing import Sequence

from calibrated_response.llm.base import LLMClient
from calibrated_response.models.variable import Variable
from calibrated_response.models.query import Estimate, EstimateUnion
from calibrated_response.models.natural_response import NaturalEstimateList
from calibrated_response.generation.prompts import PROMPTS, format_variables_for_prompt


class NaturalEstimateGenerator:
    """Generate estimates using natural language format.
    
    This generator uses a more token-efficient format where the LLM outputs
    estimates in mathematical notation like:
    - P(X > 10) = 0.5
    - E[Cost] = 100.0
    - P(A > 5 | B = True) = 0.7
    - E[X | Y > 10] = 50.0
    - Cov(X, Y) = 25.0
    - Corr(X, Y) = 0.8
    - Corr(X, Y) > 0.2
    - Var(X) = 16.0
    - Var(X | Y > 10) = 9.0
    - Var(X | Y = True) < Var(X | Y = False)
    - E[X | Y > 10] > E[X | Y <= 10]
    - Independence: $Indep(X, Y)$ 
    - $P(Y | do(X=x))$ 
    - Likelihood Ratio/Odds: $P(A) / P(B) = 2.0$
    
    These are then parsed into structured Estimate objects.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client.
        
        Args:
            llm_client: Client for LLM queries
        """
        self.llm = llm_client
        self.last_result = None  # Store last raw LLM result for debugging
    
    def generate(
        self,
        question: str,
        variables: Sequence[Variable],
        num_estimates: int = 15,
    ) -> Sequence[EstimateUnion]:
        """Generate estimates for variables using natural language format.
        
        Args:
            question: The forecasting question to decompose
            variables: List of variables to consider for estimates
            num_estimates: Number of estimates to generate
            
        Returns:
            List of typed Estimate objects (ProbabilityEstimate, ExpectationEstimate, etc.)
        """
        user_prompt, kwargs = self._build_request(question, variables, num_estimates)
        try:
            result = self.llm.query_structured(prompt=user_prompt, **kwargs)
            self.last_result = result
            # Convert natural estimates to structured format
            return result.convert_all()
        except Exception as e:
            raise ValueError(f"Failed to generate natural estimates: {e}")

    async def agenerate(
        self,
        question: str,
        variables: Sequence[Variable],
        num_estimates: int = 15,
        max_tokens_scale: float = 1.0,
    ) -> Sequence[EstimateUnion]:
        """Async twin of :meth:`generate` (same prompt and parsing).

        ``max_tokens_scale`` multiplies the token budget — retry loops should
        escalate it, since reasoning models occasionally spend the whole
        budget thinking and return truncated JSON."""
        user_prompt, kwargs = self._build_request(question, variables, num_estimates)
        kwargs["max_tokens"] = int(kwargs["max_tokens"] * max_tokens_scale)
        try:
            result = await self.llm.aquery_structured(prompt=user_prompt, **kwargs)
            self.last_result = result
            return result.convert_all()
        except Exception as e:
            raise ValueError(f"Failed to generate natural estimates: {e}")

    @staticmethod
    def _build_request(question: str, variables: Sequence[Variable],
                       num_estimates: int):
        prompts = PROMPTS["natural_estimate_generation"]

        # Format variables for the prompt
        var_info = []
        for v in variables:
            info = {
                "name": v.name,
                "description": v.description,
                "type": v.type.value,
            }
            if hasattr(v, 'lower_bound'):
                info['lower_bound'] = v.lower_bound
            if hasattr(v, 'upper_bound'):
                info['upper_bound'] = v.upper_bound
            if hasattr(v, 'unit'):
                info['unit'] = v.unit
            var_info.append(info)

        variables_text = format_variables_for_prompt(var_info)

        user_prompt = prompts["user"].format(
            question=question,
            variables=variables_text,
            num_estimates=num_estimates,
        )
        kwargs = dict(
            response_model=NaturalEstimateList,
            system_prompt=prompts["system"],
            temperature=0.7,
            # headroom for reasoning tokens (counted against max_tokens by
            # thinking models); truncated JSON fails the whole call
            max_tokens=500 * num_estimates + 6000,
        )
        return user_prompt, kwargs
