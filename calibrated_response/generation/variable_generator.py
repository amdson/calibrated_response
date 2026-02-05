"""Generate relevant variables for a forecasting question."""

from __future__ import annotations

from typing import Optional, Union

from calibrated_response.llm.base import LLMClient
from calibrated_response.models.variable import (
    Variable, 
    VariableList,
    BinaryVariable,
    ContinuousVariable,
    # DiscreteVariable,
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
            
            # Convert to typed Variable objects from Pydantic model
            variables = []
            for var in result.variables:
                parsed_var = self._create_variable(var.model_dump())
                if parsed_var:
                    variables.append(parsed_var)
        except Exception as e:
            raise ValueError(f"Failed to generate variables: {e}")
        
        return variables
    
    def _create_variable(self, data: dict) -> Optional[Union[BinaryVariable, ContinuousVariable, DiscreteVariable]]:
        """Create a typed Variable object from parsed data."""
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        var_type = data.get("type", "continuous").lower()
        importance = data.get("importance", 0.5)
        
        if not name or not description:
            return None
        
        if var_type == "binary":
            return BinaryVariable(
                name=name,
                description=description,
                importance=importance,
                is_target=False,
                yes_label=data.get("yes_label", "yes"),
                no_label=data.get("no_label", "no"),
            )
        elif var_type == "continuous":
            return ContinuousVariable(
                name=name,
                description=description,
                importance=importance,
                is_target=False,
                lower_bound=data.get("lower_bound"),
                upper_bound=data.get("upper_bound"),
                unit=data.get("unit"),
            )
        elif var_type == "discrete":
            return DiscreteVariable(
                name=name,
                description=description,
                importance=importance,
                is_target=False,
                categories=data.get("categories", []),
            )
        else:
            # Default to continuous
            return ContinuousVariable(
                name=name,
                description=description,
                importance=importance,
                is_target=False,
            )
    
    def _create_target_variable(self, question: str, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None, unit: Optional[str] = None) -> ContinuousVariable:
        """Create the target variable from the main question."""
        return ContinuousVariable(
            name="target",
            description=f"The answer to: {question}",
            importance=1.0,
            is_target=True,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            unit=unit,
        )
