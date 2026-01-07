"""Generate relevant variables for a forecasting question."""

from __future__ import annotations

from typing import Optional

from calibrated_response.llm.base import LLMClient
from calibrated_response.models.variable import Variable, VariableList
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
        include_target: bool = True,
    ):
        """Generate relevant variables for a forecasting question.
        
        Args:
            question: The forecasting question to decompose
            n_variables: Number of variables to generate
            include_target: Whether to include the target variable itself
            
        Returns:
            List of Variable objects
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
                max_tokens=1000*n_variables+1200,
            )
            
            # Convert to Variable objects from Pydantic model
            variables = []
            for var in result.variables:
                parsed_var = self._create_variable(var.model_dump())
                if parsed_var:
                    variables.append(parsed_var)
        except Exception as e:
            raise ValueError(f"Failed to generate variables: {e}")
        
        # Optionally add target variable
        if include_target:
            target = self._create_target_variable(question)
            variables.insert(0, target)
        
        return variables
    
    def _create_variable(self, data: dict) -> Optional[Variable]:
        """Create a Variable object from parsed data."""
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        var_type = data.get("type", "continuous").lower()
        # relevance = data.get("relevance")
        importance = data.get("importance", 0.5)
        
        if not name or not description:
            return None
        
        return Variable(
            name=name,
            description=description,
            type=var_type,
            # relevance=relevance,
            importance=importance,
            is_target=False,
        )
    
    def _create_target_variable(self, question: str) -> Variable:
        """Create the target variable from the main question."""
        return Variable(
            name="target",
            description=f"The answer to: {question}",
            type="continuous",
            # relevance=None,
            importance=1.0,
            is_target=True,
        )
