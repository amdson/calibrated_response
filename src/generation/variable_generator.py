"""Generate relevant variables for a forecasting question."""

from __future__ import annotations

from typing import Optional

from calibrated_response.llm.base import LLMClient
from calibrated_response.models.variable import (
    Variable,
    VariableType,
    BinaryVariable,
    ContinuousVariable,
    DiscreteVariable,
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
        include_target: bool = True,
    ) -> list[Variable]:
        """Generate relevant variables for a forecasting question.
        
        Args:
            question: The forecasting question to decompose
            n_variables: Number of variables to generate
            include_target: Whether to include the target variable itself
            
        Returns:
            List of Variable objects
        """
        prompts = PROMPTS["variable_generation"]
        
        # Query the LLM
        response_schema = {
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "relevance": {"type": "string"},
                            "type": {"type": "string", "enum": ["binary", "continuous", "discrete"]},
                            "importance": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["name", "description", "type"],
                    },
                },
            },
            "required": ["variables"],
        }
        
        user_prompt = prompts["user"].format(
            question=question,
            n_variables=n_variables,
        )
        
        try:
            result = self.llm.query_structured(
                prompt=user_prompt,
                response_schema=response_schema,
                system_prompt=prompts["system"],
                temperature=0.7,
            )
        except Exception as e:
            # Fallback to unstructured query
            response = self.llm.query(
                prompt=user_prompt,
                system_prompt=prompts["system"],
                temperature=0.7,
            )
            # Try to parse the response
            from calibrated_response.llm.response_parser import ResponseParser
            import json
            
            # Attempt JSON extraction
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    result = json.loads(text[start:end])
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse variable generation response: {e}")
            else:
                raise ValueError(f"Failed to generate variables: {e}")
        
        # Convert to Variable objects
        variables = []
        for var_data in result.get("variables", []):
            var = self._create_variable(var_data)
            if var:
                variables.append(var)
        
        # Optionally add target variable
        if include_target:
            target = self._create_target_variable(question)
            variables.insert(0, target)
        
        return variables
    
    def _create_variable(self, data: dict) -> Optional[Variable]:
        """Create a Variable object from parsed data."""
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        var_type_str = data.get("type", "continuous").lower()
        relevance = data.get("relevance", "")
        importance = data.get("importance", 0.5)
        
        if not name or not description:
            return None
        
        # Map type string to VariableType
        type_map = {
            "binary": VariableType.BINARY,
            "continuous": VariableType.CONTINUOUS,
            "discrete": VariableType.DISCRETE,
            "ordinal": VariableType.ORDINAL,
        }
        var_type = type_map.get(var_type_str, VariableType.CONTINUOUS)
        
        # Create appropriate subclass
        if var_type == VariableType.BINARY:
            return BinaryVariable(
                name=name,
                description=description,
                relevance_explanation=relevance,
                estimated_importance=importance,
            )
        elif var_type == VariableType.CONTINUOUS:
            return ContinuousVariable(
                name=name,
                description=description,
                relevance_explanation=relevance,
                estimated_importance=importance,
            )
        else:
            return Variable(
                name=name,
                description=description,
                variable_type=var_type,
                relevance_explanation=relevance,
                estimated_importance=importance,
            )
    
    def _create_target_variable(self, question: str) -> ContinuousVariable:
        """Create the target variable from the main question."""
        return ContinuousVariable(
            name="target",
            description=f"The answer to: {question}",
            variable_type=VariableType.CONTINUOUS,
            is_target=True,
            estimated_importance=1.0,
        )
