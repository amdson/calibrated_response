"""Abstract base class for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Response from an LLM query."""
    
    text: str = Field(..., description="Raw response text")
    model: str = Field(..., description="Model that generated the response")
    
    # Usage statistics
    prompt_tokens: Optional[int] = Field(None, description="Tokens in the prompt")
    completion_tokens: Optional[int] = Field(None, description="Tokens in the completion")
    
    # Metadata
    finish_reason: Optional[str] = Field(None, description="Why generation stopped")
    raw_response: Optional[Any] = Field(None, description="Raw API response for debugging")


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a query to the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse containing the model's response
        """
        pass
    
    @abstractmethod
    def query_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Query the LLM and parse response as structured JSON.
        
        Args:
            prompt: The user prompt
            response_schema: JSON schema for expected response
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response matching the schema
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model being used."""
        pass
