"""Google Gemini API client implementation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import Field, BaseModel

from calibrated_response.llm.base import LLMClient, LLMResponse


T = TypeVar('T', bound=BaseModel)


class GeminiClient(LLMClient):
    """Client for Google's Gemini API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
    ):
        """Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
            model: Model to use (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
        """
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._model = model
        import google.genai as genai
        # genai.configure(api_key=self._api_key)
        self._client = genai.Client(api_key=self._api_key)
        # self._generative_model = genai.GenerativeModel(self._model)
        
    @property
    def model_name(self) -> str:
        return self._model
    
    def query_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> T:
        """Query Gemini and parse response as JSON."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=full_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_model,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            if not response.text:
                raise ValueError("Empty response from Gemini")
            result = response_model.model_validate_json(response.text)
            return result
        except Exception as e:
            raise ValueError(f"Failed to get structured response from Gemini: {e}")
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a query to Gemini."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=full_prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            
            # Extract response text
            text = response.text if response.text else ""
            
            # Get token counts if available
            prompt_tokens = None
            completion_tokens = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', None)
                completion_tokens = getattr(usage, 'candidates_token_count', None)
            
            return LLMResponse(
                text=text,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=None,
                raw_response=response,
            )
        except Exception as e:
            raise ValueError(f"Failed to query Gemini: {e}")
    