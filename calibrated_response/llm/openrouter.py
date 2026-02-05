"""OpenRouter API client implementation."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from calibrated_response.llm.base import LLMClient, LLMResponse


T = TypeVar('T', bound=BaseModel)


class OpenRouterClient(LLMClient):
    """Client for OpenRouter API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
            model: Model to use (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3.5-sonnet')
            site_url: Optional URL for your site (for OpenRouter rankings)
            site_name: Optional name for your site (for OpenRouter rankings)
        """
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._model = model
        self._site_url = site_url
        self._site_name = site_name
        
        from openai import OpenAI
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        )
        
    @property
    def model_name(self) -> str:
        return self._model
    
    def _get_extra_headers(self) -> dict[str, str]:
        """Get extra headers for OpenRouter API."""
        headers = {}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._site_name:
            headers["X-Title"] = self._site_name
        return headers
    
    def query_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> T:
        """Query OpenRouter and parse response as JSON."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add JSON schema instruction to the prompt
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n```json\n{schema_json}\n```"
        )
        messages.append({"role": "user", "content": structured_prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=self._get_extra_headers(),
                response_format={"type": "json_object"},
            )
            
            text = response.choices[0].message.content
            if not text:
                raise ValueError("Empty response from OpenRouter")
            
            result = response_model.model_validate_json(text)
            return result
        except Exception as e:
            raise ValueError(f"Failed to get structured response from OpenRouter: {e}")
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a query to OpenRouter."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=self._get_extra_headers(),
            )
            
            # Extract response text
            choice = response.choices[0] if response.choices else None
            text = choice.message.content if choice and choice.message else ""
            
            # Get token counts if available
            prompt_tokens = None
            completion_tokens = None
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            
            # Get finish reason
            finish_reason = choice.finish_reason if choice else None
            
            return LLMResponse(
                text=text or "",
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                raw_response=response,
            )
        except Exception as e:
            raise ValueError(f"Failed to query OpenRouter: {e}")
