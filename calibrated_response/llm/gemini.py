"""Google Gemini API client implementation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from pydantic import Field

from calibrated_response.llm.base import LLMClient, LLMResponse


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
        self._client = None
        self._generative_model = None
        
    def _ensure_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. "
                    "Install with: pip install google-generativeai"
                )
            
            genai.configure(api_key=self._api_key)
            self._client = genai
            self._generative_model = genai.GenerativeModel(self._model)
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a query to Gemini."""
        self._ensure_client()
        
        # Combine system and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Configure generation
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        response = self._generative_model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        
        # Extract response text
        text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
        
        # Get token counts if available
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, 'prompt_token_count', None)
            completion_tokens = getattr(usage, 'candidates_token_count', None)
        
        return LLMResponse(
            text=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
            raw_response=response,
        )
    
    def query_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Query Gemini and parse response as JSON."""
        # Add JSON formatting instructions
        json_instruction = (
            "Respond ONLY with valid JSON matching this schema. "
            "Do not include any other text, markdown formatting, or explanation.\n"
            f"Schema: {json.dumps(response_schema, indent=2)}"
        )
        
        full_system = system_prompt or ""
        if full_system:
            full_system += "\n\n"
        full_system += json_instruction
        
        response = self.query(
            prompt=prompt,
            system_prompt=full_system,
            temperature=temperature,
            max_tokens=2048,
        )
        
        # Parse JSON from response
        text = response.text.strip()
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            text = json_match.group(1).strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to find JSON object in the text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse was: {text[:500]}")
