"""Answer distributional queries using LLM."""

from __future__ import annotations

from typing import Optional

from calibrated_response.llm.base import LLMClient
from calibrated_response.llm.response_parser import ResponseParser
from calibrated_response.models.query import (
    Query,
    QueryType,
    QueryResult,
    ThresholdQuery,
    ConditionalQuery,
    QuantileQuery,
    ExpectationQuery,
)
from calibrated_response.generation.prompts import PROMPTS


class QueryAnswerer:
    """Answer distributional queries using an LLM."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client.
        
        Args:
            llm_client: Client for LLM queries
        """
        self.llm = llm_client
        self.parser = ResponseParser()
    
    def answer(
        self,
        query: Query,
        context: Optional[str] = None,
    ) -> QueryResult:
        """Answer a distributional query.
        
        Args:
            query: The query to answer
            context: Optional additional context (e.g., the main question)
            
        Returns:
            QueryResult containing the answer
        """
        # Select appropriate prompt based on query type
        if query.query_type == QueryType.THRESHOLD:
            return self._answer_threshold(query, context)
        elif query.query_type == QueryType.CONDITIONAL:
            return self._answer_conditional(query, context)
        elif query.query_type in (QueryType.QUANTILE, QueryType.EXPECTATION):
            return self._answer_numeric(query, context)
        else:
            return self._answer_probability(query, context)
    
    def answer_batch(
        self,
        queries: list[Query],
        context: Optional[str] = None,
    ) -> list[QueryResult]:
        """Answer multiple queries.
        
        Args:
            queries: List of queries to answer
            context: Optional additional context
            
        Returns:
            List of QueryResult objects
        """
        results = []
        for query in queries:
            result = self.answer(query, context)
            results.append(result)
        return results
    
    def _answer_threshold(
        self,
        query: ThresholdQuery,
        context: Optional[str],
    ) -> QueryResult:
        """Answer a threshold probability query."""
        prompts = PROMPTS["threshold_probability"]
        
        direction_word = "greater than" if query.direction == "greater" else "less than"
        
        user_prompt = prompts["user"].format(
            context=context or "General forecasting question",
            variable=query.target_variable,
            direction=direction_word,
            threshold=query.threshold,
        )
        
        response = self.llm.query(
            prompt=user_prompt,
            system_prompt=prompts["system"],
            temperature=0.5,
        )
        
        prob, confidence = self.parser.parse_probability_response(response.text)
        
        return QueryResult(
            query_id=query.id,
            probability=prob,
            confidence=confidence,
            raw_response=response.text,
        )
    
    def _answer_conditional(
        self,
        query: ConditionalQuery,
        context: Optional[str],
    ) -> QueryResult:
        """Answer a conditional probability query."""
        prompts = PROMPTS["conditional_probability"]
        
        user_prompt = prompts["user"].format(
            main_question=context or "Forecasting question",
            condition=query.condition_text,
            query=query.text,
        )
        
        response = self.llm.query(
            prompt=user_prompt,
            system_prompt=prompts["system"],
            temperature=0.5,
        )
        
        prob, confidence = self.parser.parse_probability_response(response.text)
        
        return QueryResult(
            query_id=query.id,
            probability=prob,
            confidence=confidence,
            raw_response=response.text,
        )
    
    def _answer_numeric(
        self,
        query: Query,
        context: Optional[str],
    ) -> QueryResult:
        """Answer a quantile or expectation query."""
        prompts = PROMPTS["quantile_query"]
        
        user_prompt = prompts["user"].format(query=query.text)
        
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"
        
        response = self.llm.query(
            prompt=user_prompt,
            system_prompt=prompts["system"],
            temperature=0.5,
        )
        
        try:
            value, confidence = self.parser.parse_quantile_response(response.text)
            return QueryResult(
                query_id=query.id,
                value=value,
                confidence=confidence,
                raw_response=response.text,
            )
        except ValueError:
            # Fall back to probability parsing if numeric fails
            prob, confidence = self.parser.parse_probability_response(response.text)
            return QueryResult(
                query_id=query.id,
                probability=prob,
                confidence=confidence * 0.5,  # Lower confidence for fallback
                raw_response=response.text,
            )
    
    def _answer_probability(
        self,
        query: Query,
        context: Optional[str],
    ) -> QueryResult:
        """Answer a general probability query."""
        prompts = PROMPTS["probability_query"]
        
        user_prompt = prompts["user"].format(query=query.text)
        
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"
        
        response = self.llm.query(
            prompt=user_prompt,
            system_prompt=prompts["system"],
            temperature=0.5,
        )
        
        prob, confidence = self.parser.parse_probability_response(response.text)
        
        return QueryResult(
            query_id=query.id,
            probability=prob,
            confidence=confidence,
            raw_response=response.text,
        )
