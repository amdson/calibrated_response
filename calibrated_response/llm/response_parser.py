"""Parse LLM responses into structured distributions and values."""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np


def parse_probability(text: str) -> Tuple[Optional[float], float]:
    """Extract a probability value from text.
    
    Handles various formats:
    - "70%" or "70 percent"
    - "0.7" or ".7"
    - "7 in 10" or "7/10"
    - "likely" (mapped to ~0.7)
    - "very likely" (mapped to ~0.85)
    
    Args:
        text: Text containing a probability
        
    Returns:
        Tuple of (probability, confidence) where confidence indicates
        how certain we are about the parsing.
    """
    text = text.lower().strip()
    
    # Try percentage format: "70%" or "70 percent"
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', text)
    if pct_match:
        prob = float(pct_match.group(1)) / 100
        return (min(max(prob, 0), 1), 0.95)
    
    # Try decimal format: "0.7" or ".7" 
    decimal_match = re.search(r'\b(0?\.\d+)\b', text)
    if decimal_match:
        prob = float(decimal_match.group(1))
        return (min(max(prob, 0), 1), 0.9)
    
    # Try fraction format: "7 in 10" or "7/10" or "7 out of 10"
    fraction_match = re.search(r'(\d+)\s*(?:in|/|out of)\s*(\d+)', text)
    if fraction_match:
        num = float(fraction_match.group(1))
        denom = float(fraction_match.group(2))
        if denom > 0:
            prob = num / denom
            return (min(max(prob, 0), 1), 0.85)
    
    # Try odds format: "3 to 1" or "3:1"
    odds_match = re.search(r'(\d+)\s*(?:to|:)\s*(\d+)', text)
    if odds_match:
        a = float(odds_match.group(1))
        b = float(odds_match.group(2))
        if a + b > 0:
            # Interpret as "a to b against" -> prob = b/(a+b)
            # or "a to b for" -> prob = a/(a+b)
            # Default to first interpretation
            prob = b / (a + b)
            return (prob, 0.7)
    
    # Try word mappings
    word_probs = {
        'certain': 0.99,
        'almost certain': 0.95,
        'very likely': 0.85,
        'likely': 0.70,
        'probable': 0.70,
        'more likely than not': 0.60,
        'possible': 0.50,
        'even chance': 0.50,
        'toss-up': 0.50,
        'unlikely': 0.30,
        'improbable': 0.30,
        'very unlikely': 0.15,
        'almost impossible': 0.05,
        'impossible': 0.01,
    }
    
    for phrase, prob in word_probs.items():
        if phrase in text:
            return (prob, 0.5)  # Lower confidence for word mappings
    
    # Last resort: look for any number that could be a probability
    num_match = re.search(r'\b(\d+)\b', text)
    if num_match:
        num = float(num_match.group(1))
        if 0 <= num <= 1:
            return (num, 0.6)
        elif 1 < num <= 100:
            return (num / 100, 0.6)
    
    return (None, 0.0)


def parse_numeric_value(text: str, expected_magnitude: Optional[float] = None) -> Tuple[Optional[float], float]:
    """Extract a numeric value from text.
    
    Handles various formats:
    - "50,000" or "50000"
    - "50k" or "50K" 
    - "50 thousand"
    - "5.2 million"
    - Scientific notation "5.2e6"
    
    Args:
        text: Text containing a number
        expected_magnitude: Optional hint about expected order of magnitude
        
    Returns:
        Tuple of (value, confidence)
    """
    text = text.lower().strip()
    
    # Magnitude suffixes
    suffixes = {
        'k': 1e3,
        'thousand': 1e3,
        'thousands': 1e3,
        'm': 1e6,
        'million': 1e6,
        'millions': 1e6,
        'b': 1e9,
        'billion': 1e9,
        'billions': 1e9,
        't': 1e12,
        'trillion': 1e12,
        'trillions': 1e12,
    }
    
    # Try scientific notation first
    sci_match = re.search(r'([\d.]+)\s*[eE]\s*([+-]?\d+)', text)
    if sci_match:
        try:
            value = float(sci_match.group(1)) * (10 ** int(sci_match.group(2)))
            return (value, 0.95)
        except ValueError:
            pass
    
    # Try number with suffix: "50k", "5.2 million"
    suffix_match = re.search(
        r'([\d,]+(?:\.\d+)?)\s*(k|thousand|thousands|m|million|millions|b|billion|billions|t|trillion|trillions)',
        text
    )
    if suffix_match:
        try:
            num_str = suffix_match.group(1).replace(',', '')
            num = float(num_str)
            multiplier = suffixes.get(suffix_match.group(2), 1)
            return (num * multiplier, 0.9)
        except ValueError:
            pass
    
    # Try plain number with optional commas
    plain_match = re.search(r'([\d,]+(?:\.\d+)?)', text)
    if plain_match:
        try:
            num_str = plain_match.group(1).replace(',', '')
            value = float(num_str)
            
            # Confidence depends on whether it matches expected magnitude
            confidence = 0.8
            if expected_magnitude and abs(np.log10(value + 1) - np.log10(expected_magnitude + 1)) > 3:
                confidence = 0.5  # Suspicious if very different from expected
            
            return (value, confidence)
        except ValueError:
            pass
    
    return (None, 0.0)


class ResponseParser:
    """Parse LLM responses into structured data for queries."""
    
    @staticmethod
    def parse_probability_response(response_text: str) -> Tuple[float, float]:
        """Parse a probability from an LLM response.
        
        Args:
            response_text: Full LLM response text
            
        Returns:
            Tuple of (probability, confidence)
        """
        prob, conf = parse_probability(response_text)
        
        if prob is None:
            # Default to maximum entropy (0.5) with low confidence
            return (0.5, 0.1)
        
        return (prob, conf)
    
    @staticmethod
    def parse_quantile_response(
        response_text: str,
        expected_magnitude: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Parse a quantile value from an LLM response.
        
        Args:
            response_text: Full LLM response text
            expected_magnitude: Optional hint about expected scale
            
        Returns:
            Tuple of (value, confidence)
        """
        value, conf = parse_numeric_value(response_text, expected_magnitude)
        
        if value is None:
            raise ValueError(f"Could not parse numeric value from: {response_text[:200]}")
        
        return (value, conf)
    
    @staticmethod
    def parse_variable_list(response_text: str) -> list[dict[str, str]]:
        """Parse a list of variables from LLM response.
        
        Expects format like:
        1. Variable Name: Description
        2. Variable Name: Description
        
        Or JSON array.
        
        Returns:
            List of dicts with 'name' and 'description' keys
        """
        import json
        
        # Try JSON first
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        # Try numbered list format
        variables = []
        pattern = r'(?:\d+\.?\s*)?([^:]+):\s*(.+?)(?=\n\d+\.|\n*$)'
        matches = re.findall(pattern, response_text, re.MULTILINE)
        
        for name, desc in matches:
            name = name.strip().strip('*').strip()
            desc = desc.strip()
            if name and desc:
                variables.append({'name': name, 'description': desc})
        
        # Fallback: split by newlines
        if not variables:
            for line in response_text.strip().split('\n'):
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    name = parts[0].strip().lstrip('0123456789.-) ').strip('*')
                    desc = parts[1].strip()
                    if name and desc:
                        variables.append({'name': name, 'description': desc})
        
        return variables
