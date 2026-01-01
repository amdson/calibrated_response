"""Configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "gemini"
    model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1024


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    n_variables: int = 5
    n_queries: int = 10
    include_conditionals: bool = True
    query_budget: int = 15


class MaxEntConfig(BaseModel):
    """MaxEnt solver configuration."""
    n_bins: int = 50
    max_iterations: int = 1000
    tolerance: float = 1e-6
    regularization: float = 0.01
    use_soft_constraints: bool = True
    constraint_weight: float = 100.0


class EvalConfig(BaseModel):
    """Evaluation configuration."""
    dataset_path: str = "questions.jsonl"
    output_dir: str = "eval_results"
    n_questions: Optional[int] = None
    random_seed: int = 42


class Config(BaseModel):
    """Main configuration container."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    maxent: MaxEntConfig = Field(default_factory=MaxEntConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            Config object
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.
        
        Environment variables are prefixed with CALIBRATED_RESPONSE_.
        For example: CALIBRATED_RESPONSE_LLM_MODEL=gemini-pro
        
        Returns:
            Config object
        """
        config = cls()
        
        # LLM settings
        if model := os.getenv("CALIBRATED_RESPONSE_LLM_MODEL"):
            config.llm.model = model
        if temp := os.getenv("CALIBRATED_RESPONSE_LLM_TEMPERATURE"):
            config.llm.temperature = float(temp)
        
        # Pipeline settings
        if n_vars := os.getenv("CALIBRATED_RESPONSE_N_VARIABLES"):
            config.pipeline.n_variables = int(n_vars)
        if n_queries := os.getenv("CALIBRATED_RESPONSE_N_QUERIES"):
            config.pipeline.n_queries = int(n_queries)
        
        # Eval settings
        if dataset := os.getenv("CALIBRATED_RESPONSE_DATASET"):
            config.evaluation.dataset_path = dataset
        
        return config
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save to
        """
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load configuration from file or environment.
    
    Args:
        path: Optional path to YAML config file. If not provided,
              looks for config.yaml in current directory, then
              falls back to environment variables.
              
    Returns:
        Config object
    """
    # Try explicit path
    if path:
        return Config.from_yaml(path)
    
    # Try default locations
    default_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path.home() / ".calibrated_response" / "config.yaml",
    ]
    
    for p in default_paths:
        if p.exists():
            return Config.from_yaml(p)
    
    # Fall back to environment
    return Config.from_env()
