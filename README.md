# Calibrated Response

A Python framework for building calibrated distribution predictions by combining multiple LLM-generated predictions using maximum entropy models.

## Overview

Given a forecasting question, this system:
1. Uses an LLM to identify relevant variables
2. Generates distributional queries about those variables
3. Collects predictions from the LLM
4. Combines them using maximum entropy methods into a coherent distribution
5. Evaluates calibration against resolved Metaculus questions

## Installation

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

## Usage

```python
from calibrated_response import Pipeline
from calibrated_response.llm import GeminiClient

# Initialize
client = GeminiClient()
pipeline = Pipeline(llm_client=client)

# Make a prediction
question = "How many people will take the train to work in San Francisco tomorrow?"
distribution = pipeline.predict(question)

print(distribution.summary())
```

## Project Structure

```
calibrated_response/
├── models/          # Data models (Question, Variable, Query, Distribution)
├── generation/      # LLM-based variable and query generation
├── maxent/          # Maximum entropy model for combining predictions
├── llm/             # LLM client implementations
├── evaluation/      # Evaluation pipeline and metrics
└── utils/           # Configuration and utilities
```

## Development

```bash
# Run tests
pytest

# Run evaluation
python -m calibrated_response.evaluation.runner
```

## License

MIT
