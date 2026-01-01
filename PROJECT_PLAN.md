# Calibrated Response: Project Plan

## Overview

A Python framework for building calibrated distribution predictions by combining multiple component predictions using maximum entropy models. The system generates relevant variables and conditional queries via LLM, collects distributional answers, and combines them into a coherent joint distribution.

## Core Concepts

### Problem Statement
Given a forecasting question (e.g., "How many people will take the train to work in San Francisco tomorrow?"), produce a well-calibrated probability distribution over possible answers.

### Approach
1. **Variable Generation**: Use an LLM to identify relevant variables (weather, day of week, population, system status, etc.)
2. **Query Generation**: Convert variables into specific distributional queries (marginals, conditionals, thresholds)
3. **Query Answering**: Obtain distribution predictions from LLM for each query
4. **Distribution Combination**: Use maximum entropy methods to combine constraints into a coherent joint distribution
5. **Evaluation**: Test calibration against resolved Metaculus questions

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline                                  │
├─────────────────────────────────────────────────────────────────┤
│  Question → VariableGenerator → QueryGenerator → QueryAnswerer  │
│                                       ↓                          │
│                              MaxEntropyModel                     │
│                                       ↓                          │
│                           Final Distribution                     │
└─────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. `models/` - Data Models
- `question.py` - Question representation (from Metaculus or custom)
- `variable.py` - Variable definitions (continuous, discrete, binary)
- `query.py` - Query types (marginal, conditional, threshold)
- `distribution.py` - Distribution representations and conversions

### 2. `generation/` - LLM-based Generation
- `variable_generator.py` - Generate relevant variables for a question
- `query_generator.py` - Generate specific queries from variables
- `query_answerer.py` - Get distributional answers from LLM
- `prompts.py` - Prompt templates for each generation task

### 3. `maxent/` - Maximum Entropy Model
- `constraints.py` - Constraint representations from query answers
- `solver.py` - MaxEnt optimization (handle conflicting constraints)
- `distribution_builder.py` - Build final distribution from solved model

### 4. `llm/` - LLM Client Abstraction
- `base.py` - Abstract LLM client interface
- `gemini.py` - Google Gemini API implementation
- `response_parser.py` - Parse LLM responses into structured distributions

### 5. `evaluation/` - Evaluation Pipeline
- `loader.py` - Load Metaculus dataset
- `metrics.py` - Calibration metrics (Brier score, log score, calibration curves)
- `runner.py` - Run evaluation experiments

### 6. `utils/` - Utilities
- `config.py` - Configuration management
- `logging.py` - Logging setup

## Key Design Decisions

### Query Types (Binary Focus)
To simplify LLM response parsing and MaxEnt constraint formulation, all queries will be converted to binary form:
- **Threshold queries**: "P(X > threshold)" 
- **Median queries**: "What is the median of X?" → converted to binary constraints
- **Conditional binary**: "P(X > threshold | condition)"

### Constraint Handling
Since constraints may conflict (LLM predictions aren't perfectly consistent):
1. Weight constraints by confidence/reliability estimates
2. Use soft constraints with slack variables
3. Minimize KL divergence from prior while approximately satisfying constraints

### Budget Management
- Fixed query budget per question (configurable, default ~10-20 queries)
- Greedy selection: LLM suggests which query would be most informative next
- Single-shot mode for initial implementation (select all queries upfront)

## Implementation Phases

### Phase 1: Core Infrastructure ✅ (Current)
- [x] Metaculus scraper (`eval_scraper.py`)
- [ ] Project structure setup
- [ ] Data models
- [ ] Configuration system

### Phase 2: LLM Integration
- [ ] Gemini client implementation
- [ ] Variable generation with prompts
- [ ] Query generation with prompts
- [ ] Response parsing (distributions from text)

### Phase 3: MaxEnt Model
- [ ] Constraint representation
- [ ] MaxEnt solver (using scipy or dedicated package)
- [ ] Handle conflicting constraints
- [ ] Distribution output

### Phase 4: Pipeline Integration
- [ ] End-to-end pipeline
- [ ] Query budget management
- [ ] Caching for API efficiency

### Phase 5: Evaluation
- [ ] Calibration metrics implementation
- [ ] Metaculus evaluation runner
- [ ] Results visualization

## Dependencies

```
# Core
numpy
scipy
pandas

# LLM
google-generativeai  # Gemini API

# MaxEnt (options to evaluate)
# - scipy.optimize for custom implementation
# - maxentropy package
# - cvxpy for convex optimization

# Evaluation
matplotlib
scikit-learn

# Utilities
requests  # for Metaculus API
pydantic  # for data validation
python-dotenv  # for API keys
```

## Configuration

```yaml
# config.yaml
llm:
  provider: gemini
  model: gemini-pro
  temperature: 0.7

pipeline:
  query_budget: 15
  variable_count: 5
  use_conditionals: true

maxent:
  constraint_tolerance: 0.1
  max_iterations: 1000
  regularization: 0.01

evaluation:
  dataset_path: questions.jsonl
  metrics: [brier_score, log_score, calibration_error]
```

## Open Questions

1. **MaxEnt Package Choice**: Should we use `maxentropy`, custom scipy implementation, or cvxpy?
2. **Distribution Discretization**: How fine-grained should continuous distributions be discretized?
3. **Confidence Weighting**: How to weight LLM confidence in constraint satisfaction?
4. **Variable Selection**: Greedy vs. batch selection of variables/queries?

## Next Steps

1. Set up project structure with all modules
2. Implement data models (Question, Variable, Query, Distribution)
3. Build Gemini client with basic prompts
4. Implement simple MaxEnt solver
5. Create minimal end-to-end pipeline
6. Run initial evaluation on small Metaculus subset
