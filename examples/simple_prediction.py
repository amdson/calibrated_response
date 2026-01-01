#!/usr/bin/env python3
"""
Example: Make a single prediction using the calibrated response pipeline.

Usage:
    python examples/simple_prediction.py "How many people will take the train to work in SF tomorrow?"
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibrated_response.llm import GeminiClient
from calibrated_response.pipeline import Pipeline


def main():
    # Get question from command line or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How many people will take the train to work in San Francisco tomorrow?"
    
    print(f"Question: {question}\n")
    
    # Initialize pipeline
    client = GeminiClient()
    pipeline = Pipeline(
        llm_client=client,
        n_variables=5,
        n_queries=8,
    )
    
    # Make prediction
    print("Generating variables and queries...")
    distribution, info = pipeline.predict(
        question=question,
        domain_min=0,
        domain_max=200000,  # Reasonable range for SF train ridership
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nVariables identified ({info['n_variables']}):")
    for v in info.get('variables', [])[:5]:
        print(f"  - {v['name']}: {v['description'][:60]}...")
    
    print(f"\nQueries asked ({info['n_queries']}):")
    for q in info.get('queries', [])[:5]:
        print(f"  - [{q['type']}] {q['text'][:60]}...")
    
    print("\n" + "-" * 60)
    print("PREDICTION SUMMARY")
    print("-" * 60)
    print(f"  Mean:   {distribution.mean():,.0f}")
    print(f"  Median: {distribution.median():,.0f}")
    print(f"  Std:    {distribution.std():,.0f}")
    print(f"  10th percentile: {distribution.quantile(0.1):,.0f}")
    print(f"  90th percentile: {distribution.quantile(0.9):,.0f}")
    
    # Solver diagnostics
    solver_info = info.get('solver_info', {})
    if solver_info:
        print(f"\n  Solver converged: {solver_info.get('success', 'N/A')}")
        print(f"  Total constraint violation: {solver_info.get('total_violation', 0):.4f}")
        print(f"  Distribution entropy: {solver_info.get('entropy', 0):.4f}")


if __name__ == "__main__":
    main()
