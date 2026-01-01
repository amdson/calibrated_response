#!/usr/bin/env python3
"""
Example: Run evaluation on the Metaculus dataset.

Usage:
    python examples/run_evaluation.py [--n-questions 10] [--dataset questions.jsonl]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibrated_response.evaluation import EvaluationRunner, EvaluationConfig


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on Metaculus dataset")
    parser.add_argument(
        "--dataset", 
        default="questions.jsonl",
        help="Path to dataset file"
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=10,
        help="Number of questions to evaluate (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"Dataset not found: {args.dataset}")
        print("Run eval_scraper.py first to create the dataset.")
        sys.exit(1)
    
    # Configure evaluation
    config = EvaluationConfig(
        dataset_path=args.dataset,
        n_questions=args.n_questions,
        output_dir=args.output_dir,
        random_seed=args.seed,
        verbose=True,
    )
    
    # Run evaluation
    print(f"Running evaluation on {args.n_questions} questions...")
    print(f"Dataset: {args.dataset}")
    print()
    
    runner = EvaluationRunner(config=config)
    results = runner.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    metrics = results.get('metrics', {})
    print(f"Questions evaluated: {results.get('n_successful', 0)}/{results.get('n_questions', 0)}")
    
    if metrics.get('mean_crps') is not None:
        print(f"Mean CRPS: {metrics['mean_crps']:.4f}")
    if metrics.get('mean_log_score') is not None:
        print(f"Mean Log Score: {metrics['mean_log_score']:.4f}")
    if metrics.get('coverage_50') is not None:
        print(f"50% Interval Coverage: {metrics['coverage_50']:.1%}")
    if metrics.get('coverage_90') is not None:
        print(f"90% Interval Coverage: {metrics['coverage_90']:.1%}")


if __name__ == "__main__":
    main()
