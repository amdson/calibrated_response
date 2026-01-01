"""Run evaluation experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from calibrated_response.models.question import MetaculusQuestion, QuestionType
from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.evaluation.loader import DatasetLoader
from calibrated_response.evaluation.metrics import CalibrationMetrics


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Dataset
    dataset_path: str = "questions.jsonl"
    n_questions: Optional[int] = None  # None = all questions
    random_seed: int = 42
    
    # Pipeline settings
    n_variables: int = 5
    n_queries: int = 10
    include_conditionals: bool = True
    
    # Domain settings (for numeric questions)
    default_n_bins: int = 50
    
    # Output
    output_dir: str = "eval_results"
    save_predictions: bool = True
    verbose: bool = True


@dataclass
class PredictionResult:
    """Result of predicting a single question."""
    
    question_id: str
    question_title: str
    question_type: str
    
    # Prediction
    predicted_distribution: Optional[dict] = None  # Serialized distribution
    predicted_median: Optional[float] = None
    predicted_mean: Optional[float] = None
    
    # Ground truth
    resolution: Optional[Any] = None
    
    # Diagnostics
    n_variables: int = 0
    n_queries: int = 0
    solver_info: dict = field(default_factory=dict)
    
    # Timing
    generation_time_s: float = 0.0
    solving_time_s: float = 0.0
    total_time_s: float = 0.0
    
    # Errors
    error: Optional[str] = None


class EvaluationRunner:
    """Run evaluation experiments on Metaculus dataset."""
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """Initialize the evaluation runner.
        
        Args:
            config: Evaluation configuration
            llm_client: LLM client for generation (if None, uses GeminiClient)
        """
        self.config = config or EvaluationConfig()
        self._llm_client = llm_client
        self._pipeline = None
    
    @property
    def llm_client(self):
        """Lazily initialize LLM client."""
        if self._llm_client is None:
            from calibrated_response.llm import GeminiClient
            self._llm_client = GeminiClient()
        return self._llm_client
    
    @property
    def pipeline(self):
        """Lazily initialize prediction pipeline."""
        if self._pipeline is None:
            from calibrated_response.pipeline import Pipeline
            self._pipeline = Pipeline(
                llm_client=self.llm_client,
                n_variables=self.config.n_variables,
                n_queries=self.config.n_queries,
            )
        return self._pipeline
    
    def run(self) -> dict[str, Any]:
        """Run the full evaluation.
        
        Returns:
            Dictionary with evaluation results and metrics
        """
        # Load dataset
        loader = DatasetLoader(self.config.dataset_path)
        
        if self.config.n_questions:
            questions = loader.load_sample(
                n=self.config.n_questions,
                resolved_only=True,
                seed=self.config.random_seed,
            )
        else:
            questions = [q for q in loader.load_all() if q.is_resolved()]
        
        if self.config.verbose:
            print(f"Loaded {len(questions)} questions for evaluation")
        
        # Run predictions
        results = []
        distributions = []
        resolutions = []
        
        for i, question in enumerate(questions):
            if self.config.verbose:
                print(f"Processing {i+1}/{len(questions)}: {question.title[:50]}...")
            
            result = self._predict_question(question)
            results.append(result)
            
            if result.predicted_distribution and result.resolution is not None:
                dist = HistogramDistribution(**result.predicted_distribution)
                distributions.append(dist)
                resolutions.append(float(result.resolution))
        
        # Compute metrics
        metrics = CalibrationMetrics.compute_continuous(distributions, resolutions)
        
        # Save results
        output = {
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'n_questions': len(questions),
            'n_successful': len(distributions),
            'metrics': metrics.to_dict(),
            'results': [self._serialize_result(r) for r in results],
        }
        
        if self.config.save_predictions:
            self._save_results(output)
        
        if self.config.verbose:
            print(f"\nEvaluation complete!")
            print(f"Successful predictions: {len(distributions)}/{len(questions)}")
            print(f"Mean CRPS: {metrics.mean_crps:.4f}" if metrics.mean_crps else "")
            print(f"90% Coverage: {metrics.coverage_90:.2%}" if metrics.coverage_90 else "")
        
        return output
    
    def _predict_question(self, question: MetaculusQuestion) -> PredictionResult:
        """Make a prediction for a single question."""
        result = PredictionResult(
            question_id=question.id,
            question_title=question.title,
            question_type=question.question_type.value,
            resolution=question.resolution,
        )
        
        try:
            start_time = time.time()
            
            # Determine domain bounds
            domain = question.get_domain()
            if domain:
                domain_min, domain_max = domain
            else:
                # Default domain
                domain_min, domain_max = 0, 100
            
            # Run prediction pipeline
            distribution, info = self.pipeline.predict(
                question=question.title,
                domain_min=domain_min,
                domain_max=domain_max,
            )
            
            result.total_time_s = time.time() - start_time
            result.predicted_distribution = {
                'bin_edges': distribution.bin_edges,
                'bin_probabilities': distribution.bin_probabilities,
            }
            result.predicted_median = distribution.median()
            result.predicted_mean = distribution.mean()
            result.n_variables = info.get('n_variables', 0)
            result.n_queries = info.get('n_queries', 0)
            result.solver_info = info.get('solver_info', {})
            
        except Exception as e:
            result.error = str(e)
            result.total_time_s = time.time() - start_time if 'start_time' in dir() else 0
        
        return result
    
    def _serialize_result(self, result: PredictionResult) -> dict:
        """Serialize a prediction result to JSON-compatible dict."""
        return {
            'question_id': result.question_id,
            'question_title': result.question_title,
            'question_type': result.question_type,
            'predicted_median': result.predicted_median,
            'predicted_mean': result.predicted_mean,
            'resolution': result.resolution,
            'n_variables': result.n_variables,
            'n_queries': result.n_queries,
            'total_time_s': result.total_time_s,
            'error': result.error,
        }
    
    def _save_results(self, output: dict) -> None:
        """Save evaluation results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"eval_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)
        
        if self.config.verbose:
            print(f"Results saved to {output_path}")
    
    def run_single(self, question: str | MetaculusQuestion) -> PredictionResult:
        """Run prediction for a single question.
        
        Args:
            question: Question text or MetaculusQuestion object
            
        Returns:
            PredictionResult
        """
        if isinstance(question, str):
            from calibrated_response.models.question import Question, QuestionType
            q = MetaculusQuestion(
                id="custom",
                metaculus_id=0,
                title=question,
                question_type=QuestionType.NUMERIC,
            )
        else:
            q = question
        
        return self._predict_question(q)
