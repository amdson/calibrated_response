"""Calibration metrics for evaluating predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibrated_response.models.distribution import Distribution, HistogramDistribution


def brier_score(predicted_prob: float, actual_outcome: bool | int) -> float:
    """Compute Brier score for a single binary prediction.
    
    Args:
        predicted_prob: Predicted probability of outcome being True/1
        actual_outcome: Actual outcome (True/1 or False/0)
        
    Returns:
        Brier score (lower is better, 0 = perfect)
    """
    actual = 1.0 if actual_outcome else 0.0
    return (predicted_prob - actual) ** 2


def log_score(predicted_prob: float, actual_outcome: bool | int, eps: float = 1e-10) -> float:
    """Compute log score (negative log probability) for a binary prediction.
    
    Args:
        predicted_prob: Predicted probability of outcome being True/1
        actual_outcome: Actual outcome (True/1 or False/0)
        eps: Small value to avoid log(0)
        
    Returns:
        Negative log probability (lower is better)
    """
    if actual_outcome:
        return -np.log(max(predicted_prob, eps))
    else:
        return -np.log(max(1 - predicted_prob, eps))


def continuous_log_score(
    distribution: HistogramDistribution,
    actual_value: float,
    eps: float = 1e-10,
) -> float:
    """Compute log score for a continuous prediction.
    
    Uses the probability density at the actual value.
    
    Args:
        distribution: Predicted distribution
        actual_value: Actual observed value
        eps: Small value to avoid log(0)
        
    Returns:
        Negative log density (lower is better)
    """
    # Find the bin containing the actual value
    edges = np.array(distribution.bin_edges)
    probs = np.array(distribution.bin_probabilities)
    
    for i, (left, right, p) in enumerate(zip(edges[:-1], edges[1:], probs)):
        if left <= actual_value < right:
            width = right - left
            density = p / width if width > 0 else eps
            return -np.log(max(density, eps))
    
    # Value outside support
    return -np.log(eps)


def crps(distribution: HistogramDistribution, actual_value: float) -> float:
    """Compute Continuous Ranked Probability Score.
    
    CRPS is a proper scoring rule for continuous distributions that
    generalizes Brier score.
    
    Args:
        distribution: Predicted distribution
        actual_value: Actual observed value
        
    Returns:
        CRPS (lower is better)
    """
    # CRPS = integral of (CDF(x) - 1(x >= actual))^2 dx
    edges = np.array(distribution.bin_edges)
    probs = np.array(distribution.bin_probabilities)
    
    score = 0.0
    cumsum = 0.0
    
    for left, right, p in zip(edges[:-1], edges[1:], probs):
        width = right - left
        
        # Indicator function: 1 if actual >= x
        if actual_value >= right:
            # Entire bin is below actual
            indicator = 1.0
            score += width * ((cumsum + p/2) - indicator) ** 2
        elif actual_value <= left:
            # Entire bin is above actual
            indicator = 0.0
            score += width * ((cumsum + p/2) - indicator) ** 2
        else:
            # Bin contains actual value
            frac_below = (actual_value - left) / width
            # Below actual
            score += (actual_value - left) * ((cumsum + p * frac_below / 2) - 1) ** 2
            # Above actual
            score += (right - actual_value) * ((cumsum + p * (1 + frac_below) / 2) - 0) ** 2
        
        cumsum += p
    
    return score


def calibration_error(
    predictions: list[float],
    outcomes: list[bool | int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error for binary predictions.
    
    ECE measures how well predicted probabilities match observed frequencies.
    
    Args:
        predictions: List of predicted probabilities
        outcomes: List of actual outcomes (True/1 or False/0)
        n_bins: Number of calibration bins
        
    Returns:
        Expected Calibration Error (lower is better, 0 = perfect calibration)
    """
    predictions = np.array(predictions)
    outcomes = np.array([1 if o else 0 for o in outcomes])
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = mask | (predictions == bin_edges[i + 1])
        
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_pred = predictions[mask].mean()
            avg_outcome = outcomes[mask].mean()
            ece += (n_in_bin / len(predictions)) * abs(avg_pred - avg_outcome)
    
    return ece


def interval_coverage(
    distributions: list[HistogramDistribution],
    actual_values: list[float],
    level: float = 0.9,
) -> float:
    """Compute empirical coverage of prediction intervals.
    
    A well-calibrated predictor should have ~90% of actual values
    fall within the 90% prediction interval.
    
    Args:
        distributions: List of predicted distributions
        actual_values: List of actual observed values
        level: Nominal coverage level (e.g., 0.9 for 90%)
        
    Returns:
        Empirical coverage rate
    """
    alpha = (1 - level) / 2
    covered = 0
    
    for dist, actual in zip(distributions, actual_values):
        lower = dist.quantile(alpha)
        upper = dist.quantile(1 - alpha)
        if lower <= actual <= upper:
            covered += 1
    
    return covered / len(distributions) if distributions else 0.0


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    
    # Scoring rules
    mean_brier_score: Optional[float] = None
    mean_log_score: Optional[float] = None
    mean_crps: Optional[float] = None
    
    # Calibration
    ece: Optional[float] = None  # Expected Calibration Error
    coverage_50: Optional[float] = None  # 50% interval coverage
    coverage_90: Optional[float] = None  # 90% interval coverage
    
    # Summary statistics
    n_predictions: int = 0
    
    @classmethod
    def compute_binary(
        cls,
        predictions: list[float],
        outcomes: list[bool | int],
    ) -> CalibrationMetrics:
        """Compute metrics for binary predictions.
        
        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes
            
        Returns:
            CalibrationMetrics object
        """
        if not predictions:
            return cls()
        
        brier_scores = [brier_score(p, o) for p, o in zip(predictions, outcomes)]
        log_scores = [log_score(p, o) for p, o in zip(predictions, outcomes)]
        
        return cls(
            mean_brier_score=np.mean(brier_scores),
            mean_log_score=np.mean(log_scores),
            ece=calibration_error(predictions, outcomes),
            n_predictions=len(predictions),
        )
    
    @classmethod
    def compute_continuous(
        cls,
        distributions: list[HistogramDistribution],
        actual_values: list[float],
    ) -> CalibrationMetrics:
        """Compute metrics for continuous predictions.
        
        Args:
            distributions: Predicted distributions
            actual_values: Actual observed values
            
        Returns:
            CalibrationMetrics object
        """
        if not distributions:
            return cls()
        
        log_scores = [
            continuous_log_score(d, v) 
            for d, v in zip(distributions, actual_values)
        ]
        crps_scores = [crps(d, v) for d, v in zip(distributions, actual_values)]
        
        return cls(
            mean_log_score=np.mean(log_scores),
            mean_crps=np.mean(crps_scores),
            coverage_50=interval_coverage(distributions, actual_values, 0.5),
            coverage_90=interval_coverage(distributions, actual_values, 0.9),
            n_predictions=len(distributions),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'mean_brier_score': self.mean_brier_score,
            'mean_log_score': self.mean_log_score,
            'mean_crps': self.mean_crps,
            'ece': self.ece,
            'coverage_50': self.coverage_50,
            'coverage_90': self.coverage_90,
            'n_predictions': self.n_predictions,
        }
