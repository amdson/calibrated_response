"""Distribution representations and utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class Distribution(BaseModel, ABC):
    """Abstract base class for probability distributions."""
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def mean(self) -> float:
        """Compute the mean of the distribution."""
        pass
    
    @abstractmethod
    def variance(self) -> float:
        """Compute the variance of the distribution."""
        pass
    
    @abstractmethod
    def quantile(self, q: float) -> float:
        """Compute the q-th quantile."""
        pass
    
    @abstractmethod
    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x)."""
        pass
    
    @abstractmethod
    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n samples from the distribution."""
        pass
    
    def median(self) -> float:
        """Compute the median."""
        return self.quantile(0.5)
    
    def std(self) -> float:
        """Compute the standard deviation."""
        return np.sqrt(self.variance())
    
    def summary(self) -> dict[str, float]:
        """Get a summary of the distribution."""
        return {
            "mean": self.mean(),
            "std": self.std(),
            "median": self.median(),
            "q10": self.quantile(0.1),
            "q90": self.quantile(0.9),
        }


class BinaryDistribution(Distribution):
    """Distribution over a binary outcome."""
    
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the positive outcome"
    )
    
    def mean(self) -> float:
        return self.probability
    
    def variance(self) -> float:
        return self.probability * (1 - self.probability)
    
    def quantile(self, q: float) -> float:
        if q < 1 - self.probability:
            return 0.0
        return 1.0
    
    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        if x < 1:
            return 1 - self.probability
        return 1.0
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.binomial(1, self.probability, size=n).astype(float)
    
    def log_odds(self) -> float:
        """Get log odds."""
        if self.probability <= 0:
            return float('-inf')
        if self.probability >= 1:
            return float('inf')
        return np.log(self.probability / (1 - self.probability))


class DiscreteDistribution(Distribution):
    """Distribution over a finite set of values."""
    
    values: list[float] = Field(..., description="Possible values")
    probabilities: list[float] = Field(..., description="Probability of each value")
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that probabilities sum to 1."""
        if len(self.values) != len(self.probabilities):
            raise ValueError("values and probabilities must have same length")
        if abs(sum(self.probabilities) - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1, got {sum(self.probabilities)}")
    
    def mean(self) -> float:
        return sum(v * p for v, p in zip(self.values, self.probabilities))
    
    def variance(self) -> float:
        mu = self.mean()
        return sum(p * (v - mu) ** 2 for v, p in zip(self.values, self.probabilities))
    
    def quantile(self, q: float) -> float:
        sorted_pairs = sorted(zip(self.values, self.probabilities))
        cumsum = 0.0
        for v, p in sorted_pairs:
            cumsum += p
            if cumsum >= q:
                return v
        return sorted_pairs[-1][0]
    
    def cdf(self, x: float) -> float:
        return sum(p for v, p in zip(self.values, self.probabilities) if v <= x)
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.choice(self.values, size=n, p=self.probabilities)
    
    def entropy(self) -> float:
        """Compute Shannon entropy."""
        return -sum(p * np.log(p) for p in self.probabilities if p > 0)


class HistogramDistribution(Distribution):
    """Distribution represented as a histogram over bins.
    
    This is the primary representation for continuous distributions
    in the maximum entropy model.
    """
    
    bin_edges: list[float] = Field(..., description="Bin edges (n+1 values for n bins)")
    bin_probabilities: list[float] = Field(..., description="Probability mass in each bin")
    
    def model_post_init(self, __context: Any) -> None:
        """Validate histogram structure."""
        if len(self.bin_edges) != len(self.bin_probabilities) + 1:
            raise ValueError("bin_edges should have one more element than bin_probabilities")
        if abs(sum(self.bin_probabilities) - 1.0) > 1e-3:
            raise ValueError(f"Probabilities must sum to 1, got {sum(self.bin_probabilities)}")
    
    @property
    def n_bins(self) -> int:
        return len(self.bin_probabilities)
    
    @property
    def bin_centers(self) -> np.ndarray:
        """Get the center of each bin."""
        edges = np.array(self.bin_edges)
        return (edges[:-1] + edges[1:]) / 2
    
    @property
    def bin_widths(self) -> np.ndarray:
        """Get the width of each bin."""
        edges = np.array(self.bin_edges)
        return edges[1:] - edges[:-1]
    
    def mean(self) -> float:
        return sum(c * p for c, p in zip(self.bin_centers, self.bin_probabilities))
    
    def variance(self) -> float:
        mu = self.mean()
        # Variance includes both within-bin and between-bin variance
        var = 0.0
        for c, w, p in zip(self.bin_centers, self.bin_widths, self.bin_probabilities):
            # Between-bin variance
            var += p * (c - mu) ** 2
            # Within-bin variance (assuming uniform within bin)
            var += p * (w ** 2) / 12
        return var
    
    def quantile(self, q: float) -> float:
        cumsum = 0.0
        for i, (p, left, right) in enumerate(zip(
            self.bin_probabilities,
            self.bin_edges[:-1],
            self.bin_edges[1:]
        )):
            if cumsum + p >= q:
                # Linear interpolation within bin
                frac = (q - cumsum) / p if p > 0 else 0
                return left + frac * (right - left)
            cumsum += p
        return self.bin_edges[-1]
    
    def cdf(self, x: float) -> float:
        cumsum = 0.0
        for p, left, right in zip(
            self.bin_probabilities,
            self.bin_edges[:-1],
            self.bin_edges[1:]
        ):
            if x <= left:
                return cumsum
            if x >= right:
                cumsum += p
            else:
                # Linear interpolation
                frac = (x - left) / (right - left)
                cumsum += frac * p
                return cumsum
        return 1.0
    
    def sample(self, n: int = 1) -> np.ndarray:
        # First sample which bin, then uniform within that bin
        bin_indices = np.random.choice(
            self.n_bins, 
            size=n, 
            p=self.bin_probabilities
        )
        samples = np.empty(n)
        for i, bi in enumerate(bin_indices):
            left = self.bin_edges[bi]
            right = self.bin_edges[bi + 1]
            samples[i] = np.random.uniform(left, right)
        return samples
    
    def entropy(self) -> float:
        """Compute differential entropy approximation."""
        h = 0.0
        for p, w in zip(self.bin_probabilities, self.bin_widths):
            if p > 0 and w > 0:
                # Density is p/w within the bin
                density = p / w
                h -= p * np.log(density)
        return h
    
    def prob_greater_than(self, threshold: float) -> float:
        """Compute P(X > threshold)."""
        return 1.0 - self.cdf(threshold)
    
    def prob_less_than(self, threshold: float) -> float:
        """Compute P(X < threshold)."""
        return self.cdf(threshold)
    
    @classmethod
    def uniform(cls, lower: float, upper: float, n_bins: int = 50) -> HistogramDistribution:
        """Create a uniform distribution over [lower, upper]."""
        edges = np.linspace(lower, upper, n_bins + 1).tolist()
        probs = [1.0 / n_bins] * n_bins
        return cls(bin_edges=edges, bin_probabilities=probs)
    
    @classmethod
    def from_samples(
        cls, 
        samples: np.ndarray, 
        n_bins: int = 50,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> HistogramDistribution:
        """Create a histogram distribution from samples."""
        if lower is None:
            lower = float(np.min(samples))
        if upper is None:
            upper = float(np.max(samples))
        
        counts, edges = np.histogram(samples, bins=n_bins, range=(lower, upper))
        probs = counts / counts.sum()
        
        return cls(bin_edges=edges.tolist(), bin_probabilities=probs.tolist())
