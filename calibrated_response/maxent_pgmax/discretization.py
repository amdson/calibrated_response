"""Discretization helpers for pgmax-backed inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from calibrated_response.models.query import EqualityProposition, InequalityProposition, PropositionUnion
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable, Variable


@dataclass(frozen=True)
class VariableBuckets:
    """Bucket metadata for one variable."""

    name: str
    bin_edges: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.bin_edges.size - 1)

    @property
    def bin_centers(self) -> np.ndarray:
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0


class DomainDiscretizer:
    """Convert typed variables to finite-state buckets."""

    def __init__(self, variables: Sequence[Variable], max_bins: int = 21):
        self.variables = list(variables)
        self.max_bins = max(2, int(max_bins))
        self._by_name = {v.name: self._build_buckets(v) for v in self.variables}

    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def buckets(self, var_name: str) -> VariableBuckets:
        if var_name not in self._by_name:
            raise ValueError(f"Unknown variable: {var_name}")
        return self._by_name[var_name]

    def all_buckets(self) -> list[VariableBuckets]:
        return [self._by_name[v.name] for v in self.variables]

    def proposition_mask(self, prop: PropositionUnion) -> np.ndarray:
        buckets = self.buckets(prop.variable)
        centers = buckets.bin_centers

        if isinstance(prop, InequalityProposition):
            if prop.is_lower_bound:
                return centers > float(prop.threshold)
            return centers < float(prop.threshold)

        if isinstance(prop, EqualityProposition):
            # Binary propositions are represented by two buckets: [0, 0.5), [0.5, 1.0]
            if isinstance(prop.value, bool):
                return centers >= 0.5 if prop.value else centers < 0.5
            return np.zeros_like(centers, dtype=bool)

        return np.zeros_like(centers, dtype=bool)

    def _build_buckets(self, variable: Variable) -> VariableBuckets:
        if isinstance(variable, BinaryVariable):
            edges = np.array([0.0, 0.5, 1.0], dtype=float)
            return VariableBuckets(name=variable.name, bin_edges=edges)

        if isinstance(variable, ContinuousVariable):
            lower, upper = variable.get_domain()
            if not np.isfinite(lower):
                lower = 0.0
            if not np.isfinite(upper):
                upper = lower + 1.0
            if upper <= lower:
                upper = lower + 1.0
            edges = np.linspace(float(lower), float(upper), self.max_bins + 1)
            return VariableBuckets(name=variable.name, bin_edges=edges)

        # Fallback for unsupported variable subclasses.
        edges = np.linspace(0.0, 1.0, self.max_bins + 1)
        return VariableBuckets(name=variable.name, bin_edges=edges)
