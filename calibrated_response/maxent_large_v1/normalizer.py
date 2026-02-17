from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from calibrated_response.models.variable import BinaryVariable, ContinuousVariable, Variable


@dataclass(frozen=True)
class NormalizedVariableInfo:
    name: str
    original_low: float
    original_high: float
    is_continuous: bool


class ContinuousDomainNormalizer:
    """Normalize continuous-variable domains to [0, 1) equivalents.

    Binary variables are preserved as-is. Continuous variables are transformed to
    identical `ContinuousVariable` objects with the same metadata and domain [0, 1].
    The normalizer stores original domains so values and bin edges can be mapped
    back and forth.
    """

    def __init__(self, variables: Sequence[Variable]):
        self.variables = list(variables)
        self._name_to_idx = {v.name: i for i, v in enumerate(self.variables)}
        self.info: list[NormalizedVariableInfo] = []

        for variable in self.variables:
            if isinstance(variable, ContinuousVariable):
                low, high = variable.get_domain()
                if not np.isfinite(low):
                    low = 0.0
                if not np.isfinite(high) or high <= low:
                    high = low + 1.0
                self.info.append(
                    NormalizedVariableInfo(
                        name=variable.name,
                        original_low=float(low),
                        original_high=float(high),
                        is_continuous=True,
                    )
                )
            else:
                self.info.append(
                    NormalizedVariableInfo(
                        name=variable.name,
                        original_low=0.0,
                        original_high=1.0,
                        is_continuous=False,
                    )
                )

    def normalized_variables(self) -> list[Variable]:
        """Return variables where continuous domains are mapped to [0, 1]."""
        out: list[Variable] = []
        for variable in self.variables:
            if isinstance(variable, ContinuousVariable):
                out.append(
                    ContinuousVariable(
                        name=variable.name,
                        description=variable.description,
                        lower_bound=0.0,
                        upper_bound=1.0,
                        unit=variable.unit,
                    )
                )
            elif isinstance(variable, BinaryVariable):
                out.append(
                    BinaryVariable(
                        name=variable.name,
                        description=variable.description,
                    )
                )
            else:
                out.append(variable)
        return out

    def normalized_bin_edges(self, var_name: str, n_bins: int) -> np.ndarray:
        """Return [0, 1) bin edges (plus terminal 1.0) for a variable."""
        n_bins = max(2, int(n_bins))
        idx = self._index(var_name)
        if self.info[idx].is_continuous:
            return np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
        return np.array([0.0, 0.5, 1.0], dtype=float)

    def normalize_value(self, var_name: str, value: float) -> float:
        idx = self._index(var_name)
        meta = self.info[idx]
        if not meta.is_continuous:
            return float(value)
        width = meta.original_high - meta.original_low
        if width <= 0:
            return 0.5
        z = (float(value) - meta.original_low) / width
        # Keep values inside [0, 1) to match normalized-bin semantics.
        return float(np.clip(z, 0.0, np.nextafter(1.0, 0.0)))

    def denormalize_value(self, var_name: str, normalized_value: float) -> float:
        idx = self._index(var_name)
        meta = self.info[idx]
        if not meta.is_continuous:
            return float(normalized_value)
        z = float(np.clip(normalized_value, 0.0, np.nextafter(1.0, 0.0)))
        return meta.original_low + z * (meta.original_high - meta.original_low)

    def normalize_edges(self, var_name: str, edges: np.ndarray) -> np.ndarray:
        idx = self._index(var_name)
        meta = self.info[idx]
        if not meta.is_continuous:
            return np.asarray(edges, dtype=float)
        width = meta.original_high - meta.original_low
        if width <= 0:
            return np.zeros_like(np.asarray(edges, dtype=float))
        normalized = (np.asarray(edges, dtype=float) - meta.original_low) / width
        return np.clip(normalized, 0.0, 1.0)

    def denormalize_edges(self, var_name: str, normalized_edges: np.ndarray) -> np.ndarray:
        idx = self._index(var_name)
        meta = self.info[idx]
        if not meta.is_continuous:
            return np.asarray(normalized_edges, dtype=float)
        z = np.asarray(normalized_edges, dtype=float)
        return meta.original_low + z * (meta.original_high - meta.original_low)

    def _index(self, var_name: str) -> int:
        if var_name not in self._name_to_idx:
            raise ValueError(f"Unknown variable: {var_name}")
        return self._name_to_idx[var_name]