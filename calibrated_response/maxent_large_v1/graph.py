from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np

from calibrated_response.models.variable import BinaryVariable, ContinuousVariable, Variable


@dataclass(frozen=True)
class SMMVariable:
    name: str
    kind: Literal["categorical", "bounded_continuous"]
    num_states: int
    low: float = 0.0
    high: float = 1.0


@dataclass(frozen=True)
class Factor:
    id: int
    var_ids: tuple[int, ...]
    theta: jnp.ndarray
    table_shape: tuple[int, ...]


@dataclass(frozen=True)
class FactorGraph:
    variables: list[SMMVariable]
    factors: list[Factor]
    var_to_factors: list[list[int]]
    factor_to_vars: list[tuple[int, ...]]

    def with_updated_thetas(self, theta_by_factor: dict[int, jnp.ndarray]) -> "FactorGraph":
        new_factors = []
        for factor in self.factors:
            theta = theta_by_factor.get(factor.id, factor.theta)
            new_factors.append(replace(factor, theta=theta))
        return replace(self, factors=new_factors)

    def n_variables(self) -> int:
        return len(self.variables)

    def cardinality(self, var_id: int) -> int:
        return self.variables[var_id].num_states


def from_variables(variables: Sequence[Variable], max_bins: int = 21) -> tuple[list[SMMVariable], list[np.ndarray]]:
    smm_vars: list[SMMVariable] = []
    bin_edges: list[np.ndarray] = []

    bins = max(2, int(max_bins))
    for variable in variables:
        if isinstance(variable, BinaryVariable):
            smm_vars.append(SMMVariable(name=variable.name, kind="categorical", num_states=2, low=0.0, high=1.0))
            bin_edges.append(np.array([0.0, 0.5, 1.0], dtype=float))
            continue

        if isinstance(variable, ContinuousVariable):
            low, high = variable.get_domain()
            if not np.isfinite(low):
                low = 0.0
            if not np.isfinite(high) or high <= low:
                high = low + 1.0
            smm_vars.append(
                SMMVariable(
                    name=variable.name,
                    kind="bounded_continuous",
                    num_states=bins,
                    low=float(low),
                    high=float(high),
                )
            )
            bin_edges.append(np.linspace(low, high, bins + 1))
            continue

        smm_vars.append(SMMVariable(name=variable.name, kind="bounded_continuous", num_states=bins, low=0.0, high=1.0))
        bin_edges.append(np.linspace(0.0, 1.0, bins + 1))

    return smm_vars, bin_edges


def build_graph(variables: Sequence[SMMVariable], factors: Sequence[Factor]) -> FactorGraph:
    var_to_factors: list[list[int]] = [[] for _ in range(len(variables))]
    factor_to_vars: list[tuple[int, ...]] = []
    for idx, factor in enumerate(factors):
        factor_to_vars.append(factor.var_ids)
        for var_id in factor.var_ids:
            var_to_factors[var_id].append(idx)
    return FactorGraph(
        variables=list(variables),
        factors=list(factors),
        var_to_factors=var_to_factors,
        factor_to_vars=factor_to_vars,
    )
