from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from calibrated_response.maxent_large_v1.adapters import adapt_problem
from calibrated_response.maxent_large_v1.training import SMMConfig, fit_smm
from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.models.query import EstimateUnion
from calibrated_response.models.variable import Variable


@dataclass(frozen=True)
class BuildInfo:
    target_variable: str
    n_variables: int
    n_estimates: int
    n_factors: int


class DistributionBuilder:
    def __init__(
        self,
        variables: Sequence[Variable],
        estimates: Sequence[EstimateUnion],
        solver_config: Optional[SMMConfig] = None,
    ):
        self.variables = list(variables)
        self.estimates = list(estimates)
        self.solver_config = solver_config or SMMConfig()
        self._var_name_to_idx = {v.name: i for i, v in enumerate(self.variables)}

    def _uniform_distribution(self, idx: int, bin_edges: list[np.ndarray]) -> HistogramDistribution:
        probs = np.ones((len(bin_edges[idx]) - 1,), dtype=float)
        probs /= probs.sum()
        return HistogramDistribution(bin_edges=bin_edges[idx].tolist(), bin_probabilities=probs.tolist())

    def build(self, target_variable: Optional[str] = None) -> tuple[HistogramDistribution, dict]:
        if not self.variables:
            raise ValueError("At least one variable is required")
        
        

        adapted = adapt_problem(self.variables, self.estimates, max_bins=self.solver_config.max_bins)
        idx = self._target_index(target_variable)

        trained_graph, history, buffer = fit_smm(
            adapted.graph,
            adapted.targets,
            self.solver_config,
            skipped_constraints=adapted.skipped_constraints,
        )
        states = np.asarray(buffer.get_states())

        marginals: dict[str, np.ndarray] = {}
        for i, variable in enumerate(self.variables):
            n_states = trained_graph.cardinality(i)
            counts = np.bincount(states[:, i], minlength=n_states).astype(float)
            total = counts.sum()
            probs = counts / total if total > 0 else np.ones((n_states,), dtype=float) / n_states
            marginals[variable.name] = probs

        target_name = self.variables[idx].name
        distribution = HistogramDistribution(
            bin_edges=np.asarray(adapted.bin_edges_list[idx], dtype=float).tolist(),
            bin_probabilities=np.asarray(marginals[target_name], dtype=float).tolist(),
        )

        info = {
            "target_variable": target_name,
            "n_variables": len(self.variables),
            "n_estimates": len(self.estimates),
            "n_factors": len(trained_graph.factors),
            "history": history,
            "skipped_constraints": adapted.skipped_constraints,
            "bin_edges_list": adapted.bin_edges_list,
            "marginals": marginals,
            "joint_note": "No dense joint is constructed; marginals estimated from persistent chains.",
        }
        return distribution, info

    def get_all_marginals(self, info: dict) -> dict[str, HistogramDistribution]:
        marginals = info.get("marginals")
        edges = info.get("bin_edges_list")
        if marginals is None or edges is None:
            raise ValueError("Solver info does not contain marginals/bin_edges_list")

        out: dict[str, HistogramDistribution] = {}
        for i, variable in enumerate(self.variables):
            probs = np.asarray(marginals[variable.name], dtype=float)
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones_like(probs) / max(len(probs), 1)
            out[variable.name] = HistogramDistribution(
                bin_edges=np.asarray(edges[i], dtype=float).tolist(),
                bin_probabilities=probs.tolist(),
            )
        return out

    def _target_index(self, target_variable: Optional[str]) -> int:
        if target_variable is None:
            return 0
        if target_variable not in self._var_name_to_idx:
            raise ValueError(f"Unknown target variable: {target_variable}")
        return self._var_name_to_idx[target_variable]
