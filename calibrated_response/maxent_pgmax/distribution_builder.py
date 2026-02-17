"""Build distributions from estimate lists using pgmax-based inference."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from calibrated_response.maxent_pgmax.solver import PGMaxMaxEntSolver, SolverConfig
from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.models.query import EstimateUnion
from calibrated_response.models.variable import Variable


def _outer_product_joint(marginals: Sequence[np.ndarray]) -> np.ndarray:
    if not marginals:
        return np.zeros((0,), dtype=float)

    joint = np.array(marginals[0], dtype=float)
    for i in range(1, len(marginals)):
        shape = (1,) * i + (-1,)
        joint = joint.reshape(joint.shape + (1,)) * marginals[i].reshape(shape)

    total = float(joint.sum())
    if total > 0:
        joint /= total
    return joint


class DistributionBuilder:
    """Build target marginals from pgmax-backed approximate inference."""

    def __init__(
        self,
        variables: Sequence[Variable],
        estimates: Sequence[EstimateUnion],
        solver_config: Optional[SolverConfig] = None,
    ):
        self.variables = list(variables)
        self.estimates = list(estimates)
        self.solver_config = solver_config or SolverConfig()
        self.solver = PGMaxMaxEntSolver(self.variables, config=self.solver_config)
        self._var_name_to_idx = {v.name: i for i, v in enumerate(self.variables)}

    def build(
        self,
        target_variable: Optional[str] = None,
    ) -> tuple[HistogramDistribution, dict]:
        if not self.variables:
            raise ValueError("At least one variable is required")

        marginals, bin_edges_list, info = self.solver.solve(self.estimates)
        joint_distribution = _outer_product_joint(marginals)

        target_idx = self._target_index(target_variable)
        distribution = HistogramDistribution(
            bin_edges=bin_edges_list[target_idx].tolist(),
            bin_probabilities=marginals[target_idx].tolist(),
        )

        info = {
            **info,
            "target_variable": self.variables[target_idx].name,
            "joint_distribution": joint_distribution,
            "bin_edges_list": bin_edges_list,
            "marginals": {v.name: marginals[i] for i, v in enumerate(self.variables)},
            "joint_note": "Joint distribution is reconstructed from marginals (independence approximation).",
        }
        return distribution, info

    def get_all_marginals(self, info: dict) -> dict[str, HistogramDistribution]:
        marginals = info.get("marginals")
        edges = info.get("bin_edges_list")
        if marginals is None or edges is None:
            raise ValueError("Solver info does not contain marginals/bin_edges_list")

        output: dict[str, HistogramDistribution] = {}
        for i, var in enumerate(self.variables):
            probs = np.asarray(marginals[var.name], dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
            output[var.name] = HistogramDistribution(
                bin_edges=np.asarray(edges[i]).tolist(),
                bin_probabilities=probs.tolist(),
            )
        return output

    def _target_index(self, target_variable: Optional[str]) -> int:
        if target_variable is None:
            return 0
        if target_variable not in self._var_name_to_idx:
            raise ValueError(f"Unknown target variable: {target_variable}")
        return self._var_name_to_idx[target_variable]
