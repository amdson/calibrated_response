"""pgmax-backed approximate MaxEnt solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from calibrated_response.maxent_pgmax.discretization import DomainDiscretizer
from calibrated_response.models.query import (
    ConditionalExpectationEstimate,
    ConditionalProbabilityEstimate,
    EstimateUnion,
    ExpectationEstimate,
    ProbabilityEstimate,
)
from calibrated_response.models.variable import Variable


@dataclass
class SolverConfig:
    """Configuration for pgmax inference."""

    max_bins: int = 21
    bp_iters: int = 200
    damping: float = 0.5
    temperature: float = 1.0

    # Soft-constraint strengths.
    unary_strength: float = 2.5
    expectation_strength: float = 0.03
    conditional_strength: float = 1.5


class PGMaxMaxEntSolver:
    """Builds a factor graph and runs loopy BP for approximate inference."""

    def __init__(
        self,
        variables: Sequence[Variable],
        config: Optional[SolverConfig] = None,
    ):
        self.variables = list(variables)
        self.config = config or SolverConfig()
        self.discretizer = DomainDiscretizer(self.variables, max_bins=self.config.max_bins)
        self._name_to_index = {name: i for i, name in enumerate(self.discretizer.names())}

    def solve(
        self,
        estimates: Sequence[EstimateUnion],
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        """Solve for per-variable marginals with pgmax BP."""
        unary_evidence = self._init_unary_evidence()
        pairwise_potentials: dict[tuple[str, str], np.ndarray] = {}
        skipped: list[str] = []

        for estimate in estimates:
            self._accumulate_estimate(
                estimate=estimate,
                unary_evidence=unary_evidence,
                pairwise_potentials=pairwise_potentials,
                skipped=skipped,
            )

        marginals = self._run_bp(unary_evidence, pairwise_potentials)
        bin_edges = [b.bin_edges for b in self.discretizer.all_buckets()]

        info = {
            "n_variables": len(self.variables),
            "n_estimates": len(estimates),
            "n_pairwise_factors": len(pairwise_potentials),
            "skipped_constraints": skipped,
            "note": (
                "Laplacian smoothing is not implemented in pgmax form. "
                "Use stronger unary priors or post-hoc kernel smoothing on marginals as an alternative."
            ),
        }
        return marginals, bin_edges, info

    def _init_unary_evidence(self) -> dict[str, np.ndarray]:
        evidence: dict[str, np.ndarray] = {}
        for buckets in self.discretizer.all_buckets():
            evidence[buckets.name] = np.zeros(buckets.n_states, dtype=float)
        return evidence

    def _accumulate_estimate(
        self,
        estimate: EstimateUnion,
        unary_evidence: dict[str, np.ndarray],
        pairwise_potentials: dict[tuple[str, str], np.ndarray],
        skipped: list[str],
    ) -> None:
        if isinstance(estimate, ProbabilityEstimate):
            self._apply_probability_estimate(estimate, unary_evidence, skipped)
            return

        if isinstance(estimate, ExpectationEstimate):
            self._apply_expectation_estimate(estimate, unary_evidence, skipped)
            return

        if isinstance(estimate, ConditionalProbabilityEstimate):
            self._apply_conditional_probability_estimate(
                estimate,
                pairwise_potentials,
                skipped,
            )
            return

        if isinstance(estimate, ConditionalExpectationEstimate):
            skipped.append(
                f"{estimate.id}: conditional expectation not implemented for pgmax factors"
            )
            return

        skipped.append(f"{getattr(estimate, 'id', 'unknown')}: unsupported estimate type")

    def _apply_probability_estimate(
        self,
        estimate: ProbabilityEstimate,
        unary_evidence: dict[str, np.ndarray],
        skipped: list[str],
    ) -> None:
        prop = estimate.proposition
        if prop.variable not in unary_evidence:
            skipped.append(f"{estimate.id}: unknown variable {prop.variable}")
            return

        mask = self.discretizer.proposition_mask(prop)
        if not np.any(mask):
            skipped.append(f"{estimate.id}: proposition produced empty support")
            return

        p = float(np.clip(estimate.probability, 1e-6, 1 - 1e-6))
        log_odds = np.log(p / (1.0 - p))

        # Push states inside/outside the event in opposite directions.
        direction = np.where(mask, 1.0, -1.0)
        unary_evidence[prop.variable] += self.config.unary_strength * log_odds * direction

    def _apply_expectation_estimate(
        self,
        estimate: ExpectationEstimate,
        unary_evidence: dict[str, np.ndarray],
        skipped: list[str],
    ) -> None:
        if estimate.variable not in unary_evidence:
            skipped.append(f"{estimate.id}: unknown variable {estimate.variable}")
            return

        buckets = self.discretizer.buckets(estimate.variable)
        centers = buckets.bin_centers
        target = float(estimate.expected_value)

        spread = float(np.std(centers))
        if spread <= 1e-8:
            skipped.append(f"{estimate.id}: degenerate bucket centers")
            return

        signed = (centers - target) / spread
        unary_evidence[estimate.variable] -= self.config.expectation_strength * signed

    def _apply_conditional_probability_estimate(
        self,
        estimate: ConditionalProbabilityEstimate,
        pairwise_potentials: dict[tuple[str, str], np.ndarray],
        skipped: list[str],
    ) -> None:
        if len(estimate.conditions) != 1:
            skipped.append(
                f"{estimate.id}: only one-condition conditional probability is supported"
            )
            return

        cond = estimate.conditions[0]
        target_prop = estimate.proposition

        if cond.variable == target_prop.variable:
            skipped.append(f"{estimate.id}: self-conditional is not supported")
            return

        if cond.variable not in self._name_to_index or target_prop.variable not in self._name_to_index:
            skipped.append(f"{estimate.id}: unknown variables in conditional estimate")
            return

        target_mask = self.discretizer.proposition_mask(target_prop).astype(float)
        cond_mask = self.discretizer.proposition_mask(cond).astype(float)
        if not np.any(target_mask) or not np.any(cond_mask):
            skipped.append(f"{estimate.id}: conditional masks are empty")
            return

        p = float(np.clip(estimate.probability, 1e-6, 1 - 1e-6))
        weight = self.config.conditional_strength * np.log(p / (1.0 - p))

        a_name = target_prop.variable
        b_name = cond.variable
        key = tuple(sorted((a_name, b_name)))

        a_states = self.discretizer.buckets(key[0]).n_states
        b_states = self.discretizer.buckets(key[1]).n_states
        matrix = pairwise_potentials.get(key)
        if matrix is None:
            matrix = np.zeros((a_states, b_states), dtype=float)

        if key[0] == a_name:
            mask_a = target_mask
            mask_b = cond_mask
        else:
            mask_a = cond_mask
            mask_b = target_mask

        # Reward event states when condition is true.
        for i in range(mask_a.size):
            for j in range(mask_b.size):
                if mask_b[j] >= 0.5:
                    matrix[i, j] += weight if mask_a[i] >= 0.5 else -weight

        pairwise_potentials[key] = matrix

    def _run_bp(
        self,
        unary_evidence: dict[str, np.ndarray],
        pairwise_potentials: dict[tuple[str, str], np.ndarray],
    ) -> list[np.ndarray]:
        # pgmax BP currently expects at least one factor. If we only have unary
        # evidence, return independent softmax marginals directly.
        if not pairwise_potentials:
            marginals: list[np.ndarray] = []
            for name in self.discretizer.names():
                logits = np.asarray(unary_evidence[name], dtype=float) / max(self.config.temperature, 1e-8)
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                total = probs.sum()
                marginals.append(probs / total if total > 0 else np.ones_like(probs) / probs.size)
            return marginals

        try:
            from pgmax import fgraph, fgroup, infer, vgroup
        except ImportError as exc:
            raise ImportError(
                "pgmax is required for maxent_pgmax. Install it with `pip install pgmax`."
            ) from exc

        names = tuple(self.discretizer.names())
        num_states = np.array([self.discretizer.buckets(name).n_states for name in names])
        variables = vgroup.VarDict(variable_names=names, num_states=num_states)
        fg = fgraph.FactorGraph(variables)

        for (name_a, name_b), log_potentials in pairwise_potentials.items():
            var_a = variables[name_a]
            var_b = variables[name_b]
            fg.add_factors(
                fgroup.PairwiseFactorGroup(
                    variables_for_factors=[(var_a, var_b)],
                    log_potential_matrix=log_potentials,
                )
            )

        bp = infer.build_inferer(fg.bp_state, backend="bp")

        evidence = {
            variables[name]: unary_evidence[name]
            for name in names
        }
        bp_arrays = bp.init(evidence_updates=evidence)
        bp_arrays = bp.run(bp_arrays, num_iters=self.config.bp_iters, damping=self.config.damping)
        beliefs = bp.get_beliefs(bp_arrays)
        marginals_by_var = infer.get_marginals(beliefs)
        group_marginals = marginals_by_var.get(variables, {})

        marginals: list[np.ndarray] = []
        for name in names:
            if name not in group_marginals:
                raise KeyError(f"Missing marginal for variable '{name}' in pgmax output.")
            belief = np.asarray(group_marginals[name], dtype=float)
            total = float(belief.sum())
            if total <= 0:
                belief = np.ones_like(belief) / belief.size
            else:
                belief = belief / total
            marginals.append(belief)

        return marginals
