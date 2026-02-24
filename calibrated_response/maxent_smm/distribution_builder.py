"""Build distributions from variable and estimate lists using HMC MaxEnt.

This module converts high-level ``Variable`` + ``EstimateUnion`` objects into
declarative ``FeatureSpec`` / target pairs, runs the persistent-HMC MaxEnt
solver, and extracts marginal ``HistogramDistribution`` objects from the
resulting chain samples.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence
import uuid

import numpy as np

from calibrated_response.maxent_smm.variable_spec import BetaPriorSpec, GaussianPriorSpec, UniformPriorSpec, VariableSpec
from calibrated_response.models.distribution import HistogramDistribution
from calibrated_response.models.query import (
    EstimateUnion,
    ProbabilityEstimate,
    ExpectationEstimate,
    ConditionalProbabilityEstimate,
    ConditionalExpectationEstimate,
    EqualityProposition,
    InequalityProposition,
)
from calibrated_response.models.variable import Variable, ContinuousVariable, BinaryVariable
from calibrated_response.maxent_smm.features import (
    FeatureSpec,
    MomentFeature,
    SoftThresholdFeature,
    SoftIndicatorFeature,
    CenteredConditionalFeature,
    CenteredConditionalMomentFeature,
)
from calibrated_response.maxent_smm.maxent_solver import MaxEntSolver, JAXSolverConfig
from calibrated_response.maxent_smm.normalizer import ContinuousDomainNormalizer
from calibrated_response.maxent_smm.energy_model import EnergyModel

def _histogram_marginal(
    states: np.ndarray,
    var_idx: int,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Histogram chain samples for one variable into ``bin_edges``.

    Parameters
    ----------
    states : (C, D)  chain samples in [0, 1]
    var_idx : int
    bin_edges : 1-D array of length n_bins + 1, in [0, 1]

    Returns
    -------
    probs : (n_bins,) normalised probabilities
    """
    counts, _ = np.histogram(states[:, var_idx], bins=bin_edges)
    total = counts.sum()
    if total > 0:
        return counts.astype(float) / total
    return np.ones(len(counts), dtype=float) / max(len(counts), 1)


# ---------------------------------------------------------------------------
# DistributionBuilder
# ---------------------------------------------------------------------------

class DistributionBuilder:
    """Build probability distributions from variables and estimates using
    continuous-HMC multivariate MaxEnt.

    The public interface matches ``maxent.DistributionBuilder`` and
    ``maxent_large_v1.DistributionBuilder``:

    * ``build(target_variable)`` → ``(HistogramDistribution, info)``
    * ``get_all_marginals(info)`` → ``dict[str, HistogramDistribution]``
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        estimates: Sequence[EstimateUnion],
        energy_fn: Callable, 
        init_theta: np.ndarray,
        solver_config: Optional[JAXSolverConfig] = None,
        extra_feature_constraints: Optional[Sequence[tuple[FeatureSpec, float]]] = None,
    ):
        self.variables = list(variables)
        self.estimates = list(estimates)
        self.energy_fn = energy_fn
        self.init_theta = init_theta
        self.solver_config = solver_config or JAXSolverConfig()
        self.extra_feature_constraints = extra_feature_constraints or []
        self.var_name_to_idx = {v.name: i for i, v in enumerate(self.variables)}
        self.var_name_to_var = {v.name: v for v in self.variables}

        # Domain normaliser  (maps original domains → [0, 1])
        self.normalizer = ContinuousDomainNormalizer(self.variables)

        # Convert estimates → (FeatureSpec, target_value) pairs
        self.var_specs: list[VariableSpec] = []
        self.feature_specs: list[FeatureSpec] = []
        self.feature_targets: list[float] = []
        self.skipped: list[str] = []
        self._build_variables() 
        self._build_features()

        # Solver
        self.solver = MaxEntSolver(self.solver_config)
        self._energy_model: Optional[EnergyModel] = None

    # ------------------------------------------------------------------
    # Feature construction from estimates
    # ------------------------------------------------------------------

    def _build_variables(self) -> None:
        """Walk through variables and append to ``self.var_specs``."""
        if self.solver_config.continuous_prior not in {"gaussian", "beta", "uniform"}:
            raise ValueError(f"Unsupported continuous_prior: {self.solver_config.continuous_prior}")
        if self.solver_config.continuous_prior == "gaussian":
            prior_cls = GaussianPriorSpec
            prior_kwargs = dict(mean=0.5, std=0.25)
        elif self.solver_config.continuous_prior == "beta":
            prior_cls = BetaPriorSpec
            prior_kwargs = dict(alpha=2.0, beta=2.0)
        else:
            prior_cls = UniformPriorSpec
            prior_kwargs = {}
        for var in self.variables:
            if isinstance(var, ContinuousVariable):
                var_spec = VariableSpec(
                    name=var.name,
                    description=var.description,
                    type="continuous",
                    prior=prior_cls(**prior_kwargs),
                )
            elif isinstance(var, BinaryVariable):
                var_spec = VariableSpec(
                    name=var.name,
                    description=var.description,
                    type="binary",
                    prior=UniformPriorSpec(),
                )
            else:
                self.skipped.append(f"{getattr(var, 'name', '?')}: unsupported variable type")
                continue
            self.var_specs.append(var_spec)

    def _build_features(self) -> None:
        """Walk through estimates and append to ``self.feature_specs`` / ``self.feature_targets``."""
        for est in self.estimates:
            try:
                if isinstance(est, ProbabilityEstimate):
                    self._add_probability_features(est)
                elif isinstance(est, ExpectationEstimate):
                    self._add_expectation_features(est)
                elif isinstance(est, ConditionalProbabilityEstimate):
                    self._add_conditional_probability_features(est)
                elif isinstance(est, ConditionalExpectationEstimate):
                    self._add_conditional_expectation_features(est)
                else:
                    self.skipped.append(f"{getattr(est, 'id', '?')}: unsupported estimate type")
            except Exception as exc:   # noqa: BLE001
                self.skipped.append(f"{getattr(est, 'id', '?')}: {exc}")
        for feature_spec, target in self.extra_feature_constraints:
            self.feature_specs.append(feature_spec)
            self.feature_targets.append(target)

    # -- Probability estimates ------------------------------------------

    def _add_probability_features(self, est: ProbabilityEstimate) -> None:
        prop = est.proposition
        var_name = prop.variable
        if var_name not in self.var_name_to_idx:
            self.skipped.append(f"{est.id}: unknown variable {var_name}")
            return
        idx = self.var_name_to_idx[var_name]

        if isinstance(prop, InequalityProposition):
            norm_thresh = self.normalizer.normalize_value(var_name, prop.threshold)
            direction = "greater" if prop.is_lower_bound else "less"
            self.feature_specs.append(
                SoftThresholdFeature(var_idx=idx, threshold=norm_thresh, direction=direction, sharpness=self.solver_config.indicator_sharpness)
            )
            self.feature_targets.append(float(est.probability))

        elif isinstance(prop, EqualityProposition):
            # Binary: P(X = True) → soft threshold at 0.5 (greater)
            if isinstance(prop.value, bool):
                direction = "greater" if prop.value else "less"
                self.feature_specs.append(
                    SoftThresholdFeature(var_idx=idx, threshold=0.5, direction=direction, sharpness=self.solver_config.indicator_sharpness)
                )
                self.feature_targets.append(float(est.probability))
            else:
                self.skipped.append(f"{est.id}: non-binary equality not supported")

    # -- Expectation estimates ------------------------------------------

    def _add_expectation_features(self, est: ExpectationEstimate) -> None:
        var_name = est.variable
        if var_name not in self.var_name_to_idx:
            self.skipped.append(f"{est.id}: unknown variable {var_name}")
            return
        idx = self.var_name_to_idx[var_name]
        norm_val = self.normalizer.normalize_value(var_name, est.expected_value)
        self.feature_specs.append(MomentFeature(var_idx=idx, order=1))
        self.feature_targets.append(float(norm_val))

    # -- Conditional probability estimates ------------------------------

    def _add_conditional_probability_features(self, est: ConditionalProbabilityEstimate) -> None:
        prop = est.proposition
        target_name = prop.variable
        if target_name not in self.var_name_to_idx:
            self.skipped.append(f"{est.id}: unknown target variable {target_name}")
            return
        target_idx = self.var_name_to_idx[target_name]

        # Parse target threshold
        if isinstance(prop, InequalityProposition):
            target_thresh = self.normalizer.normalize_value(target_name, prop.threshold)
            target_dir = "greater" if prop.is_lower_bound else "less"
        elif isinstance(prop, EqualityProposition) and isinstance(prop.value, bool):
            target_thresh = 0.5
            target_dir = "greater" if prop.value else "less"
        else:
            self.skipped.append(f"{est.id}: unsupported proposition type in conditional")
            return

        if not est.conditions:
            self.skipped.append(f"{est.id}: no conditions in conditional probability")
            return

        # For each condition variable, add one CenteredConditionalFeature.

        for cond in est.conditions:
            cond_name = cond.variable
            if cond_name not in self.var_name_to_idx:
                self.skipped.append(f"{est.id}: unknown condition variable {cond_name}")
                continue
            cond_idx = self.var_name_to_idx[cond_name]

            if isinstance(cond, InequalityProposition):
                cond_thresh = self.normalizer.normalize_value(cond_name, cond.threshold)
                cond_dir = "greater" if cond.is_lower_bound else "less"
            elif isinstance(cond, EqualityProposition) and isinstance(cond.value, bool):
                cond_thresh = 0.5
                cond_dir = "greater" if cond.value else "less"
            else:
                self.skipped.append(f"{est.id}: unsupported condition proposition type")
                continue

            # Encode P(target | cond) = p via the centered conditional feature:
            #   E[σ_cond · (σ_target − p)] = 0  ⟺  P(target | cond) = p
            # This is exact and requires no knowledge of P(cond).
            self.feature_specs.append(
                CenteredConditionalFeature(
                    target_var=target_idx,
                    target_threshold=target_thresh,
                    target_direction=target_dir,
                    cond_var=cond_idx,
                    cond_threshold=cond_thresh,
                    cond_direction=cond_dir,
                    p_target_given_cond=float(est.probability),
                    sharpness=self.solver_config.indicator_sharpness,
                )
            )
            self.feature_targets.append(0.0)


    # -- Conditional expectation estimates ------------------------------

    def _add_conditional_expectation_features(self, est: ConditionalExpectationEstimate) -> None:
        """Encode E[X | C] via the identity  E[X·I(C)] = E[X|C]·P(C).

        We add:
        1. A condition indicator feature  (target ≈ P(C))
        2. A weighted-moment feature x · I(C)  (target ≈ E[X|C] · P(C))
        """
        var_name = est.variable
        if var_name not in self.var_name_to_idx:
            self.skipped.append(f"{est.id}: unknown variable {var_name}")
            return
        target_idx = self.var_name_to_idx[var_name]
        norm_cond_exp = self.normalizer.normalize_value(var_name, est.expected_value)

        if not est.conditions:
            self.skipped.append(f"{est.id}: no conditions in conditional expectation")
            return

        for cond in est.conditions:
            cond_name = cond.variable
            if cond_name not in self.var_name_to_idx:
                self.skipped.append(f"{est.id}: unknown condition variable {cond_name}")
                continue
            cond_idx = self.var_name_to_idx[cond_name]

            if isinstance(cond, InequalityProposition):
                cond_thresh = self.normalizer.normalize_value(cond_name, cond.threshold)
                cond_dir = "greater" if cond.is_lower_bound else "less"
            elif isinstance(cond, EqualityProposition) and isinstance(cond.value, bool):
                cond_thresh = 0.5
                cond_dir = "greater" if cond.value else "less"
            else:
                self.skipped.append(f"{est.id}: unsupported condition proposition")
                continue

            # Encode E[X | cond] = μ via the centered conditional moment feature:
            #   E[σ_cond · (x − μ)] = 0  ⟺  E[X | cond] = μ
            # This is exact and requires no knowledge of P(cond).
            self.feature_specs.append(
                CenteredConditionalMomentFeature(
                    target_var=target_idx,
                    cond_var=cond_idx,
                    cond_threshold=cond_thresh,
                    cond_direction=cond_dir,
                    expected_value=float(norm_cond_exp),
                    sharpness=self.solver_config.indicator_sharpness,
                )
            )
            self.feature_targets.append(0.0)

    # ------------------------------------------------------------------
    # build / solve
    # ------------------------------------------------------------------

    def build(
        self,
        target_variable: Optional[str] = None,
    ) -> tuple[HistogramDistribution, dict]:
        """Build a distribution for the target variable.

        Returns
        -------
        distribution : HistogramDistribution
            Marginal for the requested variable (in original domain).
        info : dict
            Diagnostics, chain states, bin edges, etc.
        """
        if not self.variables:
            raise ValueError("At least one variable is required")

        target_var = self._get_target_variable(target_variable)
        target_idx = self.var_name_to_idx[target_var.name]
        n_vars = len(self.variables)
        n_bins = self.solver_config.max_bins

        # Handle no-feature case → uniform
        if not self.feature_specs:
            dist = self._uniform_distribution(target_var)
            info: dict = {
                "message": "No features derived from estimates; returning uniform.",
                "target_variable": target_var.name,
                "n_estimates": len(self.estimates),
                "skipped_constraints": self.skipped,
            }
            return dist, info

        # Build & solve
        import jax.numpy as jnp
        self.solver.build(
            var_specs=self.var_specs,
            feature_specs=self.feature_specs,
            feature_targets=jnp.array(self.feature_targets, dtype=jnp.float32),
            energy_fn=self.energy_fn,
            init_theta=self.init_theta,
        )
        print("Compiled maxent solver")

        theta, solver_info = self.solver.solve()

        theta = np.asarray(theta)  # (n_features,) numpy

        # Chain samples are in [0, 1] normalised domain
        chain_states = solver_info["chain_states"]   # (C, D) numpy

        # Compute all marginals as histograms (normalised domain)
        bin_edges_norm: dict[str, np.ndarray] = {}
        marginals: dict[str, np.ndarray] = {}
        bin_edges_orig: dict[str, np.ndarray] = {}

        for i, var in enumerate(self.variables):
            edges_n = self.normalizer.normalized_bin_edges(var.name, n_bins)
            probs = _histogram_marginal(chain_states, i, edges_n)
            edges_o = self.normalizer.denormalize_edges(var.name, edges_n)
            bin_edges_norm[var.name] = edges_n
            bin_edges_orig[var.name] = edges_o
            marginals[var.name] = probs

        # Target distribution
        distribution = HistogramDistribution(
            bin_edges=bin_edges_orig[target_var.name].tolist(),
            bin_probabilities=marginals[target_var.name].tolist(),
        )

        # Build the energy model for downstream use
        self._energy_model = EnergyModel(
            theta=self.solver._theta,
            energy_fn=self.solver._energy_fn,
            grad_energy_fn=self.solver._grad_energy_fn,
            feature_vector_fn=self.solver._feature_vector_fn,
            hmc_step_fn=self.solver._hmc_step_fn,
            normalizer=self.normalizer,
            variables=self.variables,
            hmc_config=self.solver._hmc_config,
        )

        info = {
            "target_variable": target_var.name,
            "n_variables": n_vars,
            "n_estimates": len(self.estimates),
            "n_features": len(self.feature_specs),
            "skipped_constraints": self.skipped,
            "history": solver_info.get("history"),
            "bin_edges_list": [bin_edges_orig[v.name] for v in self.variables],
            "bin_edges_list_normalized": [bin_edges_norm[v.name] for v in self.variables],
            "marginals": marginals,
            "chain_states": chain_states,
            "theta": solver_info.get("theta"),
            "converged": solver_info.get("converged", False),
            "energy_model": self._energy_model,
            "joint_note": "No dense joint is constructed; marginals estimated from persistent chains.",
        }
        return distribution, info

    # ------------------------------------------------------------------
    # get_all_marginals
    # ------------------------------------------------------------------

    def get_all_marginals(self, info: dict) -> dict[str, HistogramDistribution]:
        """Extract marginal distributions for every variable from solver info."""
        marginals = info.get("marginals")
        edges_list = info.get("bin_edges_list")
        if marginals is None or edges_list is None:
            raise ValueError("Solver info does not contain marginals / bin_edges_list")

        out: dict[str, HistogramDistribution] = {}
        for i, var in enumerate(self.variables):
            probs = np.asarray(marginals[var.name], dtype=float)
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones_like(probs) / max(len(probs), 1)
            out[var.name] = HistogramDistribution(
                bin_edges=np.asarray(edges_list[i], dtype=float).tolist(),
                bin_probabilities=probs.tolist(),
            )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_energy_model(self) -> EnergyModel:
        """Return the trained energy model.

        Must be called after ``build()``.  The model provides:

        * ``energy(x)`` / ``energy_original(x)`` — evaluate E_θ(x)
        * ``log_prob(x)`` / ``log_prob_original(x)`` — unnormalised log p
        * ``sample(n)`` / ``sample_original(n)`` — draw HMC samples
        * ``sample_original_dict(n)`` — samples as ``{name: value}`` dicts
        * ``feature_vector(x)`` — evaluate f(x)
        """
        if self._energy_model is None:
            raise RuntimeError("Call build() before get_energy_model().")
        return self._energy_model

    def _get_target_variable(self, target_variable: Optional[str]) -> Variable:
        if target_variable is not None:
            if target_variable in self.var_name_to_var:
                return self.var_name_to_var[target_variable]
            raise ValueError(f"Variable '{target_variable}' not found")
        for v in self.variables:
            if getattr(v, "is_target", False):
                return v
        return self.variables[0]

    def _uniform_distribution(self, var: Variable) -> HistogramDistribution:
        n_bins = self.solver_config.max_bins
        if isinstance(var, ContinuousVariable):
            lo, hi = var.get_domain()
            edges = np.linspace(lo, hi, n_bins + 1)
        elif isinstance(var, BinaryVariable):
            edges = np.array([0.0, 0.5, 1.0])
            n_bins = 2
        else:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        probs = np.ones(n_bins) / n_bins
        return HistogramDistribution(
            bin_edges=edges.tolist(),
            bin_probabilities=probs.tolist(),
        )