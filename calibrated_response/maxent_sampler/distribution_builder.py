"""Build distributions from variable and estimate lists using the flow maxent sampler.

The sample-native counterpart of ``maxent_smm.distribution_builder``: this module
converts high-level ``Variable`` + ``EstimateUnion`` objects (LLM-elicited
probability / expectation estimates) into the flow sampler's constraint grammar,
fits a :class:`~calibrated_response.maxent_sampler.flow_model.FlowSamplerModel`
by soft-constrained maximum entropy (exact joint entropy, fresh latents every
step), and reads marginal ``HistogramDistribution`` / ``BinaryDistribution``
objects off fresh samples.

Compared to the HMC/SMM builder there is no normaliser, no chain, and no
feature-vector compilation per shape: variables live in their original domains,
each estimate becomes one ``("expect", ...)`` / ``("cond_expect", ...)``
constraint (conditionals condition on the *conjunction* of their condition
events), and every readout is a Monte-Carlo estimate on the actual sampler.

    builder = DistributionBuilder(variables, estimates)
    dist, info = builder.build(target_variable="podium_sweep_occurrence")
    dist.probability                      # P(target = True)
    builder.get_all_marginals()           # {name: Distribution}
    builder.constraint_report()           # per-estimate target vs fitted
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from calibrated_response.models.distribution import (
    BinaryDistribution,
    Distribution,
    HistogramDistribution,
)
from calibrated_response.models.query import (
    ConditionalExpectationEstimate,
    ConditionalProbabilityEstimate,
    EqualityProposition,
    EstimateUnion,
    ExpectationEstimate,
    InequalityProposition,
    ProbabilityEstimate,
)
from calibrated_response.models.variable import (
    BinaryVariable,
    ContinuousVariable,
    Variable,
)
from calibrated_response.tn.discretize import ContinuousVar
from calibrated_response.maxent_sampler.flow_model import FlowSamplerModel
from calibrated_response.maxent_sampler.model import soft_gt, soft_lt

_EPS = 1e-30


def _and_all(fs: Sequence[Callable]) -> Callable:
    """Conjunction of event indicators (product works for soft and hard alike)."""
    if len(fs) == 1:
        return fs[0]

    def cond(x):
        c = fs[0](x)
        for f in fs[1:]:
            c = c * f(x)
        return c
    return cond


def _hard_event(idx: int, threshold: float, greater: bool) -> Callable:
    """Exact indicator for evaluation readouts (never inside the loss)."""
    if greater:
        return lambda x: (x[:, idx] > threshold).astype(np.float32)
    return lambda x: (x[:, idx] < threshold).astype(np.float32)


class DistributionBuilder:
    """Build probability distributions from variables and estimates using the
    flow-sampler soft-constrained MaxEnt solver.

    The public interface matches ``maxent_smm.DistributionBuilder``:

    * ``build(target_variable)`` → ``(Distribution, info)``
    * ``get_all_marginals()`` → ``dict[str, Distribution]``

    plus sampler-native extras: ``sample`` / ``sample_dict``, exact ``entropy()``,
    and ``constraint_report()`` (per-estimate target vs fitted, with the
    conditioning budget for conditional estimates).

    Parameters
    ----------
    variables : sequence of Variable
        ``ContinuousVariable`` (needs finite ``lower_bound < upper_bound``) and
        ``BinaryVariable`` (embedded as a continuous site on ``[0, 1]``, read
        out as ``P(x > 0.5)``).  Others are recorded in ``self.skipped``.
    estimates : sequence of EstimateUnion
        Probability / expectation / conditional estimates over those variables.
        Estimates referencing unknown variables are recorded in ``self.skipped``.
    prob_sd : float
        Belief width for probability targets: weight ``1 / (2 prob_sd²)``.
    value_rel_sd : float
        Belief width for expectation targets, as a *fraction of the variable's
        span* (keeps the squared residuals unit-free across mixed domains).
    sharpness : float
        Soft-indicator sharpness in span-normalised units — the effective
        sigmoid slope at a threshold on variable ``i`` is ``sharpness / span_i``.
    robust : bool
        If True, every estimate becomes an ``onoff`` constraint: a learnable
        Bernoulli credence can convict a contradicted estimate at a KL price of
        ``-log(p_broken)`` instead of dragging the whole joint.  Read the
        per-estimate credences with ``self.model.credences(self.params)``.
    p_broken : float
        Prior break probability for robust constraints.
    n_layers, hidden, s_max :
        Flow architecture (see :class:`FlowSampler`).
    n_bins : int
        Bin count for histogram marginal readouts (fitting itself is bin-free).
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        estimates: Sequence[EstimateUnion],
        prob_sd: float = 0.05,
        value_rel_sd: float = 0.05,
        sharpness: float = 20.0,
        robust: bool = False,
        p_broken: float = 0.05,
        n_layers: int = 8,
        hidden: int = 64,
        s_max: float = 3.0,
        n_bins: int = 32,
    ):
        self.variables = list(variables)
        self.estimates = list(estimates)
        self.prob_sd = float(prob_sd)
        self.value_rel_sd = float(value_rel_sd)
        self.sharpness = float(sharpness)
        self.robust = bool(robust)
        self.p_broken = float(p_broken)
        self.n_bins = int(n_bins)

        self.skipped: list[str] = []
        self.warnings: list[str] = []

        # ---- variables → ContinuousVar sites --------------------------------
        self.cvars: list[ContinuousVar] = []
        self.is_binary: list[bool] = []
        self.var_name_to_idx: dict[str, int] = {}
        for var in self.variables:
            if isinstance(var, BinaryVariable):
                cv = ContinuousVar(var.name, 0.0, 1.0, 2)
                binary = True
            elif isinstance(var, ContinuousVariable):
                lo, hi = float(var.lower_bound), float(var.upper_bound)
                if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                    self.skipped.append(f"{var.name}: invalid domain [{lo}, {hi}]")
                    continue
                cv = ContinuousVar(var.name, lo, hi, self.n_bins)
                binary = False
            else:
                self.skipped.append(
                    f"{getattr(var, 'name', '?')}: unsupported variable type")
                continue
            self.var_name_to_idx[var.name] = len(self.cvars)
            self.cvars.append(cv)
            self.is_binary.append(binary)
        if not self.cvars:
            raise ValueError("no usable variables")

        self.model = FlowSamplerModel(self.cvars, n_layers=n_layers,
                                      hidden=hidden, s_max=s_max)

        # ---- estimates → constraints + evaluation rows -----------------------
        self.constraints: list = []
        # report rows: (est_id, description, target, eval_fn, hard_cond_or_None)
        self._report_rows: list = []
        self._build_constraints()

        self.params = None
        self.history: list[float] = []

    # ======================================================================
    # estimate → constraint conversion
    # ======================================================================
    def _span(self, idx: int) -> float:
        cv = self.cvars[idx]
        return float(cv.upper - cv.lower)

    def _proposition_events(self, prop):
        """One proposition → (soft indicator for the loss, hard for readouts).

        Raises ``ValueError`` with a reason when the proposition is unusable —
        the caller skips the whole estimate.
        """
        name = prop.variable
        if name not in self.var_name_to_idx:
            raise ValueError(f"unknown variable {name!r}")
        idx = self.var_name_to_idx[name]

        if isinstance(prop, InequalityProposition):
            thr, greater = float(prop.threshold), bool(prop.is_lower_bound)
        elif isinstance(prop, EqualityProposition):
            if not isinstance(prop.value, bool):
                raise ValueError("non-boolean equality proposition")
            if not self.is_binary[idx]:
                raise ValueError(f"boolean equality on non-binary variable {name!r}")
            thr, greater = 0.5, bool(prop.value)
        else:
            raise ValueError(f"unsupported proposition type {type(prop).__name__}")

        k = self.sharpness / self._span(idx)
        soft = soft_gt(idx, thr, k) if greater else soft_lt(idx, thr, k)
        return soft, _hard_event(idx, thr, greater)

    def _moment_events(self, var_name: str):
        """Variable name → (value feature, its site index)."""
        if var_name not in self.var_name_to_idx:
            raise ValueError(f"unknown variable {var_name!r}")
        idx = self.var_name_to_idx[var_name]
        return (lambda x: x[:, idx]), idx

    def _clip_target(self, est_id: str, idx: int, value: float) -> float:
        cv = self.cvars[idx]
        clipped = float(np.clip(value, cv.lower, cv.upper))
        if clipped != value:
            self.warnings.append(
                f"{est_id}: target {value} outside domain "
                f"[{cv.lower}, {cv.upper}] of {cv.name!r}; clipped to {clipped}")
        return clipped

    def _add(self, f, cond, target, sd, est_id, desc, eval_fn, hard_cond):
        """Append one constraint (plain or robust) plus its report row."""
        if self.robust:
            self.constraints.append(("onoff", f, cond, target, sd, self.p_broken))
        elif cond is None:
            self.constraints.append(("expect", f, target, 1.0 / (2.0 * sd * sd)))
        else:
            self.constraints.append(
                ("cond_expect", f, cond, target, 1.0 / (2.0 * sd * sd)))
        self._report_rows.append((est_id, desc, target, eval_fn, hard_cond))

    def _build_constraints(self) -> None:
        for est in self.estimates:
            try:
                if isinstance(est, ProbabilityEstimate):
                    soft, hard = self._proposition_events(est.proposition)
                    self._add(soft, None, float(est.probability), self.prob_sd,
                              est.id, est.to_query_estimate(),
                              lambda x, h=hard: float(np.mean(h(x))), None)

                elif isinstance(est, ExpectationEstimate):
                    f, idx = self._moment_events(est.variable)
                    tg = self._clip_target(est.id, idx, float(est.expected_value))
                    self._add(f, None, tg, self.value_rel_sd * self._span(idx),
                              est.id, est.to_query_estimate(),
                              lambda x, f=f: float(np.mean(f(x))), None)

                elif isinstance(est, ConditionalProbabilityEstimate):
                    if not est.conditions:
                        raise ValueError("conditional estimate with no conditions")
                    soft, hard = self._proposition_events(est.proposition)
                    pairs = [self._proposition_events(c) for c in est.conditions]
                    soft_c = _and_all([s for s, _ in pairs])
                    hard_c = _and_all([h for _, h in pairs])
                    def ev(x, h=hard, hc=hard_c):
                        c = hc(x)
                        return float(np.sum(h(x) * c) / (np.sum(c) + _EPS))
                    self._add(soft, soft_c, float(est.probability), self.prob_sd,
                              est.id, est.to_query_estimate(), ev, hard_c)

                elif isinstance(est, ConditionalExpectationEstimate):
                    if not est.conditions:
                        raise ValueError("conditional estimate with no conditions")
                    f, idx = self._moment_events(est.variable)
                    tg = self._clip_target(est.id, idx, float(est.expected_value))
                    pairs = [self._proposition_events(c) for c in est.conditions]
                    soft_c = _and_all([s for s, _ in pairs])
                    hard_c = _and_all([h for _, h in pairs])
                    def ev(x, f=f, hc=hard_c):
                        c = hc(x)
                        return float(np.sum(f(x) * c) / (np.sum(c) + _EPS))
                    self._add(f, soft_c, tg, self.value_rel_sd * self._span(idx),
                              est.id, est.to_query_estimate(), ev, hard_c)

                else:
                    self.skipped.append(
                        f"{getattr(est, 'id', '?')}: unsupported estimate type")
            except Exception as exc:   # noqa: BLE001
                self.skipped.append(f"{getattr(est, 'id', '?')}: {exc}")

    # ======================================================================
    # fit
    # ======================================================================
    def fit(self, steps: int = 3000, lr: float = 1e-3, n_samples: int = 2048,
            entropy_reg: float = 1.0, seed: int = 0, **kw):
        """Fit the flow by soft-constrained maxent (Adam, fresh latents per step)."""
        loss = self.model.constraint_loss(
            self.constraints, entropy_reg=entropy_reg, n_samples=n_samples)
        self.params, self.history = self.model.optimize(
            loss, steps=steps, lr=lr, seed=seed, **kw)
        return self.params, self.history

    def _require_fit(self):
        if self.params is None:
            raise RuntimeError("call fit() (or build()) first")

    # ======================================================================
    # readouts
    # ======================================================================
    def sample(self, n_samples: int = 20000, seed: int = 0) -> np.ndarray:
        """``(n_samples, n_vars)`` samples in original units; binary sites in [0,1]."""
        self._require_fit()
        return self.model.sample(self.params, n_samples, seed=seed)

    def sample_dict(self, n_samples: int = 20000, seed: int = 0) -> dict[str, np.ndarray]:
        """Samples as ``{variable_name: column}``; binary variables thresholded to {0,1}."""
        x = self.sample(n_samples, seed=seed)
        out = {}
        for name, i in self.var_name_to_idx.items():
            col = x[:, i]
            out[name] = (col > 0.5).astype(float) if self.is_binary[i] else col
        return out

    def marginal(self, var_name: str, n_samples: int = 20000,
                 seed: int = 0) -> Distribution:
        """Marginal of one variable: ``BinaryDistribution`` for binary variables,
        ``HistogramDistribution`` (over the variable's domain) otherwise."""
        self._require_fit()
        if var_name not in self.var_name_to_idx:
            raise KeyError(f"unknown variable {var_name!r}")
        i = self.var_name_to_idx[var_name]
        x = self.sample(n_samples, seed=seed)
        if self.is_binary[i]:
            return BinaryDistribution(probability=float(np.mean(x[:, i] > 0.5)))
        cv = self.cvars[i]
        edges = np.linspace(cv.lower, cv.upper, self.n_bins + 1)
        counts, _ = np.histogram(x[:, i], bins=edges)
        probs = counts / max(counts.sum(), 1)
        return HistogramDistribution(bin_edges=edges.tolist(),
                                     bin_probabilities=probs.tolist())

    def get_all_marginals(self, n_samples: int = 20000,
                          seed: int = 0) -> dict[str, Distribution]:
        return {name: self.marginal(name, n_samples, seed)
                for name in self.var_name_to_idx}

    def entropy(self, n_samples: int = 20000, seed: int = 0) -> float:
        """Exact joint differential entropy of the fitted flow (nats)."""
        self._require_fit()
        return self.model.entropy(self.params, n_samples=n_samples, seed=seed)

    def constraint_report(self, n_samples: int = 50000, seed: int = 0):
        """Per-estimate fit check on fresh samples with **hard** indicators.

        Returns rows ``{id, estimate, target, fitted, error, p_cond, credence}``
        — ``p_cond`` is the conditioning-event probability (the Monte-Carlo
        budget behind conditional estimates; treat fitted values with tiny
        ``p_cond`` as noisy), ``credence`` the robust gate's posterior
        ``P(active)`` when ``robust=True``.
        """
        self._require_fit()
        x = self.sample(n_samples, seed=seed)
        creds = self.model.credences(self.params) if self.robust else None
        rows = []
        for k, (eid, desc, target, ev, hard_cond) in enumerate(self._report_rows):
            fitted = ev(x)
            rows.append({
                "id": eid,
                "estimate": desc,
                "target": target,
                "fitted": fitted,
                "error": fitted - target,
                "p_cond": (float(np.mean(hard_cond(x)))
                           if hard_cond is not None else None),
                "credence": float(creds[k]) if creds is not None else None,
            })
        return rows

    # ======================================================================
    # one-call interface (parity with maxent_smm.DistributionBuilder)
    # ======================================================================
    def build(self, target_variable: Optional[str] = None, **fit_kw):
        """Fit and return ``(target marginal or None, info)``.

        ``fit_kw`` is forwarded to :meth:`fit` (``steps``, ``lr``, ``n_samples``,
        ``entropy_reg``, ``seed``, ``log_every``...).
        """
        self.fit(**fit_kw)
        info = {
            "n_variables": len(self.cvars),
            "n_estimates": len(self.estimates),
            "n_constraints": len(self.constraints),
            "skipped_constraints": self.skipped,
            "warnings": self.warnings,
            "history": self.history,
            "entropy": self.entropy(),
            "report": self.constraint_report(),
            "params": self.params,
            "model": self.model,
        }
        dist = self.marginal(target_variable) if target_variable else None
        return dist, info
