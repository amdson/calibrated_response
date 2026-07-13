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
    CorrelationEstimate,
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
        Belief width for probability targets in ABSOLUTE probability units
        (only used when ``prob_penalty="abs"``): weight ``1 / (2 prob_sd²)``.
    prob_penalty : str
        ``"logit"`` (default): probability residuals are penalised in
        log-odds, so belief slack is multiplicative and uniform across the
        scale — an absolute penalty gives tail targets (p=0.02, sd=0.05)
        several-x odds of nearly-free slack that the entropy term spends
        inflating rare events. ``"abs"``: legacy absolute-scale penalty.
    prob_logit_sd : float
        Belief width for probability targets in log-odds units (used when
        ``prob_penalty="logit"``); 0.3 ≈ a x1.35 odds tolerance.
    value_rel_sd : float
        Belief width for expectation targets, as a *fraction of the variable's
        span* (keeps the squared residuals unit-free across mixed domains).
    corr_sd : float
        Belief width for correlation targets (weight ``1 / (2 corr_sd²)``);
        correlations are scale-free so no span normalisation is needed.
    sharpness : float, optional
        Soft-indicator sharpness in span-normalised units — the effective
        sigmoid slope at a threshold on variable ``i`` is ``sharpness / span_i``.
        Default: 80 under ``prob_penalty="logit"`` (log-odds precision needs
        the soft mean to track the hard probability into the tails), 20 under
        ``"abs"``.
    robust : bool
        If True, every estimate becomes an ``onoff`` constraint: a learnable
        Bernoulli credence can convict a contradicted estimate at a KL price of
        ``-log(p_broken)`` instead of dragging the whole joint.  Read the
        per-estimate credences with ``self.model.credences(self.params)``.
    p_broken : float
        Prior break probability for robust constraints.
    anchor_variable : str, optional
        Name of a variable whose *direct* (unconditional) probability
        estimate is the anchor of the fit.  In robust mode that estimate
        stays ungated — otherwise the solver can discard the anchor at a
        cost of ``-log(p_broken)`` nats and follow an overconfident
        coupling estimate instead.
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
        prob_penalty: str = "logit",
        prob_logit_sd: float = 0.3,
        value_rel_sd: float = 0.05,
        corr_sd: float = 0.15,
        sharpness: Optional[float] = None,
        robust: bool = False,
        p_broken: float = 0.05,
        anchor_variable: Optional[str] = None,
        n_layers: int = 8,
        hidden: int = 64,
        s_max: float = 3.0,
        n_bins: int = 32,
    ):
        self.variables = list(variables)
        self.estimates = list(estimates)
        if prob_penalty not in ("logit", "abs"):
            raise ValueError(f"prob_penalty must be 'logit' or 'abs', "
                             f"got {prob_penalty!r}")
        self.prob_sd = float(prob_sd)
        self.prob_penalty = prob_penalty
        self.prob_logit_sd = float(prob_logit_sd)
        self.value_rel_sd = float(value_rel_sd)
        self.corr_sd = float(corr_sd)
        # log-odds penalties need soft indicators that track the hard event
        # to tail precision: at sharpness 20 the soft mean picks up ~0.02 of
        # sub-threshold leakage — invisible to an absolute penalty, a ~10x
        # odds error to a logit one (validated: 80 lands p=0.02 within 10%
        # in odds; 20 lands at 0.002)
        if sharpness is None:
            sharpness = 80.0 if prob_penalty == "logit" else 20.0
        self.sharpness = float(sharpness)
        self.robust = bool(robust)
        self.p_broken = float(p_broken)
        self.anchor_variable = anchor_variable
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
        # report rows: (est_id, description, target, eval_fn,
        #               hard_cond_or_None, gate_idx_or_None)
        self._report_rows: list = []
        self._n_gates = 0            # onoff gates appended so far (robust mode)
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

    def _clip_prob_target(self, est_id: str, p: float) -> float:
        """Soften certainty claims for the logit penalty.

        ``_logit`` clips at 1e-4, so an elicited p = 1.0 becomes a target of
        ~+9.2 log-odds — a near-infinite pull that lets one overconfident
        estimate outvote every other constraint (and, in robust mode, makes
        gating off the *anchor* cheaper than gating off the bad estimate).
        Elicited certainty means "very likely", not literal odds of 10^4:1.
        """
        if self.prob_penalty != "logit":
            return p
        clipped = float(np.clip(p, 0.01, 0.99))
        if clipped != p:
            self.warnings.append(
                f"{est_id}: probability {p} softened to {clipped} for the "
                f"logit penalty")
        return clipped

    def _add(self, f, cond, target, sd, est_id, desc, eval_fn, hard_cond,
             scale: float = 1.0, space: str = "abs", gated: bool = True):
        """Append one constraint (plain or robust) plus its report row.

        ``scale`` is the natural unit of the residual (the variable span for
        expectation targets, 1 for probabilities/correlations) so report
        errors are comparable across mixed-unit constraints. ``space="logit"``
        penalises the residual in log-odds (``sd`` is then a log-odds width) —
        used for probability targets so tail beliefs keep multiplicative,
        not absolute, slack. ``gated=False`` keeps the constraint hard even
        in robust mode (used for the anchor estimate)."""
        gate = None
        w = 1.0 / (2.0 * sd * sd)
        if self.robust and gated:
            self.constraints.append(
                ("onoff", f, cond, target, sd, self.p_broken, space))
            gate = self._n_gates
            self._n_gates += 1
        elif cond is None:
            kind = "logit_expect" if space == "logit" else "expect"
            self.constraints.append((kind, f, target, w))
        else:
            kind = "logit_cond_expect" if space == "logit" else "cond_expect"
            self.constraints.append((kind, f, cond, target, w))
        self._report_rows.append(
            (est_id, desc, target, eval_fn, hard_cond, gate, scale))

    def _prob_belief(self) -> tuple[float, str]:
        """(sd, space) for probability targets under the configured penalty."""
        if self.prob_penalty == "logit":
            return self.prob_logit_sd, "logit"
        return self.prob_sd, "abs"

    def _prob_sd(self, est, default_sd: float) -> float:
        """Per-estimate probability belief width, else the global default.

        ``est.sd`` is only honoured under the logit penalty (that is the space
        ``collapse_repeats`` measures the spread in — a log-odds width);
        under the legacy abs penalty its units would be wrong, so fall back."""
        sd = getattr(est, "sd", None)
        if sd is not None and self.prob_penalty == "logit":
            return float(sd)
        return default_sd

    def _value_sd(self, est, idx: int) -> float:
        """Per-estimate expectation belief width (value units), else the
        span-scaled global default."""
        sd = getattr(est, "sd", None)
        if sd is not None:
            return float(sd)
        return self.value_rel_sd * self._span(idx)

    def _build_constraints(self) -> None:
        p_sd, p_space = self._prob_belief()
        for est in self.estimates:
            try:
                if isinstance(est, ProbabilityEstimate):
                    soft, hard = self._proposition_events(est.proposition)
                    tg = self._clip_prob_target(est.id, float(est.probability))
                    is_anchor = (self.anchor_variable is not None and
                                 getattr(est.proposition, "variable", None)
                                 == self.anchor_variable)
                    self._add(soft, None, tg, self._prob_sd(est, p_sd),
                              est.id, est.to_query_estimate(),
                              lambda x, h=hard: float(np.mean(h(x))), None,
                              space=p_space, gated=not is_anchor)

                elif isinstance(est, ExpectationEstimate):
                    f, idx = self._moment_events(est.variable)
                    tg = self._clip_target(est.id, idx, float(est.expected_value))
                    self._add(f, None, tg, self._value_sd(est, idx),
                              est.id, est.to_query_estimate(),
                              lambda x, f=f: float(np.mean(f(x))), None,
                              scale=self._span(idx))

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
                    tg = self._clip_prob_target(est.id, float(est.probability))
                    self._add(soft, soft_c, tg, self._prob_sd(est, p_sd),
                              est.id, est.to_query_estimate(), ev, hard_c,
                              space=p_space)

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
                    self._add(f, soft_c, tg, self._value_sd(est, idx),
                              est.id, est.to_query_estimate(), ev, hard_c,
                              scale=self._span(idx))

                elif isinstance(est, CorrelationEstimate):
                    fa, ia = self._moment_events(est.variable_a)
                    fb, ib = self._moment_events(est.variable_b)
                    tg = float(est.correlation)
                    # scale-free "corr" constraint; the loss correlates raw
                    # site values (binary sites live on [0,1]), the report
                    # uses thresholded binaries for consistency with
                    # sample_dict semantics
                    self.constraints.append(
                        ("corr", fa, fb, tg,
                         1.0 / (2.0 * self.corr_sd * self.corr_sd)))
                    if self.robust:
                        self.warnings.append(
                            f"{est.id}: correlation constraints are not gated "
                            f"in robust mode (no credence)")

                    def ev(x, ia=ia, ib=ib):
                        a = ((x[:, ia] > 0.5).astype(float)
                             if self.is_binary[ia] else x[:, ia])
                        b = ((x[:, ib] > 0.5).astype(float)
                             if self.is_binary[ib] else x[:, ib])
                        return float(np.corrcoef(a, b)[0, 1])
                    self._report_rows.append(
                        (est.id, est.to_query_estimate(), tg, ev, None, None,
                         1.0))

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
        for eid, desc, target, ev, hard_cond, gate, scale in self._report_rows:
            fitted = ev(x)
            rows.append({
                "id": eid,
                "estimate": desc,
                "target": target,
                "fitted": fitted,
                "error": fitted - target,
                # residual in span-normalised units — comparable across
                # probabilities (scale 1) and raw-unit expectations
                "error_rel": (fitted - target) / scale,
                "p_cond": (float(np.mean(hard_cond(x)))
                           if hard_cond is not None else None),
                "credence": (float(creds[gate])
                             if creds is not None and gate is not None
                             else None),
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
