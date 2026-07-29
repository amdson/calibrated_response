"""Per-estimate-type constraint builders ("sample losses") for the flow builder.

Each :class:`SampleLoss` maps one ``EstimateUnion`` subtype to a function that
turns an estimate of that type into the flow sampler's constraint grammar —
appending ``("expect", ...)`` / ``("cond_expect", ...)`` / ``("corr", ...)`` /
``("onoff", ...)`` tuples to ``builder.constraints`` and the matching evaluation
row to ``builder._report_rows``, both via the builder's private helpers
(``_add``, ``_proposition_events``, ``_moment_events``, ``_clip_target``, ...).

``DistributionBuilder._build_constraints`` dispatches over :data:`SAMPLE_LOSSES`
by ``isinstance`` (first match wins, mirroring the original ``if/elif`` chain).
Adding a new estimate type is then a matter of writing one ``build`` function and
appending one :class:`SampleLoss` entry — no edits to the builder's control flow.

The low-level event helpers (:data:`_EPS`, :func:`_and_all`, :func:`_hard_event`)
live here too so the builder and the loss functions share one definition;
``distribution_builder`` imports them back from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from calibrated_response.models.query import (
    ConditionalExpectationEstimate,
    ConditionalProbabilityEstimate,
    CorrelationEstimate,
    EquationEstimate,
    EstimateUnion,
    ExpectationEstimate,
    ProbabilityEstimate,
)
from calibrated_response.maxent_sampler.equations import compile_residual

if TYPE_CHECKING:  # avoid an import cycle — builder imports this module
    from calibrated_response.maxent_sampler.distribution_builder import (
        DistributionBuilder,
    )

_EPS = 1e-30


# ----------------------------------------------------------------------------
# event helpers (shared with the builder's ``_proposition_events``)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# one build function per estimate type
# ----------------------------------------------------------------------------
def _probability_loss(builder: "DistributionBuilder", est: ProbabilityEstimate) -> None:
    soft, hard = builder._proposition_events(est.proposition)
    tg = builder._clip_prob_target(est.id, float(est.probability))
    is_anchor = (builder.anchor_variable is not None and
                 getattr(est.proposition, "variable", None)
                 == builder.anchor_variable)
    p_sd, p_space = builder._prob_belief()
    builder._add(soft, None, tg, builder._prob_sd(est, p_sd),
                 est.id, est.to_query_estimate(),
                 lambda x, h=hard: float(np.mean(h(x))), None,
                 space=p_space, gated=not is_anchor, direction=est.relation)


def _expectation_loss(builder: "DistributionBuilder", est: ExpectationEstimate) -> None:
    f, idx = builder._moment_events(est.variable)
    tg = builder._clip_target(est.id, idx, float(est.expected_value))
    builder._add(f, None, tg, builder._value_sd(est, idx),
                 est.id, est.to_query_estimate(),
                 lambda x, f=f: float(np.mean(f(x))), None,
                 scale=builder._span(idx), direction=est.relation)


def _conditional_probability_loss(
    builder: "DistributionBuilder", est: ConditionalProbabilityEstimate
) -> None:
    if not est.conditions:
        raise ValueError("conditional estimate with no conditions")
    soft, hard = builder._proposition_events(est.proposition)
    pairs = [builder._proposition_events(c) for c in est.conditions]
    soft_c = _and_all([s for s, _ in pairs])
    hard_c = _and_all([h for _, h in pairs])

    def ev(x, h=hard, hc=hard_c):
        c = hc(x)
        return float(np.sum(h(x) * c) / (np.sum(c) + _EPS))

    tg = builder._clip_prob_target(est.id, float(est.probability))
    p_sd, p_space = builder._prob_belief()
    builder._add(soft, soft_c, tg, builder._prob_sd(est, p_sd),
                 est.id, est.to_query_estimate(), ev, hard_c, space=p_space,
                 direction=est.relation)


def _conditional_expectation_loss(
    builder: "DistributionBuilder", est: ConditionalExpectationEstimate
) -> None:
    if not est.conditions:
        raise ValueError("conditional estimate with no conditions")
    f, idx = builder._moment_events(est.variable)
    tg = builder._clip_target(est.id, idx, float(est.expected_value))
    pairs = [builder._proposition_events(c) for c in est.conditions]
    soft_c = _and_all([s for s, _ in pairs])
    hard_c = _and_all([h for _, h in pairs])

    def ev(x, f=f, hc=hard_c):
        c = hc(x)
        return float(np.sum(f(x) * c) / (np.sum(c) + _EPS))

    builder._add(f, soft_c, tg, builder._value_sd(est, idx),
                 est.id, est.to_query_estimate(), ev, hard_c,
                 scale=builder._span(idx), direction=est.relation)


def _correlation_loss(builder: "DistributionBuilder", est: CorrelationEstimate) -> None:
    fa, ia = builder._moment_events(est.variable_a)
    fb, ib = builder._moment_events(est.variable_b)
    tg = float(est.correlation)
    w = 1.0 / (2.0 * builder.corr_sd * builder.corr_sd)
    # scale-free "corr" constraint; the loss correlates raw site values
    # (binary sites live on [0,1]), the report uses thresholded binaries for
    # consistency with sample_dict semantics. A one-sided bound (relation
    # "le"/"ge") routes to the hinge variant "corr_ineq".
    if est.relation == "eq":
        builder.constraints.append(("corr", fa, fb, tg, w))
    else:
        builder.constraints.append(("corr_ineq", fa, fb, tg, w, est.relation))
    if builder.robust:
        builder.warnings.append(
            f"{est.id}: correlation constraints are not gated "
            f"in robust mode (no credence)")

    def ev(x, ia=ia, ib=ib):
        a = ((x[:, ia] > 0.5).astype(float)
             if builder.is_binary[ia] else x[:, ia])
        b = ((x[:, ib] > 0.5).astype(float)
             if builder.is_binary[ib] else x[:, ib])
        return float(np.corrcoef(a, b)[0, 1])

    builder._report_rows.append(
        (est.id, est.to_query_estimate(), tg, ev, None, None, 1.0))


def _equation_loss(builder: "DistributionBuilder", est: EquationEstimate) -> None:
    if est.lhs not in builder.var_name_to_idx:
        raise ValueError(f"unknown lhs variable {est.lhs!r}")
    span = {name: builder._span(i)
            for name, i in builder.var_name_to_idx.items()}
    # residual r(x) = lhs - (rhs); soft (jax, for the loss) and hard (numpy, for
    # the report). Compilation raises on unknown vars / bad syntax -> skipped.
    soft_r, hard_r = compile_residual(
        est.lhs, est.rhs, builder.var_name_to_idx, span, builder.sharpness)
    if builder.robust:
        builder.warnings.append(
            f"{est.id}: equation constraints are not gated in robust mode "
            f"(no credence)")

    if not est.noise_sd:                       # deterministic identity
        tol = builder.eqn_rel_sd * builder._span(builder.var_name_to_idx[est.lhs])
        w = 1.0 / (2.0 * tol * tol)
        builder.constraints.append(("eqn_det", soft_r, w))
        target = 0.0
    else:                                       # lhs = rhs + N(0, noise_sd)
        sigma = float(est.noise_sd)
        builder.constraints.append(("eqn_dist", soft_r, sigma, builder.eqn_conf))
        target = sigma

    # report the residual RMS against its intended value (0, or the noise sd)
    def ev(x, hr=hard_r):
        r = hr(x)
        return float(np.sqrt(np.mean(r * r)))

    builder._report_rows.append(
        (est.id, est.to_query_estimate(), target, ev, None, None, 1.0))


# ----------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleLoss:
    """One estimate type and the function that maps it to flow constraints.

    ``build(builder, est)`` mutates ``builder`` in place (appending constraints
    and a report row); it may raise ``ValueError`` when the estimate is
    unusable, which the builder catches and records in ``builder.skipped``.
    """

    estimate_type: type
    build: Callable[["DistributionBuilder", EstimateUnion], None]


SAMPLE_LOSSES: list[SampleLoss] = [
    SampleLoss(ProbabilityEstimate, _probability_loss),
    SampleLoss(ExpectationEstimate, _expectation_loss),
    SampleLoss(ConditionalProbabilityEstimate, _conditional_probability_loss),
    SampleLoss(ConditionalExpectationEstimate, _conditional_expectation_loss),
    SampleLoss(CorrelationEstimate, _correlation_loss),
    SampleLoss(EquationEstimate, _equation_loss),
]
