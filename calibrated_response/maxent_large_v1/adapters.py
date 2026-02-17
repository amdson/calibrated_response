from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

from calibrated_response.maxent_large_v1.graph import Factor, FactorGraph, SMMVariable, build_graph, from_variables
from calibrated_response.maxent_large_v1.targets import TargetMoments
from calibrated_response.models.query import (
    ConditionalExpectationEstimate,
    ConditionalProbabilityEstimate,
    EqualityProposition,
    EstimateUnion,
    ExpectationEstimate,
    InequalityProposition,
    ProbabilityEstimate,
)
from calibrated_response.models.variable import Variable


@dataclass(frozen=True)
class AdaptedProblem:
    graph: FactorGraph
    targets: TargetMoments
    bin_edges_list: list[np.ndarray]
    var_name_to_idx: dict[str, int]
    skipped_constraints: list[str]


def _centers(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def _mask_from_proposition(prop: EqualityProposition | InequalityProposition, edges: np.ndarray) -> np.ndarray:
    centers = _centers(edges)
    if isinstance(prop, InequalityProposition):
        if prop.is_lower_bound:
            return centers > float(prop.threshold)
        return centers < float(prop.threshold)

    if isinstance(prop, EqualityProposition):
        if isinstance(prop.value, bool):
            return centers >= 0.5 if prop.value else centers < 0.5
        return np.zeros_like(centers, dtype=bool)

    return np.zeros_like(centers, dtype=bool)


def _distribution_from_probability(mask: np.ndarray, p: float) -> np.ndarray:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    n = mask.size
    q = np.zeros((n,), dtype=float)

    in_count = int(mask.sum())
    out_count = n - in_count
    if in_count == 0 or out_count == 0:
        q[:] = 1.0 / n
        return q

    q[mask] = p / in_count
    q[~mask] = (1.0 - p) / out_count
    q /= q.sum()
    return q


#Returns Gaussian approximated by bin distribution
def _distribution_from_expectation(edges: np.ndarray, target_mean: float) -> np.ndarray:
    c = _centers(edges)
    scale = max((edges[-1] - edges[0]) / 8.0, 1e-3)
    logits = -0.5 * ((c - float(target_mean)) / scale) ** 2
    logits -= np.max(logits)
    q = np.exp(logits)
    total = q.sum()
    return q / total if total > 0 else np.ones_like(q) / q.size

def _pair_table_from_conditional_probability(
    target_mask: np.ndarray,
    cond_mask: np.ndarray,
    probability: float,
) -> Optional[np.ndarray]:
    ka = int(target_mask.size)
    kb = int(cond_mask.size)
    if ka == 0 or kb == 0:
        return None

    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    table = np.zeros((ka, kb), dtype=float)

    n_target_true = int(target_mask.sum())
    n_target_false = ka - n_target_true
    if n_target_true == 0 or n_target_false == 0:
        return None

    for b in range(kb):
        mass_b = 1.0 / kb
        if cond_mask[b]:
            col = np.zeros((ka,), dtype=float)
            col[target_mask] = p / n_target_true
            col[~target_mask] = (1.0 - p) / n_target_false
            table[:, b] = mass_b * col
        else:
            table[:, b] = mass_b * (1.0 / ka)

    total = table.sum()
    if total <= 0:
        return None
    return table / total


def _higher_order_table_from_conditional_probability(
    target_mask: np.ndarray,
    condition_masks: list[np.ndarray],
    condition_sizes: list[int],
    probability: float,
) -> Optional[np.ndarray]:
    if not condition_sizes:
        return None

    ka = int(target_mask.size)
    if ka == 0:
        return None

    n_target_true = int(target_mask.sum())
    n_target_false = ka - n_target_true
    if n_target_true == 0 or n_target_false == 0:
        return None

    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))

    target_cond_true = np.zeros((ka,), dtype=float)
    target_cond_true[target_mask] = p / n_target_true
    target_cond_true[~target_mask] = (1.0 - p) / n_target_false
    target_uniform = np.ones((ka,), dtype=float) / ka

    shape = (ka, *condition_sizes)
    table = np.zeros(shape, dtype=float)

    cond_total = int(np.prod(condition_sizes))
    if cond_total <= 0:
        return None
    mass_per_condition_assignment = 1.0 / cond_total

    for cond_assignment in np.ndindex(*condition_sizes):
        cond_true = all(mask[idx] for mask, idx in zip(condition_masks, cond_assignment))
        target_dist = target_cond_true if cond_true else target_uniform
        table[(slice(None), *cond_assignment)] = mass_per_condition_assignment * target_dist

    total = table.sum()
    if total <= 0:
        return None
    return table / total

def _higher_order_table_from_conditional_expectation(
    target_edges: np.ndarray,
    condition_masks: list[np.ndarray],
    condition_sizes: list[int],
    expected_value: float,
) -> Optional[np.ndarray]:
    if not condition_sizes:
        return None

    ka = int(target_edges.size - 1)
    if ka == 0:
        return None

    # Use a smooth proxy marginal whose mean is near the requested conditional expectation.
    target_cond_true = _distribution_from_expectation(target_edges, expected_value)
    if target_cond_true.size != ka:
        return None
    cond_true_total = float(target_cond_true.sum())
    if cond_true_total <= 0.0:
        return None
    target_cond_true = target_cond_true / cond_true_total
    target_uniform = np.ones((ka,), dtype=float) / ka

    shape = (ka, *condition_sizes)
    table = np.zeros(shape, dtype=float)

    cond_total = int(np.prod(condition_sizes))
    if cond_total <= 0:
        return None
    mass_per_condition_assignment = 1.0 / cond_total

    for cond_assignment in np.ndindex(*condition_sizes):
        cond_true = all(mask[idx] for mask, idx in zip(condition_masks, cond_assignment))
        target_dist = target_cond_true if cond_true else target_uniform
        table[(slice(None), *cond_assignment)] = mass_per_condition_assignment * target_dist
        
    total = table.sum()
    if total <= 0:
        return None
    return table / total

def adapt_problem(
    variables: Sequence[Variable],
    estimates: Sequence[EstimateUnion],
    max_bins: int,
) -> AdaptedProblem:
    smm_vars, bin_edges_list = from_variables(variables, max_bins=max_bins)
    var_name_to_idx = {v.name: i for i, v in enumerate(variables)}

    factors: list[Factor] = []
    unary_factor_by_var: dict[int, int] = {}
    next_factor_id = 0
    for var_idx, smm_var in enumerate(smm_vars):
        shape = (smm_var.num_states,)
        factors.append(
            Factor(id=next_factor_id, var_ids=(var_idx,), theta=jnp.zeros(shape, dtype=jnp.float32), table_shape=shape)
        )
        unary_factor_by_var[var_idx] = next_factor_id
        next_factor_id += 1

    higher_factor_by_vars: dict[tuple[int, ...], int] = {}
    skipped: list[str] = []

    def ensure_higher_factor(var_ids: tuple[int, ...]) -> int:
        nonlocal next_factor_id
        key = tuple(var_ids)
        if key in higher_factor_by_vars:
            return higher_factor_by_vars[key]

        shape = tuple(smm_vars[i].num_states for i in key)
        factors.append(
            Factor(id=next_factor_id, var_ids=key, theta=jnp.zeros(shape, dtype=jnp.float32), table_shape=shape)
        )
        higher_factor_by_vars[key] = next_factor_id
        next_factor_id += 1
        return higher_factor_by_vars[key]

    accum: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    def resolve_conditional_context(
        *,
        estimate_id: str,
        target_variable: str,
        conditions: Sequence[EqualityProposition | InequalityProposition],
        unknown_message: str,
        no_conditions_message: str,
    ) -> Optional[tuple[int, list[int], list[EqualityProposition | InequalityProposition]]]:
        if len(conditions) == 0:
            skipped.append(f"{estimate_id}: {no_conditions_message}")
            return None

        # v1 supports factors up to order 3 => target + at most 2 conditions.
        if len(conditions) > 2:
            skipped.append(
                f"{estimate_id}: at most two conditions are supported (target+2 => 3-variable factor)"
            )
            return None

        condition_vars = [c.variable for c in conditions]
        if target_variable not in var_name_to_idx or any(v not in var_name_to_idx for v in condition_vars):
            skipped.append(f"{estimate_id}: {unknown_message}")
            return None

        target_idx_local = var_name_to_idx[target_variable]
        cond_indices_raw = [var_name_to_idx[v] for v in condition_vars]

        if target_idx_local in cond_indices_raw:
            skipped.append(f"{estimate_id}: self-conditionals are not supported")
            return None

        if len(set(cond_indices_raw)) != len(cond_indices_raw):
            skipped.append(f"{estimate_id}: duplicate condition variables are not supported")
            return None

        # Canonicalize condition variable order so semantically equivalent estimates share factors.
        sorted_pairs = sorted(zip(cond_indices_raw, conditions), key=lambda t: t[0])
        cond_indices_local = [idx for idx, _ in sorted_pairs]
        cond_props_local = [cond for _, cond in sorted_pairs]
        return target_idx_local, cond_indices_local, cond_props_local

    for estimate in estimates:
        if isinstance(estimate, ProbabilityEstimate):
            prop = estimate.proposition
            if prop.variable not in var_name_to_idx:
                skipped.append(f"{estimate.id}: unknown variable {prop.variable}")
                continue
            var_idx = var_name_to_idx[prop.variable]
            mask = _mask_from_proposition(prop, bin_edges_list[var_idx])
            if not np.any(mask):
                skipped.append(f"{estimate.id}: proposition mask is empty")
                continue
            target = _distribution_from_probability(mask, estimate.probability)
            fid = unary_factor_by_var[var_idx]
            accum[fid] = accum.get(fid, 0.0) + target
            counts[fid] = counts.get(fid, 0) + 1
            continue

        if isinstance(estimate, ExpectationEstimate):
            if estimate.variable not in var_name_to_idx:
                skipped.append(f"{estimate.id}: unknown variable {estimate.variable}")
                continue
            var_idx = var_name_to_idx[estimate.variable]
            target = _distribution_from_expectation(bin_edges_list[var_idx], estimate.expected_value)
            fid = unary_factor_by_var[var_idx]
            accum[fid] = accum.get(fid, 0.0) + target
            counts[fid] = counts.get(fid, 0) + 1
            continue

        if isinstance(estimate, ConditionalProbabilityEstimate):
            context = resolve_conditional_context(
                estimate_id=estimate.id,
                target_variable=estimate.proposition.variable,
                conditions=estimate.conditions,
                unknown_message="unknown variables in conditional estimate",
                no_conditions_message="conditional probability has no conditions",
            )
            if context is None:
                continue

            target_idx, cond_indices, cond_props = context

            target_mask = _mask_from_proposition(estimate.proposition, bin_edges_list[target_idx])
            cond_masks = [_mask_from_proposition(cond, bin_edges_list[idx]) for cond, idx in zip(cond_props, cond_indices)]
            if not np.any(target_mask) or any(not np.any(mask) for mask in cond_masks):
                skipped.append(f"{estimate.id}: empty target/condition mask")
                continue

            cond_sizes = [smm_vars[i].num_states for i in cond_indices]
            table = _higher_order_table_from_conditional_probability(
                target_mask=target_mask,
                condition_masks=cond_masks,
                condition_sizes=cond_sizes,
                probability=estimate.probability,
            )
            if table is None:
                skipped.append(f"{estimate.id}: could not build conditional target table")
                continue

            factor_var_ids = tuple([target_idx, *cond_indices])
            fid = ensure_higher_factor(factor_var_ids)
            accum[fid] = accum.get(fid, 0.0) + table
            counts[fid] = counts.get(fid, 0) + 1
            continue

        if isinstance(estimate, ConditionalExpectationEstimate):
            context = resolve_conditional_context(
                estimate_id=estimate.id,
                target_variable=estimate.variable,
                conditions=estimate.conditions,
                unknown_message="unknown variables in conditional expectation",
                no_conditions_message="conditional expectation has no conditions",
            )
            if context is None:
                continue

            target_idx, cond_indices, cond_props = context

            cond_masks = [_mask_from_proposition(cond, bin_edges_list[idx]) for cond, idx in zip(cond_props, cond_indices)]
            if any(not np.any(mask) for mask in cond_masks):
                skipped.append(f"{estimate.id}: empty condition mask")
                continue

            cond_sizes = [smm_vars[i].num_states for i in cond_indices]
            table = _higher_order_table_from_conditional_expectation(
                target_edges=bin_edges_list[target_idx],
                condition_masks=cond_masks,
                condition_sizes=cond_sizes,
                expected_value=estimate.expected_value,
            )
            if table is None:
                skipped.append(f"{estimate.id}: could not build conditional expectation table")
                continue

            factor_var_ids = tuple([target_idx, *cond_indices])
            fid = ensure_higher_factor(factor_var_ids)
            accum[fid] = accum.get(fid, 0.0) + table
            counts[fid] = counts.get(fid, 0) + 1
            continue

        skipped.append(f"{getattr(estimate, 'id', 'unknown')}: unsupported estimate type")

    graph = build_graph(smm_vars, factors)

    targets: dict[int, jnp.ndarray] = {}
    for factor_id, table_sum in accum.items():
        mean_table = table_sum / max(counts.get(factor_id, 1), 1)
        mean_table = np.asarray(mean_table, dtype=np.float32)
        total = mean_table.sum()
        if total > 0:
            mean_table = mean_table / total
        targets[factor_id] = jnp.asarray(mean_table)

    return AdaptedProblem(
        graph=graph,
        targets=TargetMoments(by_factor_id=targets),
        bin_edges_list=bin_edges_list,
        var_name_to_idx=var_name_to_idx,
        skipped_constraints=skipped,
    )
