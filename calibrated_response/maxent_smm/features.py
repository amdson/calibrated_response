"""Declarative feature specifications for the MaxEnt energy model.

Each feature type is a frozen dataclass that declaratively describes a function
f_k(x) → scalar.  Features are compiled to pure JAX functions via
``compile_feature`` / ``compile_feature_vector`` so they can be JIT-compiled and
differentiated by JAX for use in HMC.

Smooth sigmoid surrogates are used for all indicator / threshold operations so
that the energy landscape is differentiable everywhere — a requirement for HMC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp



@dataclass(frozen=True)
class MomentFeature:
    """Feature: x[var_idx] ** order.

    Used for mean constraints (order=1) or higher-moment constraints.
    """
    var_idx: int
    order: int = 1

default_sharpness = 60.0

@dataclass(frozen=True)
class SoftThresholdFeature:
    """Smooth surrogate for P(X > threshold) or P(X < threshold).

    Compiles to ``sigmoid(sharpness * (x[var_idx] - threshold))`` when
    ``direction == 'greater'``, or ``sigmoid(sharpness * (threshold - x[var_idx]))``
    when ``direction == 'less'``.
    """
    var_idx: int
    threshold: float
    direction: str = "greater"    # 'greater' or 'less'
    sharpness: float = default_sharpness


@dataclass(frozen=True)
class SoftIndicatorFeature:
    """Smooth surrogate for P(lower <= X <= upper).

    Product of two sigmoids forming a soft window.
    """
    var_idx: int
    lower: float
    upper: float
    sharpness: float = default_sharpness


@dataclass(frozen=True)
class ProductMomentFeature:
    """Product of first moments across several variables: prod_i x[var_indices[i]].

    Used for cross-moment constraints and conditional decompositions.
    """
    var_indices: tuple[int, ...]


@dataclass(frozen=True)
class ConditionalSoftThresholdFeature:
    """Product of a target soft-threshold and a condition soft-threshold.

    Encodes e.g. P(X > t | Y > s) · P(Y > s) via the product
    ``sigma(target) * sigma(condition)``, whose expectation equals
    E[I(X > t) · I(Y > s)].
    """
    target_var: int
    target_threshold: float
    target_direction: str          # 'greater' or 'less'
    cond_var: int
    cond_threshold: float
    cond_direction: str            # 'greater' or 'less'
    sharpness: float = default_sharpness


@dataclass(frozen=True)
class WeightedMomentConditionFeature:
    """Feature: x[target_var] * sigma(sharpness * (x[cond_var] - cond_threshold)).

    Expected value equals E[X · I(cond)] under the smooth surrogate,
    used with the identity  E[X|C] · P(C) = E[X · I(C)]  to encode
    conditional expectation constraints.
    """
    target_var: int
    cond_var: int
    cond_threshold: float
    cond_direction: str = "greater"
    sharpness: float = default_sharpness


@dataclass(frozen=True)
class CenteredConditionalFeature:
    """Feature: σ_cond(x) * (σ_target(x) - p)  with target expectation 0.

    Encodes P(target | cond) = p exactly without needing to know P(cond):

        E[σ_cond · (σ_target − p)] = 0  ⟺  P(target | cond) = p

    Unlike the joint-indicator approach (which requires P(cond) ≈ 0.5), this
    feature's target is always 0, making it valid regardless of the condition
    marginal probability.
    """
    target_var: int
    target_threshold: float
    target_direction: str          # 'greater' or 'less'
    cond_var: int
    cond_threshold: float
    cond_direction: str            # 'greater' or 'less'
    p_target_given_cond: float     # the conditional probability p
    sharpness: float = default_sharpness


@dataclass(frozen=True)
class CenteredConditionalMomentFeature:
    """Feature: σ_cond(x) * (x[target_var] - μ)  with target expectation 0.

    Encodes E[X | cond] = μ exactly without needing to know P(cond):

        E[σ_cond · (x − μ)] = 0  ⟺  E[X | cond] = μ

    Unlike the weighted-moment approach (which requires P(cond) ≈ 0.5), this
    feature's target is always 0, making it valid regardless of the condition
    marginal probability.
    """
    target_var: int
    cond_var: int
    cond_threshold: float
    cond_direction: str = "greater"
    expected_value: float = 0.0    # the conditional expectation μ (normalised domain)
    sharpness: float = default_sharpness


# Union of all feature types
FeatureSpec = Union[
    MomentFeature,
    SoftThresholdFeature,
    SoftIndicatorFeature,
    ProductMomentFeature,
    ConditionalSoftThresholdFeature,
    WeightedMomentConditionFeature,
    CenteredConditionalFeature,
    CenteredConditionalMomentFeature,
]


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def _soft_threshold(x_val: jnp.ndarray, threshold: float, direction: str, sharpness: float) -> jnp.ndarray:
    """Smooth sigmoid surrogate for an indicator threshold."""
    if direction == "greater":
        return jax.nn.sigmoid(sharpness * (x_val - threshold))
    else:
        return jax.nn.sigmoid(sharpness * (threshold - x_val))

def compile_feature(spec: FeatureSpec) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compile a single feature spec into a pure JAX function ``f(x) → scalar``.

    The returned callable is safe to JIT / grad / vmap.
    """
    if isinstance(spec, MomentFeature):
        idx, order = spec.var_idx, spec.order
        def _moment(x: jnp.ndarray) -> jnp.ndarray:
            return x[idx] ** order
        return _moment

    if isinstance(spec, SoftThresholdFeature):
        idx = spec.var_idx
        t, d, s = spec.threshold, spec.direction, spec.sharpness
        def _thresh(x: jnp.ndarray) -> jnp.ndarray:
            return _soft_threshold(x[idx], t, d, s)
        return _thresh

    if isinstance(spec, SoftIndicatorFeature):
        idx = spec.var_idx
        lo, hi, s = spec.lower, spec.upper, spec.sharpness
        def _indicator(x: jnp.ndarray) -> jnp.ndarray:
            return jax.nn.sigmoid(s * (x[idx] - lo)) * jax.nn.sigmoid(s * (hi - x[idx]))
        return _indicator

    if isinstance(spec, ProductMomentFeature):
        idxs = spec.var_indices
        def _product(x: jnp.ndarray) -> jnp.ndarray:
            val = jnp.ones(())
            for i in idxs:
                val = val * x[i]
            return val
        return _product

    if isinstance(spec, ConditionalSoftThresholdFeature):
        tv, tt, td = spec.target_var, spec.target_threshold, spec.target_direction
        cv, ct, cd = spec.cond_var, spec.cond_threshold, spec.cond_direction
        s = spec.sharpness
        def _cond_thresh(x: jnp.ndarray) -> jnp.ndarray:
            target_val = _soft_threshold(x[tv], tt, td, s)
            cond_val = _soft_threshold(x[cv], ct, cd, s)
            return target_val * cond_val
        return _cond_thresh

    if isinstance(spec, WeightedMomentConditionFeature):
        tv, cv = spec.target_var, spec.cond_var
        ct, cd, s = spec.cond_threshold, spec.cond_direction, spec.sharpness
        def _weighted(x: jnp.ndarray) -> jnp.ndarray:
            return x[tv] * _soft_threshold(x[cv], ct, cd, s)
        return _weighted

    if isinstance(spec, CenteredConditionalFeature):
        tv, tt, td = spec.target_var, spec.target_threshold, spec.target_direction
        cv, ct, cd = spec.cond_var, spec.cond_threshold, spec.cond_direction
        p, s = spec.p_target_given_cond, spec.sharpness
        def _centered_cond(x: jnp.ndarray) -> jnp.ndarray:
            return _soft_threshold(x[cv], ct, cd, s) * (
                _soft_threshold(x[tv], tt, td, s) - p
            )
        return _centered_cond

    if isinstance(spec, CenteredConditionalMomentFeature):
        tv, cv = spec.target_var, spec.cond_var
        ct, cd, mu, s = spec.cond_threshold, spec.cond_direction, spec.expected_value, spec.sharpness
        def _centered_moment(x: jnp.ndarray) -> jnp.ndarray:
            return _soft_threshold(x[cv], ct, cd, s) * (x[tv] - mu)
        return _centered_moment

    raise TypeError(f"Unknown feature spec type: {type(spec)}")


def compile_feature_vector(specs: Sequence[FeatureSpec]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compile a list of feature specs into a single JIT-compiled function
    ``f(x) → jnp.ndarray`` of shape ``(K,)`` where ``K = len(specs)``.

    Each element of the output vector corresponds to one feature evaluated at
    state ``x``.
    """
    fns = [compile_feature(s) for s in specs]
    K = len(fns)

    @jax.jit
    def _feature_vector(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([fn(x) for fn in fns])

    return _feature_vector
