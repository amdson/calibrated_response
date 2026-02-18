"""Tests for calibrated_response.maxent_large.features."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrated_response.maxent_large.features import (
    MomentFeature,
    SoftThresholdFeature,
    SoftIndicatorFeature,
    ProductMomentFeature,
    ConditionalSoftThresholdFeature,
    WeightedMomentConditionFeature,
    compile_feature,
    compile_feature_vector,
)


# ---- compile_feature: individual feature types ----------------------------

class TestMomentFeature:
    def test_identity(self):
        fn = compile_feature(MomentFeature(var_idx=0, order=1))
        x = jnp.array([0.7, 0.3])
        assert float(fn(x)) == pytest.approx(0.7, abs=1e-6)

    def test_square(self):
        fn = compile_feature(MomentFeature(var_idx=1, order=2))
        x = jnp.array([0.5, 0.4])
        assert float(fn(x)) == pytest.approx(0.16, abs=1e-6)


class TestSoftThresholdFeature:
    def test_greater_above(self):
        fn = compile_feature(SoftThresholdFeature(var_idx=0, threshold=0.3, direction="greater", sharpness=50.0))
        assert float(fn(jnp.array([0.8]))) > 0.99

    def test_greater_below(self):
        fn = compile_feature(SoftThresholdFeature(var_idx=0, threshold=0.3, direction="greater", sharpness=50.0))
        assert float(fn(jnp.array([0.1]))) < 0.01

    def test_less_above(self):
        fn = compile_feature(SoftThresholdFeature(var_idx=0, threshold=0.5, direction="less", sharpness=50.0))
        assert float(fn(jnp.array([0.9]))) < 0.01

    def test_gradient_is_nonzero(self):
        """Smooth sigmoid must have non-zero gradient (essential for HMC)."""
        fn = compile_feature(SoftThresholdFeature(var_idx=0, threshold=0.5, direction="greater"))
        grad_fn = jax.grad(lambda x: fn(x).squeeze())
        g = grad_fn(jnp.array([0.5]))
        assert float(jnp.abs(g).max()) > 0.1


class TestSoftIndicatorFeature:
    def test_inside(self):
        fn = compile_feature(SoftIndicatorFeature(var_idx=0, lower=0.2, upper=0.8, sharpness=50.0))
        assert float(fn(jnp.array([0.5]))) > 0.95

    def test_outside(self):
        fn = compile_feature(SoftIndicatorFeature(var_idx=0, lower=0.2, upper=0.8, sharpness=50.0))
        assert float(fn(jnp.array([0.05]))) < 0.05

    def test_gradient_nonzero_at_edge(self):
        fn = compile_feature(SoftIndicatorFeature(var_idx=0, lower=0.3, upper=0.7))
        grad_fn = jax.grad(lambda x: fn(x).squeeze())
        g = grad_fn(jnp.array([0.3]))
        assert float(jnp.abs(g).max()) > 0.01


class TestProductMomentFeature:
    def test_two_vars(self):
        fn = compile_feature(ProductMomentFeature(var_indices=(0, 1)))
        x = jnp.array([0.4, 0.5])
        assert float(fn(x)) == pytest.approx(0.2, abs=1e-6)


class TestConditionalSoftThresholdFeature:
    def test_both_satisfied(self):
        fn = compile_feature(
            ConditionalSoftThresholdFeature(
                target_var=0, target_threshold=0.3, target_direction="greater",
                cond_var=1, cond_threshold=0.3, cond_direction="greater",
                sharpness=50.0,
            )
        )
        x = jnp.array([0.8, 0.8])
        assert float(fn(x)) > 0.95

    def test_cond_not_satisfied(self):
        fn = compile_feature(
            ConditionalSoftThresholdFeature(
                target_var=0, target_threshold=0.3, target_direction="greater",
                cond_var=1, cond_threshold=0.7, cond_direction="greater",
                sharpness=50.0,
            )
        )
        x = jnp.array([0.9, 0.1])
        assert float(fn(x)) < 0.05


class TestWeightedMomentConditionFeature:
    def test_basic(self):
        fn = compile_feature(
            WeightedMomentConditionFeature(target_var=0, cond_var=1, cond_threshold=0.3, cond_direction="greater", sharpness=50.0)
        )
        x = jnp.array([0.6, 0.9])
        # ≈ 0.6 * 1.0
        assert float(fn(x)) == pytest.approx(0.6, abs=0.05)


# ---- compile_feature_vector -----------------------------------------------

class TestCompileFeatureVector:
    def test_shape(self):
        specs = [
            MomentFeature(var_idx=0),
            SoftThresholdFeature(var_idx=1, threshold=0.5, direction="greater"),
            ProductMomentFeature(var_indices=(0, 1)),
        ]
        fv = compile_feature_vector(specs)
        x = jnp.array([0.6, 0.7])
        result = fv(x)
        assert result.shape == (3,)

    def test_values(self):
        specs = [
            MomentFeature(var_idx=0, order=1),
            MomentFeature(var_idx=1, order=1),
        ]
        fv = compile_feature_vector(specs)
        x = jnp.array([0.3, 0.9])
        result = fv(x)
        np.testing.assert_allclose(result, jnp.array([0.3, 0.9]), atol=1e-6)

    def test_jit_compatible(self):
        specs = [MomentFeature(var_idx=0)]
        fv = compile_feature_vector(specs)
        # Should already be JIT'd, but calling jit again should be fine
        fv_jit = jax.jit(fv)
        result = fv_jit(jnp.array([0.42]))
        assert float(result[0]) == pytest.approx(0.42, abs=1e-5)
