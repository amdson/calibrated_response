"""Validation for the gaussian-KL domain objective (gaussian_kl_objective.md).

Four checks:
1. loss decomposition: gaussian-mode loss == uniform-mode loss - E[log q0]
   at identical params/latents (exact formula, mask excludes binary sites)
2. uniform-mode regression: default kwargs and explicit domain_prior="uniform"
   produce identical fits (the pre-KL objective is a strict special case)
3. prior recovery: no constraints + gaussian mode → continuous marginal
   matches N(mid, span/(2k)); binary sites stay ~uniform
4. the degenerate-range scenario + threshold defaults: E[x]=5 on [0,100]
   keeps mass interior; unconstrained P(x > 90) is the Gaussian tail, not
   the uniform 10%
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from calibrated_response.maxent_sampler import DistributionBuilder
from calibrated_response.maxent_sampler.flow_model import FlowSamplerModel
from calibrated_response.models.query import ExpectationEstimate
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable
from calibrated_response.tn.discretize import ContinuousVar

FIT_KW = dict(steps=800, n_samples=1024, entropy_reg=1.0, seed=0)

VARIABLES = [
    ContinuousVariable(name="x", description="wide-bounds quantity",
                       lower_bound=0.0, upper_bound=100.0, unit="u"),
    BinaryVariable(name="flag", description="unconstrained binary"),
]


def _fit(estimates, **builder_kw):
    b = DistributionBuilder(VARIABLES, estimates, **builder_kw)
    b.fit(**FIT_KW)
    return b


def _p_gt(builder, thr, n=40000, seed=1):
    x = builder.sample(n, seed=seed)
    return float(np.mean(x[:, 0] > thr))


# ---------------------------------------------------------------------------
def test_loss_decomposition_exact():
    """gaussian loss - uniform loss == -E[log q0] on the same batch, with the
    ref_mask zeroing the binary site's contribution."""
    model = FlowSamplerModel(
        [ContinuousVar("x", 0.0, 100.0, 32), ContinuousVar("b", 0.0, 1.0, 2)],
        n_layers=2, hidden=8)
    params = model.init_params(seed=0)
    key = jax.random.PRNGKey(3)

    loss_u = model.constraint_loss([], entropy_reg=1.0, n_samples=512)
    loss_g = model.constraint_loss([], entropy_reg=1.0, n_samples=512,
                                   domain_prior="gaussian",
                                   prior_bound_sds=2.0, ref_mask=[1.0, 0.0])

    z = jax.random.normal(key, (512, model.latent_dim))
    x, _ = model._sample_x_logdet(params, z)
    mu, sd = 50.0, 25.0                      # mid, span / (2 * 2)
    logq0 = float(np.mean(
        -0.5 * np.log(2.0 * np.pi * sd ** 2)
        - (np.asarray(x)[:, 0] - mu) ** 2 / (2.0 * sd ** 2)))

    diff = float(loss_g(params, key)) - float(loss_u(params, key))
    assert diff == pytest.approx(-logq0, abs=1e-3)


def test_bad_domain_prior_rejected():
    with pytest.raises(ValueError, match="domain_prior"):
        DistributionBuilder(VARIABLES, [], domain_prior="cauchy")


# ---------------------------------------------------------------------------
def test_uniform_mode_regression():
    """Default kwargs == explicit domain_prior='uniform': identical fits."""
    ests = [ExpectationEstimate(id="e_x", variable="x", expected_value=25.0)]
    b_default = _fit(ests)
    b_uniform = _fit(ests, domain_prior="uniform")
    s0 = b_default.sample(5000, seed=2)
    s1 = b_uniform.sample(5000, seed=2)
    np.testing.assert_allclose(s0, s1, atol=1e-6)
    assert b_default.kl_to_ref() is None


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gaussian_unconstrained():
    return _fit([], domain_prior="gaussian", prior_bound_sds=2.0)


def test_prior_recovery(gaussian_unconstrained):
    """No constraints, gaussian mode → x ~ N(50, 25) (mildly truncated at
    ±2 sd), binary stays ~uniform, KL(p ‖ q0) near 0."""
    x = gaussian_unconstrained.sample(40000, seed=1)
    mean = float(np.mean(x[:, 0]))
    q10, q90 = np.quantile(x[:, 0], [0.1, 0.9])
    assert mean == pytest.approx(50.0, abs=2.0)          # 0.08 sd
    # N(50, 25) p10/p90 = 50 ∓ 32.04; truncation at ±2 sd pulls them in
    # by ~0.1 sd, so tolerate ±4 units (0.16 sd)
    assert q10 == pytest.approx(50.0 - 32.04, abs=4.0)
    assert q90 == pytest.approx(50.0 + 32.04, abs=4.0)
    p_flag = float(np.mean(x[:, 1] > 0.5))
    assert 0.40 < p_flag < 0.60
    assert gaussian_unconstrained.kl_to_ref() < 0.5


def test_threshold_default_is_gaussian_tail(gaussian_unconstrained):
    """Unconstrained P(x > 90): uniform mode ≈ relative measure (10%);
    gaussian mode ≈ the (truncated) N(50, 25) tail at z = 1.6, ~3-5%."""
    b_uniform = _fit([])
    assert _p_gt(b_uniform, 90.0) > 0.07
    p = _p_gt(gaussian_unconstrained, 90.0)
    assert 0.005 < p < 0.08


# ---------------------------------------------------------------------------
def test_degenerate_range_scenario():
    """The motivating case: E[x] = 5 on [0, 100]. Gaussian mode must keep the
    constraint satisfied without flooding the upper box."""
    ests = [ExpectationEstimate(id="e_x", variable="x", expected_value=5.0)]
    b_uniform = _fit(ests)
    b_gauss = _fit(ests, domain_prior="gaussian", prior_bound_sds=2.0)

    for b in (b_uniform, b_gauss):
        x = b.sample(40000, seed=1)
        # soft constraint (value_rel_sd 0.05 → sd 5): mean lands near 5
        assert 1.0 < float(np.mean(x[:, 0])) < 15.0

    p_u, p_g = _p_gt(b_uniform, 50.0), _p_gt(b_gauss, 50.0)
    assert p_g < 0.05
    assert p_g <= p_u * 1.5 + 0.01           # never worse than uniform mode
