"""Validation for dummy flow dimensions (``n_dummy``).

Dummies are a pure expressiveness knob: extra flow inputs/outputs carrying no
variable.  Checks:
1. shapes: flow width is n + k, but every readout returns real sites only
2. objective invariance in expectation: at the identity init the extended
   entropy equals the no-dummy entropy exactly (dummies are independent
   Uniform(0,1) there — 0 net contribution)
3. gaussian mode: dummies keep the uniform reference, so the loss
   decomposition against real sites is unchanged
4. log_prob is refused with dummies (real-site density is an intractable
   marginal)
5. builder integration: fit with dummies satisfies constraints and readouts
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

CVARS = [ContinuousVar("x", 0.0, 100.0, 32), ContinuousVar("b", 0.0, 1.0, 2)]


def test_shapes_and_readouts():
    model = FlowSamplerModel(CVARS, n_layers=2, hidden=8, n_dummy=3)
    assert model.n == 2 and model.n_flow == 5 and model.latent_dim == 5
    params = model.init_params(seed=0)

    x = model.sample(params, 100, seed=0)
    assert x.shape == (100, 2)                       # dummies sliced off
    assert np.all((x[:, 0] >= 0.0) & (x[:, 0] <= 100.0))

    xf, ld = model._sample_x_logdet(params, model._draw_z(100, 0))
    assert xf.shape == (100, 5)                      # loss sees full width
    assert np.all((np.asarray(xf)[:, 2:] >= 0.0)
                  & (np.asarray(xf)[:, 2:] <= 1.0))  # dummies on unit box

    assert np.isfinite(model.entropy(params, n_samples=2000))
    assert 0.0 <= model.prob_gt(params, 0, 50.0, n_samples=2000) <= 1.0


def test_identity_init_entropy_analytic():
    """At the zero (identity) init the flow is u = sigmoid(z) elementwise, so
    each dummy is an independent sigmoid(N(0,1)) with unit span: it must add
    exactly H(sigmoid(Z)) = 0.5·log(2πe) + E[log σ'(Z)] nats to the joint —
    i.e. dummies enter the entropy machinery as plain unit-box dimensions
    (log span 0), nothing more.  (Zero *net* contribution is a property of
    the maxent optimum, where dummies go uniform, not of the init.)"""
    h0 = FlowSamplerModel(CVARS, n_layers=2, hidden=8)
    hk = FlowSamplerModel(CVARS, n_layers=2, hidden=8, n_dummy=4)
    p0 = {"theta": h0.net.pack_params(h0.net.zero_params())}
    pk = {"theta": hk.net.pack_params(hk.net.zero_params())}

    rng = np.random.default_rng(0)
    z = rng.standard_normal(200000)
    sig = 1.0 / (1.0 + np.exp(-z))
    h_sig = 0.5 * np.log(2.0 * np.pi * np.e) + float(
        np.mean(np.log(sig) + np.log1p(-sig)))

    diff = hk.entropy(pk, n_samples=50000) - h0.entropy(p0, n_samples=50000)
    assert diff == pytest.approx(4.0 * h_sig, abs=0.03)


def test_gaussian_decomposition_ignores_dummies():
    """gaussian loss - uniform loss == -E[log q0] over REAL masked sites only,
    on the same batch — dummies contribute nothing to the reference term."""
    model = FlowSamplerModel(CVARS, n_layers=2, hidden=8, n_dummy=3)
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


def test_log_prob_refused_with_dummies():
    model = FlowSamplerModel(CVARS, n_layers=2, hidden=8, n_dummy=1)
    params = model.init_params(seed=0)
    with pytest.raises(NotImplementedError, match="n_dummy"):
        model.log_prob(params, np.array([[50.0, 0.5]]))


def test_builder_fit_with_dummies():
    variables = [
        ContinuousVariable(name="x", description="q", lower_bound=0.0,
                           upper_bound=100.0, unit="u"),
        BinaryVariable(name="flag", description="unconstrained binary"),
    ]
    ests = [ExpectationEstimate(id="e_x", variable="x", expected_value=25.0)]
    b = DistributionBuilder(variables, ests, domain_prior="gaussian",
                            prior_bound_sds=2.0, n_dummy=4)
    b.fit(steps=800, n_samples=1024, entropy_reg=1.0, seed=0)

    x = b.sample(20000, seed=1)
    assert x.shape[1] == 2
    assert float(np.mean(x[:, 0])) == pytest.approx(25.0, abs=6.0)
    p_flag = float(np.mean(x[:, 1] > 0.5))
    assert 0.35 < p_flag < 0.65                 # dummies must not skew it
    assert np.isfinite(b.entropy())
    assert np.isfinite(b.kl_to_ref())
    assert b.marginal("x").bin_probabilities    # histogram readout intact
