"""Implicit-sampler MaxEnt-by-moment-matching with sample-native losses.

Instead of fitting an energy model ``p_θ(x) ∝ exp(-E_θ(x))`` and drawing samples
with HMC, this module fits a *sampler* directly::

    x = g_θ(z),   z ~ N(0, I)

Constraints are matched as **sample expectations** of continuous feature
functions (``E[f(x)] = target``), differentiated straight through the Monte-Carlo
average via the reparametrisation trick — no MCMC, no marginal tables, no bins.
Whole-distribution matching uses MMD (a kernel two-sample distance) rather than a
KL over a histogram.

Primary entry point is :class:`SamplerModel` (``init_params`` / ``constraint_loss``
/ ``optimize`` / query methods), built for head-to-head comparison against
:mod:`calibrated_response.tn`.  The optional maximum-entropy property is provided
by the ``entropy_reg`` knob (a soft-histogram marginal entropy proxy).
"""

from calibrated_response.maxent_sampler.model import (
    SamplerModel,
    moment, product, soft_gt, soft_lt, soft_between, hard_gt,
)
from calibrated_response.maxent_sampler.plotting import plot_pairwise
from calibrated_response.maxent_sampler.latent_credence import LatentCredenceModel
from calibrated_response.maxent_sampler.flow_sampler import FlowSampler
from calibrated_response.maxent_sampler.flow_model import FlowSamplerModel
from calibrated_response.maxent_sampler.sampler_model import (
    NeuralSampler,
    neg_gaussian_entropy_proxy,
    neg_marginal_entropy_proxy,
    neg_pairwise_entropy_proxy,
)
from calibrated_response.maxent_sampler.sampler_solver import (
    SamplerSolver,
    SamplerSolverConfig,
)
from calibrated_response.maxent_sampler.distribution_builder import DistributionBuilder

__all__ = [
    # primary sample-native model (head-to-head with calibrated_response.tn)
    "SamplerModel",
    # variant: robust credences as sampled latents (tn-style), not gate params
    "LatentCredenceModel",
    # variant: invertible flow sampler with EXACT joint entropy (scalable maxent)
    "FlowSamplerModel", "FlowSampler",
    # LLM estimates -> fitted flow -> marginal Distributions (maxent_smm-parity API)
    "DistributionBuilder",
    # constraint feature factories
    "moment", "product", "soft_gt", "soft_lt", "soft_between", "hard_gt",
    # sample-based pairwise corner plot (parallel to tn.plot_pairwise)
    "plot_pairwise",
    # underlying sampler network + entropy proxies (marginal/pairwise = bounded-domain safe)
    "NeuralSampler", "neg_gaussian_entropy_proxy", "neg_marginal_entropy_proxy",
    "neg_pairwise_entropy_proxy",
    # lower-level standalone moment-matching solver
    "SamplerSolver", "SamplerSolverConfig",
]
