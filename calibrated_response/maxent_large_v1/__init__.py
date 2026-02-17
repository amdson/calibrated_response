"""Large-scale MaxEnt via stochastic moment matching (JAX + persistent MCMC)."""

from calibrated_response.maxent_large_v1.distribution_builder import DistributionBuilder
from calibrated_response.maxent_large_v1.energy import energy, local_energy_delta
from calibrated_response.maxent_large_v1.graph import Factor, FactorGraph, SMMVariable
from calibrated_response.maxent_large_v1.targets import (
    TargetMoments,
    build_targets_from_marginals,
    build_targets_from_samples,
)
from calibrated_response.maxent_large_v1.training import SMMConfig, fit_smm

__all__ = [
    "SMMVariable",
    "Factor",
    "FactorGraph",
    "TargetMoments",
    "build_targets_from_marginals",
    "build_targets_from_samples",
    "energy",
    "local_energy_delta",
    "SMMConfig",
    "fit_smm",
    "DistributionBuilder",
]
