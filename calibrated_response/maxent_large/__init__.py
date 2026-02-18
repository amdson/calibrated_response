"""Scalable Maximum Entropy solver using continuous HMC with persistent chains."""

from calibrated_response.maxent_large.features import (
    FeatureSpec,
    MomentFeature,
    SoftThresholdFeature,
    SoftIndicatorFeature,
    ProductMomentFeature,
    ConditionalSoftThresholdFeature,
    compile_feature,
    compile_feature_vector,
)
from calibrated_response.maxent_large.mcmc import HMCConfig, PersistentBuffer
from calibrated_response.maxent_large.maxent_solver import JAXSolverConfig, MaxEntSolver
from calibrated_response.maxent_large.distribution_builder import DistributionBuilder
from calibrated_response.maxent_large.normalizer import ContinuousDomainNormalizer
from calibrated_response.maxent_large.energy_model import EnergyModel

__all__ = [
    "DistributionBuilder",
    "MaxEntSolver",
    "JAXSolverConfig",
    "HMCConfig",
    "PersistentBuffer",
    "ContinuousDomainNormalizer",
    "FeatureSpec",
    "MomentFeature",
    "SoftThresholdFeature",
    "SoftIndicatorFeature",
    "ProductMomentFeature",
    "ConditionalSoftThresholdFeature",
    "compile_feature",
    "compile_feature_vector",
    "EnergyModel",
]
