"""Scalable Maximum Entropy solver using continuous HMC with persistent chains."""

from calibrated_response.maxent_smm.features import (
    FeatureSpec,
    MomentFeature,
    SoftThresholdFeature,
    SoftIndicatorFeature,
    ProductMomentFeature,
    ConditionalSoftThresholdFeature,
    compile_feature,
    compile_feature_vector,
)
from calibrated_response.maxent_smm.mcmc import HMCConfig, PersistentBuffer
from calibrated_response.maxent_smm.maxent_solver import JAXSolverConfig, MaxEntSolver
from calibrated_response.maxent_smm.distribution_builder import DistributionBuilder
from calibrated_response.maxent_smm.normalizer import ContinuousDomainNormalizer
from calibrated_response.maxent_smm.energy_model import EnergyModel

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
