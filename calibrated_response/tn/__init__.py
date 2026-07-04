"""Tensor-network density models over discretised continuous variables.

    from calibrated_response.tn import TensorChain, ContinuousVar

A linear tensor-train / Born machine (:mod:`chain`) fit by pluggable backends,
with continuous<->discrete handled in :mod:`discretize`.
"""

from .discretize import ContinuousVar, Discretizer, latent_var, belief_var
from .chain import TensorChain, FIT_BACKENDS
from .plotting import plot_pairwise
from . import losses

__all__ = ["ContinuousVar", "Discretizer", "latent_var", "belief_var",
           "TensorChain", "FIT_BACKENDS", "plot_pairwise", "losses"]
