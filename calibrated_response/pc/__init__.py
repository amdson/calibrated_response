"""Generic JAX probabilistic-circuit solver.

A standalone, smooth & decomposable tensorized SPN (spec ``pc/spec.md``) trained
by gradient descent against arbitrary loss functions defined over its marginals
and moments. Depends only on jax/optax/numpy — nothing from the rest of the repo.

    from calibrated_response.pc import Circuit, VarSpec, train, TrainConfig
    from calibrated_response.pc import losses, sampling
"""

from .circuit import Circuit, VarSpec
from .gaussian_baseline import GaussianModel
from .train import train, TrainConfig
from . import leaves, losses, sampling
from .region_graph import build_region_graph

__all__ = [
    "Circuit",
    "VarSpec",
    "GaussianModel",
    "train",
    "TrainConfig",
    "leaves",
    "losses",
    "sampling",
    "build_region_graph",
]
