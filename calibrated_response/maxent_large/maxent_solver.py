import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
import optax

from typing import Sequence

from calibrated_response.models.variable import (Variable, BinaryVariable, ContinuousVariable)

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class JAXSolverConfig:
    """Configuration for JAX-based MaxEnt solver."""
    maxiter: int = 500
    tolerance: float = 1e-5
    constraint_weight: float = 100.0
    regularization: float = 0.001
    max_bins: int = 10
    verbose: bool = False
    regularization_type: str = 'entropy'  # 'entropy' or 'kl_gaussian'
    gaussian_mean: float = 0.5
    gaussian_std: float = 0.17
    smoothness_weight: float = 0.0

class MaxEntSolver:
    """JAX-based multivariate maximum entropy solver."""
    
    def __init__(self, config: Optional[JAXSolverConfig] = None):
        self.config = config or JAXSolverConfig()
    
    def solve(
        self,
        variables: Sequence[Variable],
        constraints: Sequence[ConstraintUnion],
    ):
        """Solve the multivariate MaxEnt problem.
        
        Args:
            variables: List of Variable objects
            constraints: List of Constraint objects
            constraint_var_indices: Index of variable for each constraint
        
        Returns:
            Tuple of (joint_distribution, bin_edges_list, info_dict)
        """

        feature_function 

def jax_partition_gradient_est(sample_features, theta):
    #Estimate partition function
    #Output gradient of log Z(theta)
    #d/dtheta log Z = E_theta(sample_features)
