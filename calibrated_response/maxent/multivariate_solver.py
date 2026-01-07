import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
import optax

from calibrated_response.models.variable import (Variable, BinaryVariable, ContinuousVariable)
from calibrated_response.maxent.constraints import (
    ConstraintSet, ProbabilityConstraint, MeanConstraint, QuantileConstraint, to_bins
)

def discretize_variables(variables: list[Variable], max_bins: int = 5):
    """Convert variables to discrete bin edges and uniform marginals."""
    bin_edges_list = [to_bins(var, max_bins=max_bins) for var in variables]
    
    # Create uniform marginals for each variable
    marginals = []
    for bins in bin_edges_list:
        n_bins = len(bins) - 1
        marginals.append(np.ones(n_bins) / n_bins)
    
    return bin_edges_list, marginals


def create_joint_distribution(marginals: list[np.ndarray]) -> np.ndarray:
    """Create joint distribution assuming independence (outer product of marginals)."""
    p = marginals[0]
    for i in range(1, len(marginals)):
        p = p.reshape(p.shape + (1,)) * marginals[i].reshape((1,) * i + (-1,))
    return p

# Helper functions for JAX

def compute_marginal_jax(p: jnp.ndarray, var_idx: int, n_vars: int) -> jnp.ndarray:
    """Compute marginal distribution for a single variable using einsum.
    
    Args:
        p: Joint distribution array
        var_idx: Index of variable to marginalize to
        n_vars: Total number of variables
    
    Returns:
        Marginal distribution for the specified variable
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    input_dims = ''.join(alphabet[i] for i in range(n_vars))
    output_dim = alphabet[var_idx]
    return jnp.einsum(f'{input_dims}->{output_dim}', p)


def evaluate_mean_constraint_jax(marginal: jnp.ndarray, bin_edges: jnp.ndarray) -> jnp.ndarray:
    """Evaluate E[X] for a marginal distribution."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return jnp.sum(marginal * bin_centers)

def evaluate_probability_constraint_jax(
    marginal: jnp.ndarray, 
    bin_edges: jnp.ndarray,
    lower_bound: float,
    upper_bound: float
) -> jnp.ndarray:
    """Evaluate P(lower <= X <= upper) for a marginal distribution."""
    # Compute overlap of each bin with [lower, upper]
    bin_lefts = bin_edges[:-1]
    bin_rights = bin_edges[1:]
    bin_widths = bin_rights - bin_lefts
    
    overlap_lefts = jnp.maximum(bin_lefts, lower_bound)
    overlap_rights = jnp.minimum(bin_rights, upper_bound)
    overlap_widths = jnp.maximum(0, overlap_rights - overlap_lefts)
    
    # Fraction of each bin that overlaps
    overlap_fracs = jnp.where(bin_widths > 0, overlap_widths / bin_widths, 0.0)
    
    return jnp.sum(marginal * overlap_fracs)


def evaluate_quantile_constraint_jax(
    marginal: jnp.ndarray,
    bin_edges: jnp.ndarray,
    quantile_value: float  # the x value where P(X <= x) = q
) -> jnp.ndarray:
    """Evaluate P(X <= value) for a marginal distribution.
    
    Returns the CDF at the quantile value, which should equal the target quantile.
    """
    bin_lefts = bin_edges[:-1]
    bin_rights = bin_edges[1:]
    bin_widths = bin_rights - bin_lefts
    
    # For each bin, compute how much of it is <= quantile_value
    # If quantile_value >= bin_right: full bin contributes
    # If quantile_value <= bin_left: nothing contributes  
    # Otherwise: linear interpolation
    frac_below = jnp.clip((quantile_value - bin_lefts) / jnp.maximum(bin_widths, 1e-10), 0.0, 1.0)
    
    return jnp.sum(marginal * frac_below)



