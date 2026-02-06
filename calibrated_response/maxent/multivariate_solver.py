import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
import optax

from calibrated_response.models.variable import (Variable, BinaryVariable, ContinuousVariable)
from calibrated_response.maxent.constraints import (ConditionalMeanConstraint, ConditionalThresholdConstraint, ConditionalThresholdConstraint, Constraint,
    ConstraintSet, ConstraintUnion, ProbabilityConstraint, MeanConstraint, ThresholdConstraint, ThresholdConstraint, to_bins
)

def discretize_variables(variables: Sequence[Variable], max_bins: int = 5, normalized: bool = False):
    """Convert variables to discrete bin edges and uniform marginals.
    
    Args:
        variables: List of Variable objects
        max_bins: Maximum number of bins per variable
        normalized: If True, all continuous variables use [0, 1] domain
    
    Returns:
        Tuple of (bin_edges_list, marginals)
    """
    bin_edges_list = [to_bins(var, max_bins=max_bins, normalized=normalized) for var in variables]
    
    # Create uniform marginals for each variable
    marginals = []
    for bins in bin_edges_list:
        n_bins = len(bins) - 1
        marginals.append(np.ones(n_bins) / n_bins)
    
    return bin_edges_list, marginals


def create_joint_distribution(marginals: Sequence[np.ndarray]) -> np.ndarray:
    """Create joint distribution assuming independence (outer product of marginals)."""
    p = marginals[0]
    for i in range(1, len(marginals)):
        p = p.reshape(p.shape + (1,)) * marginals[i].reshape((1,) * i + (-1,))
    return p

from typing import List, Sequence
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

def evaluate_threshold_constraint_jax(
    marginal: jnp.ndarray,
    bin_edges: jnp.ndarray,
    threshold_value: float  # the x value where P(X <= x) = q
) -> jnp.ndarray:
    """Evaluate P(X <= value) for a marginal distribution.
    
    Returns the CDF at the threshold value, which should equal the target probability.
    """
    bin_lefts = bin_edges[:-1]
    bin_rights = bin_edges[1:]
    bin_widths = bin_rights - bin_lefts
    
    # For each bin, compute how much of it is <= quantile_value
    # If quantile_value >= bin_right: full bin contributes
    # If quantile_value <= bin_left: nothing contributes  
    # Otherwise: linear interpolation
    frac_below = jnp.clip((threshold_value - bin_lefts) / jnp.maximum(bin_widths, 1e-10), 0.0, 1.0)
    
    return jnp.sum(marginal * frac_below)

def evaluate_precondition_threshold_constraint_jax(p: jnp.ndarray, precondition_mask: jnp.ndarray, marginal_index: int, 
                                                  bin_edges: jnp.ndarray, threshold_value: float) -> jnp.ndarray:
    """Evaluate a threshold constraint under a precondition mask."""
    # Apply precondition mask to joint distribution
    p_masked = p * precondition_mask
    p_masked /= jnp.sum(p_masked)  # Renormalize
    # Compute marginal for the specified variable
    marginal = compute_marginal_jax(p_masked, marginal_index, len(p.shape))
    # Evaluate threshold constraint on the marginal
    bin_lefts = bin_edges[:-1]
    bin_rights = bin_edges[1:]
    bin_widths = bin_rights - bin_lefts
    frac_below = jnp.clip((threshold_value - bin_lefts) / jnp.maximum(bin_widths, 1e-10), 0.0, 1.0)
    return jnp.sum(marginal * frac_below)

def evaluate_precondition_mean_constraint_jax(p: jnp.ndarray, precondition_mask: jnp.ndarray, marginal_index: int, 
                                                  bin_edges: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a mean constraint under a precondition mask."""
    # Apply precondition mask to joint distribution
    p_masked = p * precondition_mask
    p_masked /= jnp.sum(p_masked)  # Renormalize
    # Compute marginal for the specified variable
    marginal = compute_marginal_jax(p_masked, marginal_index, len(p.shape))
    # Evaluate mean constraint on the marginal
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return jnp.sum(marginal * bin_centers)

def gen_threshold_precondition_mask(bin_edges: List[jnp.ndarray], var_lower_bounds: np.ndarray, var_upper_bounds: np.ndarray):
    m = []
    for edges, lower, upper in zip(bin_edges, var_lower_bounds, var_upper_bounds):
        bin_lefts = edges[:-1]
        bin_rights = edges[1:]
        bin_widths = bin_rights - bin_lefts
        frac_below = jnp.clip((upper - bin_lefts) / jnp.maximum(bin_widths, 1e-10), 0.0, 1.0)
        frac_above = jnp.clip((bin_rights - lower) / jnp.maximum(bin_widths, 1e-10), 0.0, 1.0)
        frac_total = 1 - (1 - frac_below) - (1 - frac_above)
        frac_total = jnp.clip(frac_total, 0.0, 1.0)
        m.append(frac_total)

        # print(f'bin edges: {edges}')
        # print(f"Condition value: {q}, frac_below: {frac_below}")
    # Create joint mask via outer product
    precondition_mask = m[0]
    for i in range(1, len(m)):
        precondition_mask = precondition_mask.reshape(precondition_mask.shape + (1,)) * m[i].reshape((1,) * i + (-1,))
    return precondition_mask

def create_objective_fn(
    variables: Sequence[Variable],
    constraints: Sequence[Constraint],
    bin_edges_list: Sequence[np.ndarray],
    constraint_weight: float = 3.0,
    regularization: float = 0.01
):
    """Create a JAX-compatible objective function for the MaxEnt problem.
    
    The function operates on log-probabilities for numerical stability.
    """
    # Convert bin edges to JAX arrays
    n_vars = len(variables)
    bin_edges_jax = [jnp.array(b) for b in bin_edges_list]
    
    # Extract constraint parameters
    constraint_params = []
    for c in constraints:
        if isinstance(c, MeanConstraint):
            constraint_params.append(('mean', c.mean, c.confidence))
        elif isinstance(c, ProbabilityConstraint):
            constraint_params.append(('probability', (c.lower_bound, c.upper_bound, c.probability), c.confidence))
        elif isinstance(c, ThresholdConstraint):
            constraint_params.append(('threshold', (c.threshold, c.probability), c.confidence))
        elif isinstance(c, (ConditionalThresholdConstraint, ConditionalMeanConstraint)):
            var_lower_bounds = np.full((n_vars,), -np.inf)
            var_upper_bounds = np.full((n_vars,), np.inf)
            for cv, var, is_lower_bound in zip(c.condition_values, 
                            c.condition_variables, 
                            c.is_lower_bound):
                cvar_idx = variables.index(var)
                if is_lower_bound:
                    var_lower_bounds[cvar_idx] = max(cv, var_lower_bounds[cvar_idx])
                else:
                    var_upper_bounds[cvar_idx] = min(cv, var_upper_bounds[cvar_idx])
        
            precondition_mask = gen_threshold_precondition_mask(bin_edges_jax, var_lower_bounds, var_upper_bounds)
            if isinstance(c, ConditionalThresholdConstraint):
                constraint_params.append(('conditional_threshold', (c.threshold, c.probability, precondition_mask), c.confidence))
            elif isinstance(c, ConditionalMeanConstraint):
                constraint_params.append(('conditional_mean', (c.value, precondition_mask), c.confidence))
    
    def objective(log_p_flat: jnp.ndarray, shape: tuple) -> jnp.ndarray:
        """Objective function: -entropy + constraint penalties.
        
        Args:
            log_p_flat: Flattened log-probabilities
            shape: Original shape of the joint distribution
        
        Returns:
            Scalar objective value
        """
        # Reshape and normalize
        log_p = log_p_flat.reshape(shape)
        log_p_normalized = log_p - logsumexp(log_p)  # log-softmax for normalization
        p = jnp.exp(log_p_normalized)
        
        # Entropy: H(p) = -sum(p * log(p))
        # Using log_p_normalized to avoid numerical issues
        entropy = -jnp.sum(p * log_p_normalized)
        neg_entropy = -entropy
        
        # Constraint violations
        penalty = 0.0
        constraint_var_ind = [variables.index(c.target_variable) for c in constraints]
        for var_idx, (ctype, params, confidence) in zip(constraint_var_ind, constraint_params):
            marginal = compute_marginal_jax(p, var_idx, len(variables))
            bins = bin_edges_jax[var_idx]

            if ctype == 'mean':
                actual = evaluate_mean_constraint_jax(marginal, bins)
                target = params
            elif ctype == 'probability':
                lower, upper, prob = params
                actual = evaluate_probability_constraint_jax(marginal, bins, lower, upper)
                target = prob
            elif ctype == 'threshold':
                value, threshold = params
                actual = evaluate_threshold_constraint_jax(marginal, bins, value)
                target = threshold
            elif ctype == 'conditional_threshold':
                value, threshold, precondition_mask = params
                # Generate precondition mask
                actual = evaluate_precondition_threshold_constraint_jax(p, precondition_mask, var_idx, bins, value)
                target = threshold
            elif ctype == 'conditional_mean':
                value, precondition_mask = params
                # Generate precondition mask
                actual = evaluate_precondition_mean_constraint_jax(p, precondition_mask, var_idx, bins)
                target = value
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")
            
            # print(f"Constraint {c.id}: type: {ctype}, target: {target}, actual: {actual}")
            
            violation = (actual - target) ** 2
            penalty += confidence * violation
        
        # L2 regularization on log-probabilities
        reg = regularization * jnp.sum(log_p_flat ** 2)

        return neg_entropy + constraint_weight * penalty + reg
    
    return objective

from jaxopt import LBFGS

def solve_maxent_lbfgs(
    objective_fn,
    p0: np.ndarray,
    maxiter: int = 500,
    tol: float = 1e-6,
    verbose: bool = True
):
    """Solve the MaxEnt problem using JAX and jaxopt L-BFGS.
    
    Args:
        objective_fn: JAX-compatible objective function
        p0: Initial distribution
        maxiter: Maximum number of iterations
        tol: Convergence tolerance
        verbose: Whether to print progress
    
    Returns:
        Tuple of (optimized_distribution, info_dict)
    """
    shape = p0.shape
    log_p0 = jnp.log(p0.flatten() + 1e-12)
    
    # Create objective that takes shape as a fixed argument
    def obj_with_shape(log_p_flat):
        return objective_fn(log_p_flat, shape)
    
    # Create L-BFGS solver
    solver = LBFGS(
        fun=obj_with_shape,
        maxiter=maxiter,
        tol=tol,
        verbose=verbose
    )
    
    # Run optimization
    result = solver.run(log_p0)
    log_p_optimal = result.params
    
    # Extract final distribution
    log_p_normalized = log_p_optimal.reshape(shape) - logsumexp(log_p_optimal)
    p_final = np.array(jnp.exp(log_p_normalized))
    
    # Compute final entropy
    entropy = -np.sum(p_final * np.log(p_final + 1e-12))
    
    # Get final loss
    final_loss = float(obj_with_shape(log_p_optimal))
    
    info = {
        'n_iterations': int(result.state.iter_num),
        'final_loss': final_loss,
        'entropy': entropy,
        'converged': result.state.error < tol,
        'error': float(result.state.error),
        'state': result.state
    }
    
    if verbose:
        print(f"\nL-BFGS completed:")
        print(f"  Iterations: {info['n_iterations']}")
        print(f"  Final error: {info['error']:.2e}")
        print(f"  Converged: {info['converged']}")
    
    return p_final, info

def extract_marginal(p: np.ndarray, var_idx: int) -> np.ndarray:
    """Extract marginal distribution for a variable using NumPy.
    
    Args:
        p: Joint distribution array
        var_idx: Index of variable to marginalize to
    
    Returns:
        Marginal distribution for the specified variable
    """
    n_vars = len(p.shape)
    axes_to_sum = tuple(i for i in range(n_vars) if i != var_idx)
    return p.sum(axis=axes_to_sum)

def jax_evaluate(
    constraint: Constraint, 
    p: np.ndarray, 
    var_idx: int, 
    bin_edges: np.ndarray,
    variables: Sequence[Variable]
) -> float:
    """Evaluate a constraint on a distribution using JAX functions.
    
    This function is used for testing constraint adherence after optimization.
    
    Args:
        constraint: The constraint to evaluate
        p: Joint distribution array
        var_idx: Index of the target variable
        bin_edges: Bin edges for the target variable
        variables: Sequence[Variable] of all variables (needed for conditional constraints)
    
    Returns:
        The actual value of the constraint on the distribution
    """
    # Convert to JAX arrays
    p_jax = jnp.array(p)
    bin_edges_jax = jnp.array(bin_edges)
    
    # Compute marginal for the target variable
    marginal = compute_marginal_jax(p_jax, var_idx, len(p.shape))
    
    # Evaluate based on constraint type
    if isinstance(constraint, MeanConstraint):
        actual = float(evaluate_mean_constraint_jax(marginal, bin_edges_jax))
    
    elif isinstance(constraint, ProbabilityConstraint):
        actual = float(evaluate_probability_constraint_jax(
            marginal, bin_edges_jax, 
            constraint.lower_bound, constraint.upper_bound
        ))
    
    elif isinstance(constraint, ThresholdConstraint):
        # Simple threshold constraint
        actual = float(evaluate_threshold_constraint_jax(
            marginal, bin_edges_jax, constraint.threshold
        ))
    
    elif isinstance(constraint, (ConditionalThresholdConstraint, ConditionalMeanConstraint)):

        # Conditional constraint (quantile or mean)
        # Build the precondition mask
        n_vars = len(variables)
        var_lower_bounds = np.full((n_vars,), -np.inf)
        var_upper_bounds = np.full((n_vars,), np.inf)

        for cv, var, is_lower_bound in zip(constraint.condition_values, 
                                 constraint.condition_variables, 
                                 constraint.is_lower_bound):
            cvar_idx = variables.index(var)
            if is_lower_bound:
                var_lower_bounds[cvar_idx] = max(cv, var_lower_bounds[cvar_idx])
            else:
                var_upper_bounds[cvar_idx] = min(cv, var_upper_bounds[cvar_idx])
        
        # Get bin edges for all variables (use normalized domain)
        all_bin_edges = [jnp.array(to_bins(v, max_bins=len(bin_edges)-1, normalized=True)) for v in variables]
        precondition_mask = gen_threshold_precondition_mask(all_bin_edges, var_lower_bounds, var_upper_bounds)
        
        if isinstance(constraint, ConditionalThresholdConstraint):
            # Conditional threshold constraint
            actual = float(evaluate_precondition_threshold_constraint_jax(
                p_jax, precondition_mask, var_idx, bin_edges_jax, constraint.threshold
            ))
        else:
            # Conditional mean constraint
            actual = float(evaluate_precondition_mean_constraint_jax(
                p_jax, precondition_mask, var_idx, bin_edges_jax,
            ))
    else:
        raise ValueError(f"Unknown constraint type: {type(constraint)}")
    
    return actual

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

class MultivariateMaxEntSolver:
    """JAX-based multivariate maximum entropy solver using L-BFGS."""
    
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
        # Discretize using normalized [0, 1] domain for all variables
        # The DistributionBuilder normalizes constraints and denormalizes results
        bin_edges_list, marginals = discretize_variables(
            variables, max_bins=self.config.max_bins, normalized=True
        )
        p0 = create_joint_distribution(marginals)
        
        # Create objective
        objective_fn = create_objective_fn(
            variables=variables,
            constraints=constraints,
            bin_edges_list=bin_edges_list,
            constraint_weight=self.config.constraint_weight,
            regularization=self.config.regularization
        )

        initial_error = objective_fn(jnp.log(p0.flatten() + 1e-12), p0.shape)
        if self.config.verbose:
            print(f"Initial objective value: {initial_error:.6f}")
        
        # Solve using L-BFGS
        p_optimal, info = solve_maxent_lbfgs(
            objective_fn,
            p0,
            maxiter=self.config.maxiter,
            tol=self.config.tolerance,
            verbose=self.config.verbose
        )
        
        # Add constraint satisfaction to info
        info['constraint_satisfaction'] = {}
        constraint_var_ind = [variables.index(c.target_variable) for c in constraints]
        for c, var_idx in zip(constraints, constraint_var_ind):
            actual = jax_evaluate(c, p_optimal, var_idx, bin_edges_list[var_idx], variables)
            target = c.target_value()
            info['constraint_satisfaction'][c.id] = {
                'target': target,
                'actual': actual,
                'error': abs(actual - target)
            }
            if self.config.verbose:
                print(f"Constraint {c.id}: target={target}, actual={actual}, error={abs(actual - target)}")
        
        return p_optimal, bin_edges_list, info
