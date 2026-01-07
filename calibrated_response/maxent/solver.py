"""Maximum entropy solver for distribution estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize

from calibrated_response.maxent.constraints import Constraint, ConstraintSet, to_bins
from calibrated_response.models.variable import Variable


@dataclass
class SolverConfig:
    """Configuration for the MaxEnt solver."""
    
    # Optimization parameters
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Regularization
    regularization: float = 0.01  # L2 regularization on Lagrange multipliers
    
    # Soft constraint handling
    use_soft_constraints: bool = True
    constraint_weight: float = 100.0  # Weight for constraint violations
    
    # Prior distribution
    use_uniform_prior: bool = True
    
    # Output
    verbose: bool = False


def bins2marginal(bins):
    return np.full(len(bins) - 1, 1.0 / (len(bins) - 1))

class MaxEntSolver:
    """Maximum entropy solver for distribution estimation.
    
    Finds the distribution with maximum entropy that (approximately)
    satisfies a set of constraints from query results.
    
    For hard constraints, we use Lagrange multipliers.
    For soft constraints, we minimize:
        -H(p) + weight * sum(violations^2)
    where H(p) is the entropy.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
    
    def todiscrete(self, constraint_set: ConstraintSet):
        variables = constraint_set.variables
        constraint_variables = constraint_set.constraint_variables
        constraints = constraint_set.constraints


        variable_bins = [to_bins(var) for var in variables]
        variable_marginals = [bins2marginal(bins) for bins in variable_bins]

        p0 = variable_marginals[0]
        for i in range(1, len(variables)):
            p0 = p0.reshape(p0.shape + (1,)) * variable_marginals[i].reshape((1,) * i + (-1,))
        return variable_bins, p0
    
    def solve(self, constraint_set: ConstraintSet) -> Tuple[np.ndarray, dict]:
        """Solve for the maximum entropy distribution.
        
        Args:
            constraint_set: Set of constraints to satisfy
            
        Returns:
            Tuple of (distribution, info_dict) where distribution is the
            probability mass in each bin, and info_dict contains diagnostics.
        """
        variables = constraint_set.variables
        constraint_variables = constraint_set.constraint_variables
        constraints = constraint_set.constraints


        variable_bins, p0 = self.todiscrete(constraint_set)

        return self._solve_soft(constraint_set, p0, variable_bins)
    
    def _solve_soft(
        self,
        constraint_set: ConstraintSet,
        p0: np.ndarray,
        variable_bins: list[np.ndarray],
    ) -> Tuple[np.ndarray, dict]:
        """Solve using soft constraints.
        
        Minimizes: -H(p) + weight * sum(confidence_i * violation_i^2)
        subject to: sum(p) = 1, p >= 0
        """
        
        constraints = constraint_set.constraints
        
        def objective(log_p: np.ndarray) -> float:
            """Objective function: negative entropy + constraint penalties."""
            # Use log parameterization for positivity
            p = np.exp(log_p)
            p = p / p.sum()  # Normalize
            
            # Entropy (we want to maximize, so negate)
            entropy = -np.sum(p * np.log(p + 1e-12))
            neg_entropy = -entropy
            
            # Constraint violations
            penalty = 0.0
            for c in constraints:
                actual = c.evaluate(p, bin_edges)
                target = c.target_value()
                violation = (actual - target) ** 2
                penalty += c.confidence * violation
            
            # Regularization
            reg = self.config.regularization * np.sum(log_p ** 2)
            
            return neg_entropy + self.config.constraint_weight * penalty + reg
        
        def gradient(log_p: np.ndarray) -> np.ndarray:
            """Gradient of objective (numerical approximation)."""
            eps = 1e-6
            grad = np.zeros_like(log_p)
            f0 = objective(log_p)
            for i in range(len(log_p)):
                log_p_eps = log_p.copy()
                log_p_eps[i] += eps
                grad[i] = (objective(log_p_eps) - f0) / eps
            return grad
        
        # Optimize
        log_p0 = np.log(p0 + 1e-12)
        
        result = optimize.minimize(
            objective,
            log_p0,
            method='L-BFGS-B',
            jac=gradient,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.tolerance,
                'disp': self.config.verbose,
            },
        )
        
        # Extract solution
        log_p = result.x
        p = np.exp(log_p)
        p = p / p.sum()  # Normalize
        
        # Compute diagnostics
        info = {
            'success': result.success,
            'message': result.message,
            'iterations': result.nit,
            'final_objective': result.fun,
            'constraint_violations': constraint_set.constraint_violations(p),
            'total_violation': constraint_set.total_violation(p),
            'entropy': -np.sum(p * np.log(p + 1e-12)),
        }
        
        return p, info


# class MaxEntSolver:
#     """Maximum entropy solver for distribution estimation.
    
#     Finds the distribution with maximum entropy that (approximately)
#     satisfies a set of constraints from query results.
    
#     For hard constraints, we use Lagrange multipliers.
#     For soft constraints, we minimize:
#         -H(p) + weight * sum(violations^2)
#     where H(p) is the entropy.
#     """
    
#     def __init__(self, config: Optional[SolverConfig] = None):
#         """Initialize the solver.
        
#         Args:
#             config: Solver configuration
#         """
#         self.config = config or SolverConfig()
    
#     def solve(self, constraint_set: ConstraintSet) -> Tuple[np.ndarray, dict]:
#         """Solve for the maximum entropy distribution.
        
#         Args:
#             constraint_set: Set of constraints to satisfy
            
#         Returns:
#             Tuple of (distribution, info_dict) where distribution is the
#             probability mass in each bin, and info_dict contains diagnostics.
#         """
#         n_bins = constraint_set.n_bins
#         bin_edges = constraint_set.bin_edges
        
#         # Initialize with uniform distribution
#         if self.config.use_uniform_prior:
#             p0 = np.ones(n_bins) / n_bins
#         else:
#             p0 = np.ones(n_bins) / n_bins
        
#         if self.config.use_soft_constraints:
#             return self._solve_soft(constraint_set, p0)
#         else:
#             return self._solve_hard(constraint_set, p0)
    
#     def _solve_soft(
#         self,
#         constraint_set: ConstraintSet,
#         p0: np.ndarray,
#     ) -> Tuple[np.ndarray, dict]:
#         """Solve using soft constraints.
        
#         Minimizes: -H(p) + weight * sum(confidence_i * violation_i^2)
#         subject to: sum(p) = 1, p >= 0
#         """
#         n_bins = len(p0)
#         bin_edges = constraint_set.bin_edges
#         constraints = constraint_set.constraints
        
#         def objective(log_p: np.ndarray) -> float:
#             """Objective function: negative entropy + constraint penalties."""
#             # Use log parameterization for positivity
#             p = np.exp(log_p)
#             p = p / p.sum()  # Normalize
            
#             # Entropy (we want to maximize, so negate)
#             entropy = -np.sum(p * np.log(p + 1e-12))
#             neg_entropy = -entropy
            
#             # Constraint violations
#             penalty = 0.0
#             for c in constraints:
#                 actual = c.evaluate(p, bin_edges)
#                 target = c.target_value()
#                 violation = (actual - target) ** 2
#                 penalty += c.confidence * violation
            
#             # Regularization
#             reg = self.config.regularization * np.sum(log_p ** 2)
            
#             return neg_entropy + self.config.constraint_weight * penalty + reg
        
#         def gradient(log_p: np.ndarray) -> np.ndarray:
#             """Gradient of objective (numerical approximation)."""
#             eps = 1e-6
#             grad = np.zeros_like(log_p)
#             f0 = objective(log_p)
#             for i in range(len(log_p)):
#                 log_p_eps = log_p.copy()
#                 log_p_eps[i] += eps
#                 grad[i] = (objective(log_p_eps) - f0) / eps
#             return grad
        
#         # Optimize
#         log_p0 = np.log(p0 + 1e-12)
        
#         result = optimize.minimize(
#             objective,
#             log_p0,
#             method='L-BFGS-B',
#             jac=gradient,
#             options={
#                 'maxiter': self.config.max_iterations,
#                 'ftol': self.config.tolerance,
#                 'disp': self.config.verbose,
#             },
#         )
        
#         # Extract solution
#         log_p = result.x
#         p = np.exp(log_p)
#         p = p / p.sum()  # Normalize
        
#         # Compute diagnostics
#         info = {
#             'success': result.success,
#             'message': result.message,
#             'iterations': result.nit,
#             'final_objective': result.fun,
#             'constraint_violations': constraint_set.constraint_violations(p),
#             'total_violation': constraint_set.total_violation(p),
#             'entropy': -np.sum(p * np.log(p + 1e-12)),
#         }
        
#         return p, info
    
#     def _solve_hard(
#         self,
#         constraint_set: ConstraintSet,
#         p0: np.ndarray,
#     ) -> Tuple[np.ndarray, dict]:
#         """Solve using hard constraints via Lagrange multipliers.
        
#         This is more complex and may fail if constraints are inconsistent.
#         Falls back to soft constraints if optimization fails.
#         """
#         # For simplicity, we use soft constraints with very high weight
#         # as an approximation to hard constraints
#         original_weight = self.config.constraint_weight
#         self.config.constraint_weight = 10000.0
        
#         try:
#             result, info = self._solve_soft(constraint_set, p0)
            
#             # Check if constraints are approximately satisfied
#             if info['total_violation'] > 0.1:
#                 info['warning'] = 'Hard constraints could not be exactly satisfied'
            
#             return result, info
#         finally:
#             self.config.constraint_weight = original_weight
    
#     def solve_iterative(
#         self,
#         constraint_set: ConstraintSet,
#         n_iterations: int = 5,
#     ) -> Tuple[np.ndarray, dict]:
#         """Solve iteratively, adding constraints one at a time.
        
#         This can help when constraints are inconsistent by identifying
#         which constraints cause problems.
        
#         Args:
#             constraint_set: Full set of constraints
#             n_iterations: Number of iterative refinement passes
            
#         Returns:
#             Tuple of (distribution, info_dict)
#         """
#         # Sort constraints by confidence
#         sorted_constraints = sorted(
#             constraint_set.constraints,
#             key=lambda c: c.confidence,
#             reverse=True,
#         )
        
#         # Start with empty constraint set
#         current_set = ConstraintSet(
#             domain_min=constraint_set.domain_min,
#             domain_max=constraint_set.domain_max,
#             n_bins=constraint_set.n_bins,
#         )
        
#         p = np.ones(constraint_set.n_bins) / constraint_set.n_bins
#         added_constraints = []
#         rejected_constraints = []
        
#         for constraint in sorted_constraints:
#             # Try adding constraint
#             current_set.add(constraint)
            
#             try:
#                 p_new, info = self.solve(current_set)
                
#                 # Check if solution is reasonable
#                 if info['total_violation'] < 0.5:
#                     p = p_new
#                     added_constraints.append(constraint.id)
#                 else:
#                     # Constraint causes too much violation, remove it
#                     current_set.constraints.pop()
#                     rejected_constraints.append(constraint.id)
#             except Exception as e:
#                 # Optimization failed, remove constraint
#                 current_set.constraints.pop()
#                 rejected_constraints.append(constraint.id)
        
#         # Final solve with all accepted constraints
#         p, info = self.solve(current_set)
#         info['added_constraints'] = added_constraints
#         info['rejected_constraints'] = rejected_constraints
        
#         return p, info
