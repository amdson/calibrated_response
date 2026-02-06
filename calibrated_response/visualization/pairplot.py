"""Pairwise marginal visualization for multivariate distributions."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from calibrated_response.models.variable import Variable, ContinuousVariable, BinaryVariable


def compute_2d_marginal(
    joint_distribution: np.ndarray,
    bin_edges_list: list[np.ndarray],
    var_idx_i: int,
    var_idx_j: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2D marginal distribution for two variables.
    
    Args:
        joint_distribution: N-dimensional joint probability array
        bin_edges_list: List of bin edges for each variable
        var_idx_i: Index of first variable (row in plot)
        var_idx_j: Index of second variable (column in plot)
        
    Returns:
        Tuple of (2d_marginal, bin_edges_i, bin_edges_j)
    """
    n_vars = joint_distribution.ndim
    
    # Sum over all axes except i and j
    axes_to_sum = [k for k in range(n_vars) if k not in (var_idx_i, var_idx_j)]
    
    marginal_2d = joint_distribution
    # Sum in reverse order to keep indices valid
    for ax in sorted(axes_to_sum, reverse=True):
        marginal_2d = marginal_2d.sum(axis=ax)
    
    # If i > j, we need to transpose to get correct orientation
    if var_idx_i > var_idx_j:
        marginal_2d = marginal_2d.T
        return marginal_2d, bin_edges_list[var_idx_j], bin_edges_list[var_idx_i]
    
    return marginal_2d, bin_edges_list[var_idx_i], bin_edges_list[var_idx_j]


def compute_1d_marginal(
    joint_distribution: np.ndarray,
    bin_edges_list: list[np.ndarray],
    var_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 1D marginal distribution for a single variable.
    
    Args:
        joint_distribution: N-dimensional joint probability array
        bin_edges_list: List of bin edges for each variable
        var_idx: Index of the variable
        
    Returns:
        Tuple of (1d_marginal, bin_edges)
    """
    n_vars = joint_distribution.ndim
    
    # Sum over all axes except var_idx
    axes_to_sum = [k for k in range(n_vars) if k != var_idx]
    
    marginal_1d = joint_distribution
    for ax in sorted(axes_to_sum, reverse=True):
        marginal_1d = marginal_1d.sum(axis=ax)
    
    return marginal_1d, bin_edges_list[var_idx]


def plot_pairwise_marginals(
    joint_distribution: np.ndarray,
    bin_edges_list: list[np.ndarray],
    variables: Sequence[Variable],
    target_variable: Optional[str] = None,
    constraints: Optional[list] = None,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = 'Blues',
    diagonal_color: str = 'steelblue',
    constraint_color: str = 'red',
) -> Figure:
    """
    Create a lower triangular NxN pairwise marginal plot.
    
    The diagonal shows 1D marginal distributions.
    The lower triangle shows 2D marginal heatmaps.
    The upper triangle is left empty.
    
    Args:
        joint_distribution: N-dimensional joint probability array
        bin_edges_list: List of bin edges for each variable (length N)
        variables: List of Variable objects (length N)
        target_variable: Name of target variable to put in first row/column
        constraints: Optional list of constraints to overlay
        figsize: Figure size (width, height). Defaults to (3*N, 3*N)
        cmap: Colormap for 2D marginals
        diagonal_color: Color for 1D marginal histograms
        constraint_color: Color for constraint overlays
        
    Returns:
        matplotlib Figure object
    """
    n_vars = len(variables)
    
    # Reorder variables to put target first
    var_order = list(range(n_vars))
    var_names = [v.name for v in variables]
    
    if target_variable and target_variable in var_names:
        target_idx = var_names.index(target_variable)
        var_order = [target_idx] + [i for i in range(n_vars) if i != target_idx]
    
    # Reordered variables and bin edges
    ordered_vars = [variables[i] for i in var_order]
    ordered_edges = [bin_edges_list[i] for i in var_order]
    
    # We need to reorder the joint distribution axes too
    ordered_joint = np.moveaxis(joint_distribution, var_order, list(range(n_vars)))
    
    # Create figure
    if figsize is None:
        figsize = (3 * n_vars, 3 * n_vars)
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize)
    
    # Handle single variable case
    if n_vars == 1:
        axes = np.array([[axes]])
    
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: 1D marginal histogram
                marginal, edges = compute_1d_marginal(ordered_joint, ordered_edges, i)
                centers = (edges[:-1] + edges[1:]) / 2
                widths = edges[1:] - edges[:-1]
                
                ax.bar(centers, marginal, width=widths * 0.9, 
                       alpha=0.7, color=diagonal_color, edgecolor='navy')
                ax.set_xlim(edges[0], edges[-1])
                
                # Add mean line
                mean = np.sum(marginal * centers)
                ax.axvline(mean, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                
            elif i > j:
                # Lower triangle: 2D marginal heatmap
                marginal_2d, edges_j, edges_i = compute_2d_marginal(
                    ordered_joint, ordered_edges, j, i
                )
                
                # Create meshgrid for pcolormesh
                X, Y = np.meshgrid(edges_j, edges_i)
                
                im = ax.pcolormesh(X, Y, marginal_2d.T, cmap=cmap, shading='flat')
                ax.set_xlim(edges_j[0], edges_j[-1])
                ax.set_ylim(edges_i[0], edges_i[-1])
                
                # Overlay constraints if applicable
                if constraints:
                    _overlay_constraints(
                        ax, constraints, 
                        ordered_vars[j].name, ordered_vars[i].name,
                        edges_j, edges_i,
                        constraint_color
                    )
                
            else:
                # Upper triangle: empty
                ax.axis('off')
            
            # Labels
            if i == n_vars - 1:
                # Bottom row: x-axis labels
                label = ordered_vars[j].name
                if hasattr(ordered_vars[j], 'unit') and ordered_vars[j].unit:
                    label += f"\n({ordered_vars[j].unit})"
                ax.set_xlabel(label, fontsize=9)
            else:
                ax.set_xticklabels([])
                
            if j == 0 and i > 0:
                # Left column (excluding diagonal): y-axis labels
                label = ordered_vars[i].name
                if hasattr(ordered_vars[i], 'unit') and ordered_vars[i].unit:
                    label = f"({ordered_vars[i].unit})\n" + label
                ax.set_ylabel(label, fontsize=9)
            elif i == j:
                # Diagonal gets y-label too
                ax.set_ylabel('Probability', fontsize=9)
            else:
                ax.set_yticklabels([])
    
    # Title
    title = 'Pairwise Marginal Distributions'
    if target_variable:
        title += f'\n(Target: {target_variable})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def _overlay_constraints(
    ax: Axes,
    constraints: list,
    var_x: str,
    var_y: str,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    color: str,
) -> None:
    """
    Overlay constraint lines/regions on a 2D marginal plot.
    
    Args:
        ax: Matplotlib axes
        constraints: List of constraint objects
        var_x: Name of x-axis variable
        var_y: Name of y-axis variable
        edges_x: Bin edges for x variable
        edges_y: Bin edges for y variable
        color: Color for constraint lines
    """
    for constraint in constraints:
        # Check if constraint involves either variable
        target_var = getattr(constraint, 'target_variable', None)
        if target_var is None:
            continue
        
        constraint_var = getattr(target_var, 'name', None)
        if constraint_var is None:
            continue
            
        # For probability constraints, get the threshold from bounds
        lower_bound = getattr(constraint, 'lower_bound', None)
        upper_bound = getattr(constraint, 'upper_bound', None)
        
        if lower_bound is not None and upper_bound is not None:
            # Use the non-infinite bound as threshold
            if lower_bound > edges_x[0] and constraint_var == var_x:
                ax.axvline(lower_bound, color=color, linestyle='--', 
                          linewidth=1.5, alpha=0.7)
            elif upper_bound < edges_x[-1] and constraint_var == var_x:
                ax.axvline(upper_bound, color=color, linestyle='--', 
                          linewidth=1.5, alpha=0.7)
            elif lower_bound > edges_y[0] and constraint_var == var_y:
                ax.axhline(lower_bound, color=color, linestyle='--', 
                          linewidth=1.5, alpha=0.7)
            elif upper_bound < edges_y[-1] and constraint_var == var_y:
                ax.axhline(upper_bound, color=color, linestyle='--', 
                          linewidth=1.5, alpha=0.7)


def plot_pairwise_from_builder(
    builder,
    target_variable: Optional[str] = None,
    show_constraints: bool = True,
    **kwargs,
) -> Figure:
    """
    Convenience function to create pairwise plot directly from a DistributionBuilder.
    
    Args:
        builder: DistributionBuilder instance (after calling build())
        target_variable: Name of target variable for first row/column
        show_constraints: Whether to overlay constraint lines
        **kwargs: Additional arguments passed to plot_pairwise_marginals
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(builder, '_last_joint') or builder._last_joint is None:
        raise ValueError("Must call builder.build() before plotting")
    
    constraints = builder.constraints if show_constraints else None
    
    return plot_pairwise_marginals(
        joint_distribution=builder._last_joint,
        bin_edges_list=builder._last_bin_edges,
        variables=builder.variables,
        target_variable=target_variable,
        constraints=constraints,
        **kwargs,
    )
