"""Pairwise marginal visualization for multivariate distributions."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from calibrated_response.models.variable import Variable, ContinuousVariable, BinaryVariable
from calibrated_response.maxent.constraints import (
    Constraint, ThresholdConstraint, MeanConstraint, ProbabilityConstraint,
    ConditionalThresholdConstraint, ConditionalMeanConstraint,
)

# Colors for distinguishing multiple conditional constraints
_COND_COLORS = ['#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']


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


def _compute_conditional_marginal(
    ordered_joint: np.ndarray,
    ordered_edges: list[np.ndarray],
    ordered_vars: list[Variable],
    target_var_idx: int,
    constraint,
) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
    """Compute a conditional 1D marginal for a conditional constraint.

    Applies the constraint's conditions as a mask on the joint distribution,
    renormalizes, and marginalizes to the target variable.

    Returns:
        Tuple of (marginal, bin_edges, label) or None if no mass in condition region.
    """
    n_vars = ordered_joint.ndim
    mask = np.ones_like(ordered_joint)
    label_parts = []

    for cond_var, cond_val, is_lower in zip(
        constraint.condition_variables,
        constraint.condition_values,
        constraint.is_lower_bound,
    ):
        cond_name = cond_var.name
        cond_idx = next(
            (i for i, v in enumerate(ordered_vars) if v.name == cond_name), None
        )
        if cond_idx is None:
            continue

        edges = ordered_edges[cond_idx]
        centers = (edges[:-1] + edges[1:]) / 2

        if is_lower:
            bin_mask = (centers > cond_val).astype(float)
            label_parts.append(f'{cond_name}>{cond_val:.2g}')
        else:
            bin_mask = (centers <= cond_val).astype(float)
            label_parts.append(f'{cond_name}\u2264{cond_val:.2g}')

        shape = [1] * n_vars
        shape[cond_idx] = len(centers)
        mask *= bin_mask.reshape(shape)

    conditioned = ordered_joint * mask
    total = conditioned.sum()
    if total < 1e-12:
        return None
    conditioned /= total

    marginal = compute_1d_marginal(conditioned, ordered_edges, target_var_idx)
    label = f'P(\u00b7|{", ".join(label_parts)})'
    return marginal[0], marginal[1], label


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

                # Overlay constraint info on diagonal
                if constraints:
                    _overlay_constraints_diagonal(
                        ax, constraints, ordered_vars, ordered_edges,
                        ordered_joint, i,
                    )
                
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
                        constraint_color,
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


def _overlay_constraints_diagonal(
    ax: Axes,
    constraints: list,
    ordered_vars: list[Variable],
    ordered_edges: list[np.ndarray],
    ordered_joint: np.ndarray,
    var_idx: int,
) -> None:
    """Overlay constraint visualizations on a diagonal (1D marginal) plot.

    For simple constraints: draws threshold / mean lines.
    For conditional constraints: overlays the conditional marginal as a
    step-function outline so the shift from the unconditional marginal
    is immediately visible.
    """
    var_name = ordered_vars[var_idx].name
    cond_count = 0
    has_legend = False

    for constraint in constraints:
        target_var = getattr(constraint, 'target_variable', None)
        if target_var is None or getattr(target_var, 'name', None) != var_name:
            continue

        # --- Simple threshold constraint ---
        if isinstance(constraint, ThresholdConstraint) and not isinstance(
            constraint, ConditionalThresholdConstraint
        ):
            ax.axvline(
                constraint.threshold, color='green', linestyle=':',
                linewidth=1.5, alpha=0.7,
                label=f'P(\u2264{constraint.threshold:.2g})={constraint.probability:.2g}',
            )
            has_legend = True

        # --- Simple mean constraint ---
        elif isinstance(constraint, MeanConstraint):
            ax.axvline(
                constraint.mean, color='green', linestyle=':',
                linewidth=1.5, alpha=0.7,
                label=f'E[X]={constraint.mean:.2g}',
            )
            has_legend = True

        # --- Conditional constraints: overlay conditional marginal ---
        elif isinstance(constraint, (ConditionalThresholdConstraint, ConditionalMeanConstraint)):
            result = _compute_conditional_marginal(
                ordered_joint, ordered_edges, ordered_vars,
                var_idx, constraint,
            )
            if result is None:
                continue
            marginal, edges, label = result
            color = _COND_COLORS[cond_count % len(_COND_COLORS)]

            # Draw step-function outline of conditional marginal
            step_x = np.repeat(edges, 2)[1:-1]
            step_y = np.repeat(marginal, 2)
            ax.plot(step_x, step_y, color=color, linewidth=2, alpha=0.85, label=label)

            # Mark the target value on the x-axis
            if isinstance(constraint, ConditionalThresholdConstraint):
                ax.axvline(
                    constraint.threshold, color=color, linestyle=':',
                    linewidth=1.5, alpha=0.6,
                )
            elif isinstance(constraint, ConditionalMeanConstraint):
                centers = (edges[:-1] + edges[1:]) / 2
                cond_mean = float(np.sum(marginal * centers))
                ax.axvline(
                    cond_mean, color=color, linestyle='--',
                    linewidth=1.5, alpha=0.6,
                )

            cond_count += 1
            has_legend = True

    if has_legend:
        ax.legend(fontsize=6, loc='upper right')


def _overlay_constraints(
    ax: Axes,
    constraints: list,
    var_x: str,
    var_y: str,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    color: str,
) -> None:
    """Overlay constraint lines/regions on a 2D marginal plot.

    Handles all constraint types:
    - ProbabilityConstraint: dashed line at each finite bound
    - ThresholdConstraint: dashed line at the threshold value
    - MeanConstraint: dashed line at the mean value
    - ConditionalThresholdConstraint / ConditionalMeanConstraint:
        * shaded region showing the conditioning range
        * dashed line showing the target threshold / mean
    """
    cond_count = 0

    for constraint in constraints:
        target_var = getattr(constraint, 'target_variable', None)
        if target_var is None:
            continue
        target_name = getattr(target_var, 'name', None)
        if target_name is None:
            continue

        # ---- Conditional constraints ----
        if isinstance(constraint, (ConditionalThresholdConstraint, ConditionalMeanConstraint)):
            cond_color = _COND_COLORS[cond_count % len(_COND_COLORS)]

            # Draw condition regions
            for cond_var, cond_val, is_lower in zip(
                constraint.condition_variables,
                constraint.condition_values,
                constraint.is_lower_bound,
            ):
                cond_name = cond_var.name

                if cond_name == var_x:
                    if is_lower:
                        ax.axvspan(cond_val, edges_x[-1], alpha=0.12, color=cond_color, zorder=0)
                    else:
                        ax.axvspan(edges_x[0], cond_val, alpha=0.12, color=cond_color, zorder=0)
                    ax.axvline(cond_val, color=cond_color, linestyle=':', linewidth=1.5, alpha=0.6)
                elif cond_name == var_y:
                    if is_lower:
                        ax.axhspan(cond_val, edges_y[-1], alpha=0.12, color=cond_color, zorder=0)
                    else:
                        ax.axhspan(edges_y[0], cond_val, alpha=0.12, color=cond_color, zorder=0)
                    ax.axhline(cond_val, color=cond_color, linestyle=':', linewidth=1.5, alpha=0.6)

            # Draw target value (threshold or mean)
            if isinstance(constraint, ConditionalThresholdConstraint):
                target_val = constraint.threshold
            else:
                target_val = constraint.value

            if target_name == var_x:
                ax.axvline(target_val, color=cond_color, linestyle='--', linewidth=2, alpha=0.8)
            elif target_name == var_y:
                ax.axhline(target_val, color=cond_color, linestyle='--', linewidth=2, alpha=0.8)

            cond_count += 1

        # ---- Simple threshold ----
        elif isinstance(constraint, ThresholdConstraint):
            if target_name == var_x:
                ax.axvline(constraint.threshold, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            elif target_name == var_y:
                ax.axhline(constraint.threshold, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

        # ---- Simple mean ----
        elif isinstance(constraint, MeanConstraint):
            if target_name == var_x:
                ax.axvline(constraint.mean, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            elif target_name == var_y:
                ax.axhline(constraint.mean, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

        # ---- Probability constraint ----
        elif isinstance(constraint, ProbabilityConstraint):
            lower_bound = constraint.lower_bound
            upper_bound = constraint.upper_bound
            if target_name == var_x:
                if lower_bound > edges_x[0]:
                    ax.axvline(lower_bound, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                if upper_bound < edges_x[-1]:
                    ax.axvline(upper_bound, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            elif target_name == var_y:
                if lower_bound > edges_y[0]:
                    ax.axhline(lower_bound, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                if upper_bound < edges_y[-1]:
                    ax.axhline(upper_bound, color=color, linestyle='--', linewidth=1.5, alpha=0.7)


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
