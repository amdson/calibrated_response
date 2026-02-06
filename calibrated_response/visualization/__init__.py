"""Visualization utilities for calibrated response distributions."""

from calibrated_response.visualization.pairplot import (
    plot_pairwise_marginals,
    plot_pairwise_from_builder,
    compute_1d_marginal,
    compute_2d_marginal,
)

__all__ = [
    "plot_pairwise_marginals",
    "plot_pairwise_from_builder",
    "compute_1d_marginal",
    "compute_2d_marginal",
]
