"""Continuous <-> discrete bridge for tensor-network density models.

A tensor-train / Born machine works over *categorical* sites. We only model
continuous variables, so every variable is binned onto a uniform grid before it
enters the network and decoded back (bin centre + optional in-bin jitter) on the
way out. This module is the only place that knows about continuous domains; the
network in :mod:`chain` sees nothing but integer site values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContinuousVar:
    """One continuous variable discretised onto ``n_bins`` uniform cells."""

    name: str
    lower: float
    upper: float
    n_bins: int = 32


def latent_var(name: str, lower: float, upper: float, n_bins: int = 32) -> "ContinuousVar":
    """A :class:`ContinuousVar` used as a *latent target* site (see
    :func:`calibrated_response.tn.losses.robust_expectation`).

    Structurally identical to a :class:`ContinuousVar`, but its domain spans the
    plausible range of the *functional* being softly constrained (e.g. ``E[f(X)]``)
    rather than of a data variable, and it is marginalised out for prediction.
    Place it **adjacent** to the variable it governs in the ``vars`` list so the
    coupling bond stays cheap. Kept as a named alias so latent sites read
    distinctly where a chain is declared.
    """
    return ContinuousVar(name, lower, upper, n_bins)


def belief_var(name: str, lower: float, upper: float, n_value_bins: int = 24) -> "ContinuousVar":
    """Latent site for a :func:`calibrated_response.tn.losses.belief_expectation`.

    Holds a robust constraint's scenario: **bin 0 = broken** (the constraint
    asserts nothing) and bins 1..``n_value_bins`` = the discretised true *value*
    of the constrained functional, uniform on ``[lower, upper]``. Span the value
    range generously around the stated target — about ``target +- (4-6) *
    value_sd`` — since values outside it are only reachable via *broken*. The
    broken/value meaning is assigned by the loss, not the chain; place the site
    adjacent to its data variable. The site has ``n_value_bins + 1`` bins.
    """
    return ContinuousVar(name, lower, upper, n_value_bins + 1)


class Discretizer:
    """Maps a batch of continuous rows to/from integer bin indices.

    Column order matches the order of ``vars``. Bins are uniform on
    ``[lower, upper]``; values are clipped into range before binning.
    """

    def __init__(self, vars: Sequence[ContinuousVar]):
        self.vars = list(vars)
        self.names = [v.name for v in self.vars]
        self.n_sites = len(self.vars)
        self.dims = [v.n_bins for v in self.vars]
        self.lower = np.array([v.lower for v in self.vars], dtype=np.float64)
        self.upper = np.array([v.upper for v in self.vars], dtype=np.float64)
        self.width = (self.upper - self.lower) / np.array(self.dims, dtype=np.float64)

    # ------------------------------------------------------------------
    def to_index(self, X: np.ndarray) -> np.ndarray:
        """Continuous ``(N, D)`` -> integer bin indices ``(N, D)``."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        idx = np.floor((X - self.lower) / self.width).astype(np.int64)
        return np.clip(idx, 0, np.array(self.dims) - 1)

    def bin_centers(self, site: int) -> np.ndarray:
        v = self.vars[site]
        edges = np.linspace(v.lower, v.upper, v.n_bins + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def to_value(self, idx: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Integer bin indices ``(N, D)`` -> continuous ``(N, D)``.

        With ``rng`` given, samples uniformly *within* each bin (so decoded
        samples fill the domain instead of stacking on centres); otherwise
        returns bin centres.
        """
        idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
        lo = self.lower + idx * self.width
        if rng is None:
            return lo + 0.5 * self.width
        return lo + rng.random(idx.shape) * self.width

    # ------------------------------------------------------------------
    def bins_above(self, site: int, threshold: float) -> np.ndarray:
        """Boolean ``(n_bins,)`` mask of bins whose interval lies above ``threshold``.

        Exact when ``threshold`` falls on a bin edge (the intended usage);
        otherwise a bin straddling the threshold is counted as above iff its
        centre exceeds it.
        """
        centers = self.bin_centers(site)
        return centers > threshold

    def prob_gt(self, site: int, threshold: float, bin_mass: np.ndarray) -> float:
        """P(X_site > threshold) from a per-bin mass vector ``(n_bins,)``."""
        mask = self.bins_above(site, threshold)
        return float(np.sum(np.asarray(bin_mass)[mask]))
