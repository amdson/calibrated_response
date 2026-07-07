"""Corner-style pairwise-marginal plots for the implicit sampler.

The sample-based parallel to :func:`calibrated_response.tn.plot_pairwise`.  Where
the tensor-network version draws *exact* marginal contractions, this draws
histograms of a batch of model **samples** — the natural read-out for an implicit
sampler, which has no closed-form marginals.

    from calibrated_response.maxent_sampler import plot_pairwise
    fig, axes = plot_pairwise(model, params, sites=[0, 3, 7], ref_samples=Xref[:, [0,3,7]])

Pass ``ref_samples`` (a ground-truth / reference batch over the same ``sites``) to
overlay the true marginals (red outline on the diagonal, white contours on the
2-D panels) so the chart doubles as a recovery check.
"""

from __future__ import annotations

import numpy as np


def plot_pairwise(model, params, sites=None, names=None, n_samples=20000, seed=0,
                  bins=40, cmap="viridis", figsize=None, threshold=None,
                  ref_samples=None, save=None):
    """Corner plot of the sampler's 1-D and 2-D marginals over a subset of sites.

    Diagonal: per-variable sample histogram ``p(X_i)``.
    Lower triangle ``(a, b)``: 2-D histogram of ``p(X_b, X_a)`` (``X_b`` on x,
    ``X_a`` on y).  Upper triangle is hidden (standard corner layout).

    Parameters
    ----------
    model : SamplerModel
    params : fitted parameters
    sites : sequence of int, optional
        Which variables to include (defaults to all ``model.n``).  Use a subset —
        a full 30-variable corner plot is unreadable.
    names : sequence of str, optional
        Axis labels for ``sites`` (defaults to the variables' names).
    n_samples, seed : int
        Size / seed of the sample batch drawn from the model.
    bins, cmap, figsize : plotting controls.
    threshold : scalar or {name: value}, optional
        Draws reference lines (handy for threshold-defined problems).
    ref_samples : array (M, len(sites)), optional
        Ground-truth / reference samples over the same ``sites``; overlaid as the
        true marginals (diagonal outline + 2-D contours) for a recovery check.
    save : path, optional
        If given, save the figure there.
    """
    import matplotlib.pyplot as plt

    disc = model.disc
    if sites is None:
        sites = list(range(model.n))
    sites = list(sites)
    m = len(sites)
    names = names or [disc.names[s] for s in sites]
    if figsize is None:
        figsize = (2.4 * m, 2.4 * m)

    X = np.asarray(model.sample(params, n_samples, seed=seed))[:, sites]
    R = None if ref_samples is None else np.asarray(ref_samples)
    lo = [float(disc.lower[s]) for s in sites]
    hi = [float(disc.upper[s]) for s in sites]

    def thr(k):
        if threshold is None:
            return None
        if np.isscalar(threshold):
            return float(threshold)
        return threshold.get(names[k])

    fig, axes = plt.subplots(m, m, figsize=figsize, squeeze=False)
    for a in range(m):
        for b in range(m):
            ax = axes[a][b]
            if b > a:                                   # hide upper triangle
                ax.axis("off")
                continue

            if a == b:                                  # diagonal: 1-D marginal
                ax.hist(X[:, a], bins=bins, range=(lo[a], hi[a]),
                        color="0.6", density=True)
                if R is not None:
                    ax.hist(R[:, a], bins=bins, range=(lo[a], hi[a]),
                            histtype="step", color="crimson", lw=1.3, density=True)
                ax.set_xlim(lo[a], hi[a])
                ax.set_yticks([])
                t = thr(a)
                if t is not None:
                    ax.axvline(t, color="crimson", lw=1, ls="--")
            else:                                       # lower: 2-D marginal
                ax.hist2d(X[:, b], X[:, a], bins=bins,
                          range=[[lo[b], hi[b]], [lo[a], hi[a]]], cmap=cmap)
                if R is not None:
                    H, xe, ye = np.histogram2d(
                        R[:, b], R[:, a], bins=bins,
                        range=[[lo[b], hi[b]], [lo[a], hi[a]]])
                    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
                    ax.contour(xc, yc, H.T, levels=3, colors="w",
                               linewidths=0.7, alpha=0.7)
                ti, tj = thr(a), thr(b)
                if tj is not None:
                    ax.axvline(tj, color="w", lw=0.8, ls="--", alpha=0.7)
                if ti is not None:
                    ax.axhline(ti, color="w", lw=0.8, ls="--", alpha=0.7)

            if a == m - 1:
                ax.set_xlabel(names[b])
            else:
                ax.set_xticklabels([])
            if b == 0:
                ax.set_ylabel(names[a])
            else:
                ax.set_yticklabels([])

    fig.suptitle("pairwise marginals — sampler"
                 + (f" (N={n_samples}) vs truth (red)" if R is not None
                    else f" (N={n_samples})"))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig, axes
