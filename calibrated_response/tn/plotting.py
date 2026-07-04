"""Corner-style pairwise-marginal plots for tensor-chain models.

Every panel is an *exact* marginal of the network (:meth:`TensorChain.site_marginal`
and :meth:`TensorChain.pair_marginal`), not a sample histogram — so the picture
is the model's true low-order structure, computed by contraction.

    from calibrated_response.tn import plot_pairwise
    fig, axes = plot_pairwise(model, params, threshold=50)
"""

from __future__ import annotations

import numpy as np


def _edges(disc, site):
    v = disc.vars[site]
    return np.linspace(v.lower, v.upper, v.n_bins + 1)


def plot_pairwise(model, params, names=None, cmap="viridis", figsize=None,
                  threshold=None, save=None):
    """Corner plot of the model's 1-D and 2-D marginals.

    Diagonal: per-variable marginal ``p(X_i)`` as a bar over the domain.
    Lower triangle ``(i, j)``: heatmap of ``p(X_i, X_j)`` (``X_j`` on x, ``X_i``
    on y). Upper triangle is hidden (standard corner layout).

    ``threshold`` (a scalar, or a dict ``{name: value}``) draws reference lines,
    handy for threshold-defined problems like the A->B->C chain at 50.
    """
    import matplotlib.pyplot as plt

    disc = model.disc
    n = model.n
    names = names or disc.names
    if figsize is None:
        figsize = (2.4 * n, 2.4 * n)

    def thr(site):
        if threshold is None:
            return None
        if np.isscalar(threshold):
            return float(threshold)
        return threshold.get(names[site])

    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if j > i:                                   # hide upper triangle
                ax.axis("off")
                continue

            if i == j:                                  # diagonal: 1-D marginal
                mass = np.asarray(model.site_marginal(params, i))
                centers = disc.bin_centers(i)
                width = disc.width[i]
                ax.bar(centers, mass, width=width, color="0.4", align="center")
                ax.set_xlim(disc.lower[i], disc.upper[i])
                ax.set_yticks([])
                t = thr(i)
                if t is not None:
                    ax.axvline(t, color="crimson", lw=1, ls="--")
            else:                                       # lower: 2-D marginal
                M = np.asarray(model.pair_marginal(params, i, j))   # (d_i, d_j)
                ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                          extent=[disc.lower[j], disc.upper[j],
                                  disc.lower[i], disc.upper[i]])
                ti, tj = thr(i), thr(j)
                if tj is not None:
                    ax.axvline(tj, color="w", lw=0.8, ls="--", alpha=0.7)
                if ti is not None:
                    ax.axhline(ti, color="w", lw=0.8, ls="--", alpha=0.7)

            if i == n - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(names[i])
            elif j == 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticklabels([])

    fig.suptitle(f"pairwise marginals ({model.kind} tensor chain, r={model.r})")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig, axes
