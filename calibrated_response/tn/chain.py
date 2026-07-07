"""Tensor-*chain* density model — a path-topology special case of :class:`TensorTree`.

A linear tensor train (matrix product state) is exactly a tree whose topology is a
path, so :class:`TensorChain` is a thin subclass of
:class:`~calibrated_response.tn.tree.TensorTree` that fixes ``edges=path_edges(n)``.
**All** queries, marginals, event/conditional probabilities, ``constraint_loss`` /
``optimize`` fitting glue, and the born/nonneg contraction engine are inherited from
the tree — there is a single implementation, so a fix in one place fixes both.

Only the quantities that genuinely exploit the *linear order* live here (the ones the
tree docstring lists as "not ported"):

    linear_moments            exact (E, Var) of Y = a·X in one sweep
    projection_distribution   full pmf of Y = a·X (partial-sum transfer DP)
    prob_lt                   P(a·X < c)
    renyi2_entropy            joint Rényi-2 / collision entropy
    amplitude_roughness       gauge-invariant <psi|L_i|psi>/<psi|psi> smoothness
    sample / sample_idx       Born-rule ancestral sampling

These reuse the original left-to-right MPS algorithms verbatim. The only wrinkle is
layout: a path node's core is stored **physical-axis-first** by the tree —
``(d, r)`` at the ends, ``(d, r_L, r_R)`` in the middle — whereas the classic MPS
algorithms expect ``(d, r)`` / ``(r, d, r)`` / ``(r, d)``. :meth:`_chain_cores`
transposes each core back to that layout (verified to reproduce the tree's
``log_prob`` / ``site_marginal`` / ``expectation`` to float32 precision for both
kinds), so every method below is fed chain-layout cores and is unchanged from the
standalone implementation.

Continuous <-> integer is handled by the shared
:class:`~calibrated_response.tn.discretize.Discretizer`; fitting by the shared
:data:`~calibrated_response.tn.backends.FIT_BACKENDS`.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .discretize import ContinuousVar
from .backends import _apply_kind, FIT_BACKENDS, reusable_adam, _fit_adam, _fit_lbfgs
from .tree import TensorTree, path_edges

_EPS = 1e-30

# re-exported so existing ``from .chain import FIT_BACKENDS, reusable_adam`` keeps working
__all__ = ["TensorChain", "FIT_BACKENDS", "reusable_adam"]


class TensorChain(TensorTree):
    """MPS density model over discretised continuous variables (a path TensorTree)."""

    def __init__(self, vars: Sequence[ContinuousVar], bond_dim: int = 8,
                 kind: str = "born"):
        if len(vars) < 2:
            raise ValueError("need at least 2 sites")
        super().__init__(vars, edges=path_edges(len(vars)), bond_dim=bond_dim, kind=kind)

    # ---- adapter: tree (physical-first) cores -> classic MPS (r,d,r) layout ----
    def _chain_cores(self, params):
        """The kind-applied cores transposed to the classic MPS layout the path-only
        algorithms below expect: ``(d, r)`` / ``(r, d, r)`` / ``(r, d)``."""
        cs = self._cores(params)
        n = self.n
        out = []
        for i, c in enumerate(cs):
            if i == 0:
                out.append(c)                                # (d, r)
            elif i == n - 1:
                out.append(c.T)                              # (d, r) -> (r, d)
            else:
                out.append(jnp.transpose(c, (1, 0, 2)))      # (d, rL, rR) -> (rL, d, rR)
        return out

    # ==================================================================
    # Rényi-2 / collision entropy  (chain transfer sweeps)
    # ==================================================================
    def _log_Z_doubled(self, cores):
        """log sum_x psi(x)^2 via the doubled transfer matrix."""
        L = jnp.einsum("xa,xb->ab", cores[0], cores[0])       # (r, r)
        logZ = 0.0
        for i in range(1, self.n - 1):
            L = jnp.einsum("ab,axc,bxd->cd", L, cores[i], cores[i])
            nrm = jnp.sqrt(jnp.sum(L * L)) + _EPS
            L = L / nrm
            logZ = logZ + jnp.log(nrm)
        z = jnp.einsum("ab,ax,bx->", L, cores[-1], cores[-1])
        return logZ + jnp.log(z + _EPS)

    def _log_Z_single(self, cores):
        """log sum_x psi(x) via the single (all-ones) transfer vector."""
        l = jnp.sum(cores[0], axis=0)                         # (r,)
        logZ = 0.0
        for i in range(1, self.n - 1):
            l = jnp.einsum("a,axc->c", l, cores[i])
            nrm = jnp.sqrt(jnp.sum(l * l)) + _EPS
            l = l / nrm
            logZ = logZ + jnp.log(nrm)
        z = jnp.einsum("a,ax->", l, cores[-1])
        return logZ + jnp.log(z + _EPS)

    def _log_Z_quadrupled(self, cores):
        """log sum_x psi(x)^4 via the quadrupled (four-copy) transfer.

        Same rescale-and-accumulate scheme as :meth:`_log_Z_doubled`, but the
        running state is four bond copies ``(r, r, r, r)`` — cost grows steeply
        with ``r`` (~O(n·d·r^5) with a good contraction path), so this is fine
        for moderate bond dims but is the first thing to feel a large ``r``.
        """
        c = cores[0]
        E = jnp.einsum("xa,xb,xc,xd->abcd", c, c, c, c)       # (r, r, r, r)
        logv = 0.0
        for i in range(1, self.n - 1):
            c = cores[i]
            E = jnp.einsum("abcd,axe,bxf,cxg,dxh->efgh", E, c, c, c, c)
            nrm = jnp.sqrt(jnp.sum(E * E)) + _EPS
            E = E / nrm
            logv = logv + jnp.log(nrm)
        c = cores[-1]
        v = jnp.einsum("abcd,ax,bx,cx,dx->", E, c, c, c, c)
        return logv + jnp.log(v + _EPS)

    def renyi2_entropy(self, params):
        """Exact joint Rényi-2 entropy ``H2(p) = -log sum_x p(x)^2`` — differentiable.

        .. deprecated::
            **Deprecated for now** — kept working but not maintained; it has not yet
            proved useful and will be revisited/reworked later (e.g. a tree-general
            collision-entropy contraction rather than the path-only four-copy sweep).
            Emits a :class:`DeprecationWarning` on call. Avoid depending on it.

        Unlike the per-site marginal entropies (:func:`losses.neg_marginal_entropy`),
        this is a *joint* quantity: it is maximised (at ``sum_i log d_i``) only by
        the uniform over full configs, so it pays for spurious correlations between
        unconstrained variables that marginal regularizers cannot see, and it lower-
        bounds the Shannon entropy. Tractable in one sweep for both kinds:

        * born:   ``sum_x p^2 = sum_x psi^4 / Z^2`` — four-copy transfer
          (:meth:`_log_Z_quadrupled`) over the doubled ``Z``.
        * nonneg: ``sum_x p^2 = sum_x psi^2 / Z^2`` — the two-copy transfer
          (:meth:`_log_Z_doubled`) the born kind already uses for its ``Z``.
        """
        warnings.warn("TensorChain.renyi2_entropy is deprecated for now (unmaintained; "
                      "to be reworked later) - see its docstring.",
                      DeprecationWarning, stacklevel=2)
        cores = self._chain_cores(params)
        if self.kind == "born":
            return 2.0 * self._log_Z_doubled(cores) - self._log_Z_quadrupled(cores)
        return 2.0 * self._log_Z_single(cores) - self._log_Z_doubled(cores)

    # ==================================================================
    # exact moments of a weighted sum  Y = a·X
    # ==================================================================
    def linear_moments(self, params, a, within_bin: bool = False):
        """Exact ``(E[Y], Var[Y])`` of ``Y = a·X`` in one O(n·r^2) sweep.

        ``X`` takes bin-centre values; ``a`` is a length-``n`` coefficient vector.
        ``Y = sum_i g_i(x_i)`` with ``g_i(k) = a_i · centre_i(k)`` is additive over
        sites, so the doubled (born) / single (nonneg) transfer carries a triple of
        running raw moments ``(m0, m1, m2)`` and combines them site by site — the
        chain analog of spec §5.3, and cheap even for very large ``n``.

        ``within_bin=True`` adds each site's uniform in-bin variance
        ``a_i^2 · width_i^2 / 12`` (a param-independent constant), for a
        continuous- rather than bin-centre-faithful ``Var``.
        """
        cores = self._chain_cores(params)
        born = self.kind == "born"
        a = jnp.asarray(a, jnp.float32)
        g = [a[i] * jnp.asarray(self.disc.bin_centers(i), jnp.float32)
             for i in range(self.n)]

        # weighted physical-sum of the local doubled/single transfer.
        # weight w over the physical index; L is the running bond state.
        if born:
            def step(L, c, w):    # middle site (r,d,r): apply from left
                return jnp.einsum("ab,axc,x,bxd->cd", L, c, w, c)
            def fin(L, c, w):     # last site (r,d): close to scalar
                return jnp.einsum("ab,ax,x,bx->", L, c, w, c)
            c0 = cores[0]
            ones0 = jnp.ones(self.dims[0])
            L0 = jnp.einsum("xa,x,xb->ab", c0, ones0, c0)
            L1 = jnp.einsum("xa,x,xb->ab", c0, g[0], c0)
            L2 = jnp.einsum("xa,x,xb->ab", c0, g[0] ** 2, c0)
        else:
            def step(L, c, w):
                return jnp.einsum("a,axc,x->c", L, c, w)
            def fin(L, c, w):
                return jnp.einsum("a,ax,x->", L, c, w)
            c0 = cores[0]
            ones0 = jnp.ones(self.dims[0])
            L0 = jnp.einsum("xa,x->a", c0, ones0)
            L1 = jnp.einsum("xa,x->a", c0, g[0])
            L2 = jnp.einsum("xa,x->a", c0, g[0] ** 2)

        for i in range(1, self.n - 1):
            c, gi = cores[i], g[i]
            ones = jnp.ones(self.dims[i])
            t0L0 = step(L0, c, ones)
            nL0 = t0L0
            nL1 = step(L1, c, ones) + step(L0, c, gi)
            nL2 = step(L2, c, ones) + 2.0 * step(L1, c, gi) + step(L0, c, gi ** 2)
            s = jnp.sqrt(jnp.sum(nL0 ** 2)) + _EPS      # scale all three together
            L0, L1, L2 = nL0 / s, nL1 / s, nL2 / s

        c, gl = cores[-1], g[-1]
        ones = jnp.ones(self.dims[-1])
        M0 = fin(L0, c, ones)
        M1 = fin(L1, c, ones) + fin(L0, c, gl)
        M2 = fin(L2, c, ones) + 2.0 * fin(L1, c, gl) + fin(L0, c, gl ** 2)

        mean = M1 / M0
        var = jnp.clip(M2 / M0 - mean ** 2, a_min=0.0)
        if within_bin:
            var = var + jnp.sum((a ** 2) * (jnp.asarray(self.disc.width) ** 2) / 12.0)
        return mean, var

    # ==================================================================
    # full distribution of  Y = a·X  (partial-sum transfer DP)
    # ==================================================================
    def _projection_pmf(self, cores, a, n_grid: int = 201):
        """Exact pmf of ``Y = a·X`` on a value grid, differentiable in the cores.

        The threshold event ``1[a·X < c]`` does not factorise over sites, so unlike
        moments it cannot be carried by a per-site weight. Instead we carry the
        *distribution of the running partial sum* as an extra axis on the doubled
        (born) / single (nonneg) transfer: state ``(grid, bond[, bond])``. Each
        physical value ``k`` both applies the local transfer and shifts the
        partial-sum axis by ``round(a_i·centre_i(k)/delta)`` grid steps. Exact up to
        the ``delta`` value-grid resolution; ``O(n·d·G·r^{2..3})``.

        Returns ``(y_values (G,), pmf (G,))`` with ``pmf`` summing to 1.
        """
        born = self.kind == "born"
        a = np.asarray(a, np.float64)
        contrib = [a[i] * np.asarray(self.disc.bin_centers(i)) for i in range(self.n)]
        ymin = float(sum(c.min() for c in contrib))
        ymax = float(sum(c.max() for c in contrib))
        if ymax - ymin < 1e-12:                       # degenerate (a≈0): point mass
            return jnp.asarray([ymin]), jnp.asarray([1.0])

        delta = (ymax - ymin) / (n_grid - 1)
        t = [np.rint(c / delta).astype(np.int64) for c in contrib]      # integer steps
        u = [ti - int(ti.min()) for ti in t]                           # per-site >= 0
        base = int(sum(int(ti.min()) for ti in t))
        G = int(sum(int(ui.max()) for ui in u)) + 1
        r = self.r

        def shift(arr, s):     # pad the leading (grid) axis by s, truncate back to G
            pad = [(s, 0)] + [(0, 0)] * (arr.ndim - 1)
            return jnp.pad(arr, pad)[:G]

        c0 = cores[0]
        if born:
            D = jnp.zeros((G, r, r))
            for k in range(self.dims[0]):
                D = D.at[int(u[0][k])].add(jnp.outer(c0[k], c0[k]))
        else:
            D = jnp.zeros((G, r))
            for k in range(self.dims[0]):
                D = D.at[int(u[0][k])].add(c0[k])

        for i in range(1, self.n - 1):
            ci = cores[i]
            Dn = jnp.zeros_like(D)
            for k in range(self.dims[i]):
                if born:
                    M = ci[:, k, :]
                    trans = jnp.einsum("gab,ac,bd->gcd", D, M, M)
                else:
                    trans = jnp.einsum("ga,ac->gc", D, ci[:, k, :])
                Dn = Dn + shift(trans, int(u[i][k]))
            D = Dn

        cl = cores[-1]
        pmf = jnp.zeros((G,))
        for k in range(self.dims[-1]):
            if born:
                v = cl[:, k]
                vals = jnp.einsum("gab,a,b->g", D, v, v)
            else:
                vals = jnp.einsum("ga,a->g", D, cl[:, k])
            pmf = pmf + shift(vals, int(u[-1][k]))

        pmf = pmf / jnp.sum(pmf)
        y = (jnp.arange(G) + base) * delta
        return y, pmf

    def projection_distribution(self, params, a, n_grid: int = 201):
        """``(y_values, pmf)`` of ``Y = a·X`` as numpy arrays (see _projection_pmf)."""
        y, pmf = self._projection_pmf(self._chain_cores(params), a, n_grid)
        return np.asarray(y), np.asarray(pmf)

    def prob_lt(self, params, a, c, n_grid: int = 201):
        """P(a·X < c) — the projected CDF at ``c`` (exact up to grid resolution)."""
        y, pmf = self._projection_pmf(self._chain_cores(params), a, n_grid)
        return float(jnp.sum(jnp.where(y < c, pmf, 0.0)))

    # ==================================================================
    # gauge-invariant amplitude roughness  <psi|L_i|psi> / Z
    # ==================================================================
    def _roughness_matrix(self, d: int, order: int = 2):
        """Discrete roughness metric ``L = D^T D`` on ``d`` bins (see smothing_notes.md).

        ``order=2``: second-difference ``D`` (rows ``(1,-2,1)``), so ``L`` is
        pentadiagonal (interior stencil ``(1,-4,6,-4,1)``) and penalises *curvature*
        (``ker L`` = constants + linear ramps). ``order=1``: first-difference ``D``,
        the tridiagonal path Laplacian penalising *slope* (``ker L`` = constants).
        """
        D = np.zeros((max(d - order, 0), d), np.float32)
        if order == 1:
            for k in range(d - 1):
                D[k, k], D[k, k + 1] = -1.0, 1.0
        elif order == 2:
            for k in range(d - 2):
                D[k, k], D[k, k + 1], D[k, k + 2] = 1.0, -2.0, 1.0
        else:
            raise ValueError("order must be 1 or 2")
        return jnp.asarray(D.T @ D)

    def _log_matrix_contract(self, cores, site_mats: dict):
        """log of the two-copy contraction with a (d,d) coupler at each site.

        The doubled (ket⊗bra) transfer, but where site ``s`` couples the ket physical
        index ``x`` and the bra index ``y`` through ``site_mats[s][x, y]`` instead of
        the diagonal identity ``delta_{xy}`` used for the norm ``<psi|psi>``. With no
        couplers this *is* ``log <psi|psi>`` (the born ``Z``). Inserting ``L`` at one
        site gives ``log <psi|L_i|psi>``. Quadratic in ``psi`` (needs both copies), so
        it is defined for any real cores and is a pure function of the amplitude
        ``psi`` — invariant to MPS gauge, unlike a raw per-core difference.
        """
        def W(site, d):
            m = site_mats.get(site)
            return jnp.eye(d) if m is None else m

        c = cores[0]
        E = jnp.einsum("xa,xy,yb->ab", c, W(0, self.dims[0]), c)
        nrm = jnp.sqrt(jnp.sum(E * E)) + _EPS
        E = E / nrm
        logc = jnp.log(nrm)
        for i in range(1, self.n - 1):
            c = cores[i]
            E = jnp.einsum("ab,axc,xy,byd->cd", E, c, W(i, self.dims[i]), c)
            nrm = jnp.sqrt(jnp.sum(E * E)) + _EPS
            E = E / nrm
            logc = logc + jnp.log(nrm)
        c = cores[-1]
        val = jnp.einsum("ab,ax,xy,by->", E, c, W(self.n - 1, self.dims[-1]), c)
        return logc + jnp.log(jnp.clip(val, a_min=0.0) + _EPS)

    def amplitude_roughness(self, params, sites=None, order: int = 2):
        """Mean over ``sites`` of the amplitude-curvature penalty ``<psi|L_i|psi> / Z``.

        The gauge-invariant smoothness penalty of ``smothing_notes.md``: for each
        site ``i`` it inserts the roughness metric ``L_i = D^T D`` between the ket and
        bra copies of ``psi`` at that site's physical leg and divides by ``<psi|psi>``
        (self-normalising => invariant to ``psi -> c*psi`` and to any MPS gauge).
        Unlike :func:`losses.core_curvature`, which finite-differences the raw cores
        in whatever (non-canonical) gauge the optimiser holds, this is a pure function
        of the represented amplitude, so it cannot be gauged away. Penalises the
        curvature of ``psi`` (per the notes' Born caveat, *not* ``p = psi^2``).
        ``order=2`` curvature, ``order=1`` slope. Differentiable in the cores.
        """
        cores = self._chain_cores(params)
        idxs = range(self.n) if sites is None else list(sites)
        logZ = self._log_matrix_contract(cores, {})
        tot = 0.0
        for i in idxs:
            Lmat = self._roughness_matrix(self.dims[i], order)
            tot = tot + jnp.exp(self._log_matrix_contract(cores, {i: Lmat}) - logZ)
        return tot / len(idxs)

    # ==================================================================
    # Born-rule ancestral sampling (numpy, host)
    # ==================================================================
    def sample_idx(self, params, n_samples: int, seed: int = 0):
        cores = [np.asarray(c) for c in self._chain_cores(params)]
        born = self.kind == "born"
        n, dims = self.n, self.dims
        rng = np.random.default_rng(seed)

        # right environments RE[i] over sites i..n-1 (summed).
        # born: (r,r) matrices; nonneg: (r,) vectors.
        RE = [None] * n
        if born:
            RE[n - 1] = np.einsum("ax,bx->ab", cores[-1], cores[-1])
            for i in range(n - 2, 0, -1):
                RE[i] = np.einsum("axc,bxd,cd->ab", cores[i], cores[i], RE[i + 1])
        else:
            RE[n - 1] = np.sum(cores[-1], axis=1)             # (r,)
            for i in range(n - 2, 0, -1):
                RE[i] = np.einsum("axc,c->ax", cores[i], RE[i + 1]).sum(axis=1)

        def choose(w):
            w = np.clip(w, 0, None)
            s = w.sum()
            return int(rng.choice(len(w), p=(w / s) if s > 0 else None))

        out = np.zeros((n_samples, n), dtype=np.int64)
        for t in range(n_samples):
            c0 = cores[0]
            if born:
                w0 = np.einsum("ka,ab,kb->k", c0, RE[1], c0)
            else:
                w0 = np.einsum("ka,a->k", c0, RE[1])
            k0 = choose(w0)
            out[t, 0] = k0
            l = c0[k0]                                        # (r,)
            for i in range(1, n - 1):
                ci = cores[i]
                lk = np.einsum("a,akc->kc", l, ci)            # (d_i, r)
                if born:
                    w = np.einsum("kc,cd,kd->k", lk, RE[i + 1], lk)
                else:
                    w = np.einsum("kc,c->k", lk, RE[i + 1])
                ki = choose(w)
                out[t, i] = ki
                l = lk[ki]
            cl = cores[-1]                                    # (r, d)
            proj = np.einsum("a,ak->k", l, cl)
            w = proj ** 2 if born else proj
            out[t, n - 1] = choose(w)
        return out

    def sample(self, params, n_samples: int, seed: int = 0, jitter: bool = True):
        idx = self.sample_idx(params, n_samples, seed=seed)
        rng = np.random.default_rng(seed + 1) if jitter else None
        return self.disc.to_value(idx, rng=rng)
