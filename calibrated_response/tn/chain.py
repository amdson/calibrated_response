"""Tensor-chain density model over discretised continuous variables.

A linear tensor train (matrix product state) over ``n`` categorical *sites*
(the discretised variables). Two model kinds share almost all machinery:

* ``"born"``     — Born machine, ``p(x) = psi(x)^2 / Z`` with real cores that may
                   be negative (interference => more expressive than nonneg TT).
* ``"nonneg"``   — plain nonnegative tensor train, ``p(x) = psi(x) / Z`` with
                   nonnegative cores (an HMM in disguise). No interference.

Cores (bond dimension ``r``, site dims ``d_i``):

    core[0]      : (d_0, r)
    core[i]      : (r, d_i, r)         0 < i < n-1
    core[n-1]    : (r, d_{n-1})

Everything the network needs is integer site values; continuous <-> integer is
handled by :class:`~calibrated_response.tn.discretize.Discretizer`.

Exposed queries (all exact, all differentiable in the cores except sampling):

    log_prob_idx        log p over full integer configs
    marginal_log_prob   log p over an observed subset (rest summed out)
    prob_gt             P(X_v > threshold) via exact single-site marginals
    sample_idx          Born-rule ancestral sampling (numpy, host)

Fitting is delegated to a small registry of *backends* (:data:`FIT_BACKENDS`) so
different optimisation / decomposition processes can be swapped freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .discretize import ContinuousVar, Discretizer

_EPS = 1e-30


# ======================================================================
# core parameterisation
# ======================================================================

def _apply_kind(cores, kind):
    """Map raw (unconstrained) cores to the cores used in contractions."""
    if kind == "born":
        return cores                      # real, may be negative
    if kind == "nonneg":
        return [jnp.abs(c) for c in cores]  # nonnegative TT
    raise ValueError(f"unknown model kind {kind!r}")


# ======================================================================
# model
# ======================================================================

class TensorChain:
    """MPS density model over discretised continuous variables."""

    def __init__(self, vars: Sequence[ContinuousVar], bond_dim: int = 8,
                 kind: str = "born"):
        self.disc = Discretizer(vars)
        self.n = self.disc.n_sites
        self.dims = self.disc.dims
        self.r = int(bond_dim)
        self.kind = kind
        if self.n < 2:
            raise ValueError("need at least 2 sites")

    # ---- parameters -------------------------------------------------
    def init_params(self, seed: int = 0, scale: float = 0.3,
                    init: str = "random", noise: float = 0.02):
        """Initialise the cores.

        ``init="random"`` (default): i.i.d. normal cores.
        ``init="uniform"``: constant cores (=> psi(x) independent of x, i.e. the
        model starts as the *uniform* distribution — flat marginals) plus a small
        ``noise`` perturbation to break the bond-channel symmetry so gradients can
        use the full bond dimension. A natural max-entropy starting point for
        constraint fitting.
        """
        rng = np.random.default_rng(seed)
        r, dims, n = self.r, self.dims, self.n

        def make(shape):
            if init == "uniform":
                return np.ones(shape, np.float32) + rng.normal(0, noise, size=shape)
            if init == "random":
                return rng.normal(0, scale, size=shape)
            raise ValueError(f"unknown init {init!r}")

        cores = [jnp.asarray(make((dims[0], r)), jnp.float32)]
        for i in range(1, n - 1):
            cores.append(jnp.asarray(make((r, dims[i], r)), jnp.float32))
        cores.append(jnp.asarray(make((r, dims[-1])), jnp.float32))
        # a small positive offset for the nonneg kind so |.| doesn't kill grads
        if self.kind == "nonneg" and init == "random":
            cores = [c + scale for c in cores]
        return {"cores": cores}

    def _cores(self, params):
        return _apply_kind(params["cores"], self.kind)

    # ---- forward: log|psi| over a batch of integer configs ----------
    def _log_abs_psi(self, cores, X):
        """X: (B, n) int -> (B,) log|psi(x)| (log psi for nonneg)."""
        h = cores[0][X[:, 0]]                                  # (B, r)
        logabs = jnp.zeros(X.shape[0])
        for i in range(1, self.n - 1):
            M = cores[i][:, X[:, i], :]                        # (r, B, r)
            h = jnp.einsum("ba,abc->bc", h, M)                 # (B, r)
            nrm = jnp.sqrt(jnp.sum(h * h, axis=1, keepdims=True)) + _EPS
            h = h / nrm
            logabs = logabs + jnp.log(nrm[:, 0])
        gl = cores[-1][:, X[:, -1]]                            # (r, B)
        s = jnp.einsum("ba,ab->b", h, gl)                      # (B,)
        return logabs + jnp.log(jnp.abs(s) + _EPS)

    # ---- log normalisation Z ---------------------------------------
    def _log_Z(self, cores):
        if self.kind == "born":
            return self._log_Z_doubled(cores)
        return self._log_Z_single(cores)

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

    # ---- joint Rényi-2 (collision) entropy --------------------------
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
        cores = self._cores(params)
        if self.kind == "born":
            return 2.0 * self._log_Z_doubled(cores) - self._log_Z_quadrupled(cores)
        return 2.0 * self._log_Z_single(cores) - self._log_Z_doubled(cores)

    # ---- public: full-config log prob ------------------------------
    def log_prob_idx(self, params, X):
        cores = self._cores(params)
        X = jnp.asarray(X, jnp.int32)
        factor = 2.0 if self.kind == "born" else 1.0
        return factor * self._log_abs_psi(cores, X) - self._log_Z(cores)

    # ---- exact subset marginal -------------------------------------
    def marginal_log_prob_idx(self, params, observed: dict):
        """log p(X_S = x_S) with unobserved sites summed out.

        ``observed`` maps site index -> integer value. Uses the doubled (born)
        or single (nonneg) transfer, rank-1 clamped at observed sites.
        """
        cores = self._cores(params)
        born = self.kind == "born"

        def site0():
            c = cores[0]
            if 0 in observed:
                v = c[observed[0]]
                return jnp.outer(v, v) if born else v
            return jnp.einsum("xa,xb->ab", c, c) if born else jnp.sum(c, axis=0)

        L = site0()
        for i in range(1, self.n - 1):
            c = cores[i]
            if i in observed:
                m = c[:, observed[i], :]                       # (r, r)
                L = jnp.einsum("ab,ac,bd->cd", L, m, m) if born else jnp.einsum("a,ac->c", L, m)
            else:
                L = jnp.einsum("ab,axc,bxd->cd", L, c, c) if born else jnp.einsum("a,axc->c", L, c)
        c = cores[-1]
        if (self.n - 1) in observed:
            v = c[:, observed[self.n - 1]]
            val = jnp.einsum("ab,a,b->", L, v, v) if born else jnp.einsum("a,a->", L, v)
        else:
            val = jnp.einsum("ab,ax,bx->", L, c, c) if born else jnp.einsum("a,ax->", L, c)
        return jnp.log(val + _EPS) - self._log_Z(cores)

    # ---- differentiable event / conditional probabilities ----------
    # An *event* is a dict {site: mask (d_site,)}, mask in [0,1] selecting which
    # bins are "in" the event; unlisted sites are fully summed. These are the
    # queries the constraint losses (spec §6) are built on.

    def _log_event_contract(self, cores, masks):
        """log of the (nonnegative) masked chain contraction. masks: {site: (d,)}."""
        born = self.kind == "born"

        def w(site, d):
            m = masks.get(site)
            return jnp.ones((d,)) if m is None else m

        c = cores[0]
        L = (jnp.einsum("xa,x,xb->ab", c, w(0, self.dims[0]), c) if born
             else jnp.einsum("xa,x->a", c, w(0, self.dims[0])))
        logc = jnp.log(jnp.sqrt(jnp.sum(L * L)) + _EPS)
        L = L / (jnp.sqrt(jnp.sum(L * L)) + _EPS)
        for i in range(1, self.n - 1):
            c = cores[i]
            L = (jnp.einsum("ab,axc,x,bxd->cd", L, c, w(i, self.dims[i]), c) if born
                 else jnp.einsum("a,axc,x->c", L, c, w(i, self.dims[i])))
            nrm = jnp.sqrt(jnp.sum(L * L)) + _EPS
            L = L / nrm
            logc = logc + jnp.log(nrm)
        c = cores[-1]
        val = (jnp.einsum("ab,ax,x,bx->", L, c, w(self.n - 1, self.dims[-1]), c) if born
               else jnp.einsum("a,ax,x->", L, c, w(self.n - 1, self.dims[-1])))
        return logc + jnp.log(val + _EPS)

    def threshold_mask(self, site: int, threshold: float, above: bool = True):
        """Bin mask for the event ``X_site > threshold`` (or ``< threshold``)."""
        m = self.disc.bins_above(site, threshold).astype(np.float32)
        return jnp.asarray(m if above else 1.0 - m)

    def event_prob(self, params, event: dict):
        """P(event) — differentiable. ``event`` is {site: mask}."""
        cores = self._cores(params)
        return jnp.exp(self._log_event_contract(cores, event)
                       - self._log_event_contract(cores, {}))

    def cond_prob(self, params, event: dict, given: dict):
        """P(event | given) — differentiable. Z cancels, so no normalisation needed."""
        cores = self._cores(params)
        merged = dict(given)
        for s, m in event.items():
            merged[s] = merged[s] * m if s in merged else m
        return jnp.exp(self._log_event_contract(cores, merged)
                       - self._log_event_contract(cores, given))

    # ---- one open site (rest clamped / summed) ---------------------
    def _open_contract(self, cores, clamped: dict, open_site: int):
        """Unnormalised marginal vector over ``open_site`` (rest clamped/summed).

        ``clamped`` maps site -> fixed value; all other non-open sites are summed
        out. Returns a length ``d[open_site]`` vector whose sum over the open site
        equals Z, so ``vec / vec.sum()`` is the (conditional-free) marginal.
        Boundaries handled; used for exact 1-D and 2-D marginals / plots.
        """
        born = self.kind == "born"

        # ---- site 0 (right bond r); env carries a leading M dim (1 until open)
        c = cores[0]
        if open_site == 0:
            env = jnp.einsum("ka,kb->kab", c, c) if born else c            # (d0,r[,r])
        else:
            if 0 in clamped:
                v = c[clamped[0]]
                e = jnp.outer(v, v) if born else v
            else:
                e = jnp.einsum("xa,xb->ab", c, c) if born else jnp.sum(c, axis=0)
            env = e[None]                                                  # (1,r[,r])

        # ---- middle sites
        for s in range(1, self.n - 1):
            c = cores[s]
            if s == open_site:
                env = (jnp.einsum("nab,akc,bkd->kcd", env, c, c) if born
                       else jnp.einsum("na,akc->kc", env, c))
            elif s in clamped:
                m = c[:, clamped[s], :]
                env = (jnp.einsum("nab,ac,bd->ncd", env, m, m) if born
                       else jnp.einsum("na,ac->nc", env, m))
            else:
                env = (jnp.einsum("nab,axc,bxd->ncd", env, c, c) if born
                       else jnp.einsum("na,axc->nc", env, c))

        # ---- last site (left bond r)
        c = cores[-1]
        last = self.n - 1
        if open_site == last:
            res = jnp.einsum("nab,ak,bk->k", env, c, c) if born else jnp.einsum("na,ak->k", env, c)
        elif last in clamped:
            v = c[:, clamped[last]]
            res = jnp.einsum("nab,a,b->n", env, v, v) if born else jnp.einsum("na,a->n", env, v)
        else:
            res = jnp.einsum("nab,ax,bx->n", env, c, c) if born else jnp.einsum("na,ax->n", env, c)
        return res

    def site_marginal(self, params, site: int):
        """Exact per-bin mass vector p(X_site = k), shape (d_site,)."""
        vec = self._open_contract(self._cores(params), {}, site)
        return vec / jnp.sum(vec)

    def pair_marginal(self, params, i: int, j: int):
        """Exact joint marginal table ``p(X_i = ., X_j = .)`` of shape (d_i, d_j).

        Rows index ``X_i``, columns index ``X_j`` (in original bin order).
        """
        if i == j:
            raise ValueError("pair_marginal needs two distinct sites")
        cores = self._cores(params)
        lo, hi = (i, j) if i < j else (j, i)
        rows = [self._open_contract(cores, {lo: k}, hi) for k in range(self.dims[lo])]
        M = jnp.stack(rows)                       # (d_lo, d_hi), unnormalised
        M = M / jnp.sum(M)
        return M if i < j else M.T

    def joint_marginal(self, params, sites, masks=None, normalize: bool = True):
        """Exact joint marginal ``p(X_sites)`` over an arbitrary set of sites.

        Generalises :meth:`site_marginal` (1 site) and :meth:`pair_marginal`
        (2 sites) to any tuple, in a single left-to-right sweep (no clamping
        loop). Returns an array whose axes follow ``sorted(sites)`` in original
        bin order. Differentiable in ``params``.

        ``masks`` (optional) is ``{site: (d_site,) weight}`` applied to *summed*
        (non-open) sites — a ``0/1`` mask conditions the joint on an event, e.g.
        ``masks={a: threshold_mask(a, 50)}`` restricts to ``X_a > 50``. Masking an
        open site is ignored (open sites are the targets).

        ``normalize=True`` (default) rescales to sum 1: the plain joint when
        unmasked, or the **event-conditional** joint ``p(sites | event)`` when
        masked. Use this in any *loss* term — the raw masked table carries an
        arbitrary positive factor (a ``Z``-like scale plus per-site rescalings
        applied to keep the sweep inside float32 range), which nothing pins, so
        an un-normalised loss could be gamed by rescaling the cores.
        ``normalize=False`` returns the arbitrarily-scaled contraction; only use
        it inside a *single-call* ratio where the factor cancels (see
        :meth:`cond_expectation`) — never compare raw tables across calls.

        Cost is ``O(n · r^2 · prod(d_open))``; the table grows as ``prod(d)``
        over the opened sites, so keep the open set small (1--3).
        """
        if isinstance(sites, int):
            sites = (sites,)
        sites = tuple(sorted(int(s) for s in sites))
        if len(set(sites)) != len(sites):
            raise ValueError("joint_marginal needs distinct sites")
        cores = self._cores(params)
        born = self.kind == "born"
        openset = set(sites)

        def scaled(site, core):
            """Core with a summed site's mask folded onto its physical axis (once)."""
            m = None if masks is None else masks.get(site)
            if m is None or site in openset:
                return core
            m = jnp.asarray(m, core.dtype)
            if site == 0:
                return core * m[:, None]              # (d, r)
            if site == self.n - 1:
                return core * m[None, :]              # (r, d)
            return core * m[None, :, None]            # (r, d, r)

        def rescale(E):
            """Keep the running transfer O(1) so long / high-dim chains don't
            overflow float32 (cf. the per-step norm in :meth:`_log_Z_doubled`).
            The output is scale-invariant, so a stop-gradient factor is exact."""
            s = jax.lax.stop_gradient(jnp.max(jnp.abs(E))) + _EPS
            return E / s

        _ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        def take(k, used):
            out = []
            for ch in _ALPHA:
                if ch not in used:
                    out.append(ch); used.add(ch)
                    if len(out) == k:
                        return out
            raise RuntimeError("joint_marginal: ran out of einsum letters")

        o = ""  # letters currently labelling E's open axes (in site order)

        # ---- site 0 (core (d, r)) ----
        c = cores[0]
        used = set()
        if born:
            ka, kb = take(2, used); (p,) = take(1, used)
            if 0 in openset:
                E = jnp.einsum(f"{p}{ka},{p}{kb}->{ka}{kb}{p}", c, c); o += p
            else:
                E = jnp.einsum(f"{p}{ka},{p}{kb}->{ka}{kb}", scaled(0, c), c)
        else:
            (ka,) = take(1, used); (p,) = take(1, used)
            if 0 in openset:
                E = jnp.einsum(f"{p}{ka}->{ka}{p}", c); o += p
            else:
                E = jnp.einsum(f"{p}{ka}->{ka}", scaled(0, c))
        E = rescale(E)

        # ---- middle sites (core (r, d, r)) ----
        for s in range(1, self.n - 1):
            c = cores[s]
            used = set(o)
            if born:
                ea, eb = take(2, used); (p,) = take(1, used); na, nb = take(2, used)
                if s in openset:
                    E = jnp.einsum(f"{ea}{eb}{o},{ea}{p}{na},{eb}{p}{nb}->{na}{nb}{o}{p}", E, c, c); o += p
                else:
                    E = jnp.einsum(f"{ea}{eb}{o},{ea}{p}{na},{eb}{p}{nb}->{na}{nb}{o}", E, scaled(s, c), c)
            else:
                (ea,) = take(1, used); (p,) = take(1, used); (na,) = take(1, used)
                if s in openset:
                    E = jnp.einsum(f"{ea}{o},{ea}{p}{na}->{na}{o}{p}", E, c); o += p
                else:
                    E = jnp.einsum(f"{ea}{o},{ea}{p}{na}->{na}{o}", E, scaled(s, c))
            E = rescale(E)

        # ---- last site (core (r, d)) ----
        c = cores[-1]
        last = self.n - 1
        used = set(o)
        if born:
            ea, eb = take(2, used); (p,) = take(1, used)
            if last in openset:
                E = jnp.einsum(f"{ea}{eb}{o},{ea}{p},{eb}{p}->{o}{p}", E, c, c); o += p
            else:
                E = jnp.einsum(f"{ea}{eb}{o},{ea}{p},{eb}{p}->{o}", E, scaled(last, c), c)
        else:
            (ea,) = take(1, used); (p,) = take(1, used)
            if last in openset:
                E = jnp.einsum(f"{ea}{o},{ea}{p}->{o}{p}", E, c); o += p
            else:
                E = jnp.einsum(f"{ea}{o},{ea}{p}->{o}", E, scaled(last, c))

        return E / jnp.sum(E) if normalize else E

    def cond_expectation(self, params, site, given, f=None):
        """Exact conditional expectation ``E[f(X_site) | given]`` — differentiable.

        ``given`` is an event ``{site: mask}`` (see :meth:`threshold_mask`), and
        ``f`` is a callable applied to ``site``'s bin centres (``None`` => identity,
        i.e. the conditional *mean*). Reads the single masked sub-marginal
        ``p(X_site, given)`` off one contraction; the normaliser ``Z`` cancels in
        the ratio, so no separate ``Z`` is needed.
        """
        v = self.joint_marginal(params, (site,), masks=given, normalize=False)  # (d_site,)
        cen = self.disc.bin_centers(site)
        fx = jnp.asarray(cen if f is None else f(cen), jnp.float32)
        return (fx @ v) / (jnp.sum(v) + _EPS)

    def marginal_kl(self, params, sites, ref, direction: str = "forward"):
        """KL divergence between the exact joint marginal over ``sites`` and ``ref``.

        ``sites`` is an int or a tuple (see :meth:`joint_marginal`); ``ref`` is a
        reference distribution broadcastable to the joint-marginal shape (it need
        not be normalised — it is renormalised here). ``direction``:

        - ``"forward"`` (default): ``KL(ref || p)`` — mode-covering; heavily
          penalises the model placing ~0 mass where ``ref`` has mass. This is the
          right choice for *fitting the model to a target* distribution.
        - ``"reverse"``: ``KL(p || ref)`` — mode-seeking.

        Differentiable in ``params``; ``>= 0``, ``0`` iff ``p == ref``.
        """
        p = self.joint_marginal(params, sites)
        r = jnp.asarray(ref, jnp.float32)
        r = r / jnp.sum(r)
        if direction == "reverse":
            return jnp.sum(p * (jnp.log(p + _EPS) - jnp.log(r + _EPS)))
        elif direction == "forward":
            return jnp.sum(r * (jnp.log(r + _EPS) - jnp.log(p + _EPS)))
        raise ValueError("direction must be 'forward' or 'reverse'")

    def expectation(self, params, values):
        """Exact expected value of a function of one or more variables.

        Two forms:

        - **separable** — ``values`` is a dict ``{site: g}`` with each ``g`` a
          length-``d_site`` array of the function's value on each bin. Returns
          ``E[sum_i g_i(X_i)]`` (e.g. ``{i: bin_centers(i)}`` is the mean of
          ``X_i``; ``{i: a_i * bin_centers(i) for i in ...}`` is ``E[a·X]``).
        - **joint** — ``values`` is a tuple ``(sites, table)`` with ``table``
          shaped like the joint marginal over ``sites``. Returns
          ``E[g(X_sites)]`` for a genuinely non-separable ``g``.

        Differentiable in ``params``.
        """
        if isinstance(values, tuple):
            sites, table = values
            p = self.joint_marginal(params, sites)
            return jnp.sum(p * jnp.asarray(table, jnp.float32))
        tot = 0.0
        for site, g in values.items():
            p = self.site_marginal(params, site)
            tot = tot + jnp.sum(p * jnp.asarray(g, jnp.float32))
        return tot

    def prob_gt(self, params, site: int, threshold: float):
        """Exact P(X_site > threshold) from the site marginal."""
        mass = np.asarray(self.site_marginal(params, site))
        return self.disc.prob_gt(site, threshold, mass)

    # ---- exact moments of a weighted sum Y = a·X -------------------
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
        cores = self._cores(params)
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

    # ---- full distribution of Y = a·X (partial-sum transfer DP) ----
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
        y, pmf = self._projection_pmf(self._cores(params), a, n_grid)
        return np.asarray(y), np.asarray(pmf)

    def prob_lt(self, params, a, c, n_grid: int = 201):
        """P(a·X < c) — the projected CDF at ``c`` (exact up to grid resolution)."""
        y, pmf = self._projection_pmf(self._cores(params), a, n_grid)
        return float(jnp.sum(jnp.where(y < c, pmf, 0.0)))

    # ---- gauge-invariant amplitude roughness  <psi|L_i|psi> / Z ----
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
        cores = self._cores(params)
        idxs = range(self.n) if sites is None else list(sites)
        logZ = self._log_matrix_contract(cores, {})
        tot = 0.0
        for i in idxs:
            Lmat = self._roughness_matrix(self.dims[i], order)
            tot = tot + jnp.exp(self._log_matrix_contract(cores, {i: Lmat}) - logZ)
        return tot / len(idxs)

    # ---- Born-rule ancestral sampling (numpy, host) ----------------
    def sample_idx(self, params, n_samples: int, seed: int = 0):
        cores = [np.asarray(c) for c in _apply_kind(params["cores"], self.kind)]
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

    # ---- optimisation ----------------------------------------------
    def optimize(self, loss_fn, backend: str = "adam", seed: int = 0,
                 init=None, **kw):
        """Minimise an arbitrary ``loss_fn(params) -> scalar`` (spec §6)."""
        params = self.init_params(seed=seed) if init is None else init
        return FIT_BACKENDS[backend](loss_fn, params, **kw)

    def nll_loss(self, Xi):
        """NLL of integer data — forward-KL to a sample."""
        Xi = jnp.asarray(Xi, jnp.int32)
        return lambda p: -jnp.mean(self.log_prob_idx(p, Xi))

    def constraint_loss(self, constraints, weight_reg=0.0):
        """Squared-error loss on marginal / conditional constraints (spec §6).

        ``constraints`` is a list of any of
            ("prob", event_dict, target)                       squared-error
            ("cond", event_dict, given_dict, target)           squared-error
            ("kl",   sites, ref[, weight])                      KL(ref || p_sites)
            ("expect", values, target[, weight])               squared-error
            ("cond_expect", site, given, target[, weight])     squared-error
        where event/given dicts map site -> bin mask (see :meth:`threshold_mask`),
        ``sites``/``ref`` are as in :meth:`marginal_kl`, and ``values`` is as in
        :meth:`expectation`. ``kl``/``expect`` take an optional trailing weight
        (default ``1.0``) since they live on a different scale than the ``[0,1]``
        probability residuals. ``weight_reg`` adds a small pull of the cores toward
        small magnitude (a mild anti-degeneracy prior on the under-determined part).
        """
        def loss(p):
            tot = 0.0
            for cst in constraints:
                kind = cst[0]
                if kind == "prob":
                    _, ev, tg = cst
                    tot = tot + (self.event_prob(p, ev) - tg) ** 2
                elif kind == "cond":
                    _, ev, gv, tg = cst
                    tot = tot + (self.cond_prob(p, ev, gv) - tg) ** 2
                elif kind == "kl":
                    sites, ref = cst[1], cst[2]
                    w = cst[3] if len(cst) > 3 else 1.0
                    tot = tot + w * self.marginal_kl(p, sites, ref)
                elif kind == "expect":
                    values, tg = cst[1], cst[2]
                    w = cst[3] if len(cst) > 3 else 1.0
                    tot = tot + w * (self.expectation(p, values) - tg) ** 2
                elif kind == "cond_expect":
                    site, given, tg = cst[1], cst[2], cst[3]
                    w = cst[4] if len(cst) > 4 else 1.0
                    tot = tot + w * (self.cond_expectation(p, site, given) - tg) ** 2
                else:
                    raise ValueError(f"unknown constraint kind {kind!r}")
            if weight_reg:
                tot = tot + weight_reg * sum(jnp.mean(c ** 2) for c in p["cores"])
            return tot
        return loss

    def fit(self, X_continuous, backend: str = "adam", seed: int = 0, **kw):
        """Convenience: fit by NLL to a continuous dataset (discretised first)."""
        Xi = self.disc.to_index(np.asarray(X_continuous))
        return self.optimize(self.nll_loss(Xi), backend=backend, seed=seed, **kw)

    def sample(self, params, n_samples: int, seed: int = 0, jitter: bool = True):
        idx = self.sample_idx(params, n_samples, seed=seed)
        rng = np.random.default_rng(seed + 1) if jitter else None
        return self.disc.to_value(idx, rng=rng)

    def nll(self, params, Xi):
        return float(-jnp.mean(self.log_prob_idx(params, Xi)))


# ======================================================================
# fit backends  (pluggable "decomposition processes")
# ======================================================================

def _fit_adam(loss_fn, params, steps=1500, lr=5e-2, grad_clip=5.0, log_every=0):
    import optax

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    state = opt.init(params)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    history = []
    for it in range(steps):
        loss, g = vg(params)
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  [adam] step {it:5d}  loss {float(loss):.6f}")
    return params, history


def _fit_lbfgs(loss_fn, params, maxiter=800, log_every=0):
    """Scipy L-BFGS-B over flattened cores (a genuinely different process)."""
    from scipy.optimize import minimize
    from jax.flatten_util import ravel_pytree

    x0, unravel = ravel_pytree(params)
    x0 = np.asarray(x0, np.float64)

    vg = jax.jit(jax.value_and_grad(lambda flat: loss_fn(unravel(flat))))
    history = []

    def fun(flat):
        v, g = vg(jnp.asarray(flat, jnp.float32))
        history.append(float(v))
        return float(v), np.asarray(g, np.float64)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9})
    if log_every:
        print(f"  [lbfgs] {res.nit} iters  loss {float(res.fun):.6f}  ({res.message})")
    return unravel(jnp.asarray(res.x, jnp.float32)), history


FIT_BACKENDS: dict[str, Callable] = {
    "adam": _fit_adam,
    "lbfgs": _fit_lbfgs,
}
