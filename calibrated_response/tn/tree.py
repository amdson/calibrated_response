"""Tensor-*tree* density model over discretised continuous variables.

A tree tensor network (TTN) that generalises :class:`~calibrated_response.tn.chain.TensorChain`
from a line to an arbitrary tree: every variable is still a single *node* carrying
one categorical physical (bin) index, but the virtual bonds follow the edges of a
user-supplied tree instead of a 1-D chain. The chain is the special case where the
tree is a path.

Why a tree. On a chain the only cheap couplings are between adjacent sites, so a
latent target has to be placed *next to* the data variable it governs (see the
robust-constraint notebooks). A tree lets you wire each latent **directly** to its
data variable (e.g. a star with ``X`` at the centre) no matter how many there are,
and the contraction stays exact and cheap because a tree has no loops.

Node cores (bond dim ``r``, site dims ``d_i``, ``deg_i`` = number of tree
neighbours of site ``i``):

    core[i] : (d_i, r, r, ..., r)          # physical axis first, then one bond
                                           # axis per neighbour, in the order of
                                           # ``self.adj[i]`` (sorted neighbours)

Two model kinds, exactly as the chain:

* ``"born"``   — ``p(x) = psi(x)^2 / Z`` with real cores (interference).
* ``"nonneg"`` — ``p(x) = psi(x)  / Z`` with nonnegative cores.

All queries are exact contractions, computed by leaves->root message passing
(rooted at node 0) — the tree analogue of the chain's left->right transfer sweep.
Continuous <-> integer is handled by the shared
:class:`~calibrated_response.tn.discretize.Discretizer`.

**Ported subset (deliberately minimal).** ``log_prob_idx``, ``marginal_log_prob_idx``,
``event_prob`` / ``cond_prob``, ``joint_marginal`` (+ ``site_marginal`` /
``pair_marginal``), ``expectation``, ``cond_expectation``, ``marginal_kl``, plus the
fitting glue (``constraint_loss`` / ``optimize`` / ``fit``). This is enough to run
every loss in :mod:`losses` that is defined purely on marginals / expectations
(constraint SSE, the robust / belief / on-off constraints, KL regularizers, ...).

**Not ported yet** (the "complicated" losses): joint Rényi-2 / H2 entropy, the
gauge-invariant ``amplitude_roughness``, ``linear_moments`` and the projection PMF,
and the core-difference smoothness penalties. Calling those against a tree will
raise ``AttributeError`` until they are added.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .discretize import ContinuousVar, Discretizer
from .backends import FIT_BACKENDS, _apply_kind

_EPS = 1e-30
_ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ======================================================================
# tree structure helpers
# ======================================================================

def path_edges(n: int):
    """Edge list of a path (chain) over ``n`` nodes — the chain topology."""
    return [(i, i + 1) for i in range(n - 1)]


def star_edges(n: int, center: int = 0):
    """Edge list of a star: every node bonded directly to ``center``.

    The natural layout for one data variable plus several latent targets — each
    latent gets its own cheap bond to ``center`` with no ordering hack.
    """
    return [(center, j) for j in range(n) if j != center]


def _build_adjacency(n: int, edges):
    """Validate that ``edges`` is a tree on ``n`` nodes and return sorted adjacency."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        a, b = int(a), int(b)
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"bad edge {(a, b)} for {n} nodes")
        adj[a].append(b)
        adj[b].append(a)
    if len(edges) != n - 1:
        raise ValueError(f"a tree on {n} nodes needs {n - 1} edges, got {len(edges)}")
    # connectivity (a graph with n-1 edges that is connected is necessarily a tree)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    if len(seen) != n:
        raise ValueError("edges do not form a connected tree")
    return [sorted(a) for a in adj]


def _alloc(used: set) -> str:
    for ch in _ALPHA:
        if ch not in used:
            used.add(ch)
            return ch
    raise RuntimeError("tree contraction: ran out of einsum letters")


# ======================================================================
# model
# ======================================================================

class TensorTree:
    """Tree tensor-network density model over discretised continuous variables."""

    def __init__(self, vars: Sequence[ContinuousVar], edges=None, bond_dim: int = 8,
                 kind: str = "born"):
        self.disc = Discretizer(vars)
        self.n = self.disc.n_sites
        self.dims = self.disc.dims
        self.r = int(bond_dim)
        self.kind = kind
        if self.n < 1:
            raise ValueError("need at least 1 site")
        self.edges = list(path_edges(self.n) if edges is None else
                          [(int(a), int(b)) for a, b in edges])
        self.adj = _build_adjacency(self.n, self.edges)
        self.deg = [len(a) for a in self.adj]

    # ---- parameters -------------------------------------------------
    def init_params(self, seed: int = 0, scale: float = 0.3,
                    init: str = "random", noise: float = 0.02):
        """Initialise the node cores (see :meth:`TensorChain.init_params`)."""
        rng = np.random.default_rng(seed)

        def make(shape):
            if init == "uniform":
                return np.ones(shape, np.float32) + rng.normal(0, noise, size=shape)
            if init == "random":
                return rng.normal(0, scale, size=shape)
            raise ValueError(f"unknown init {init!r}")

        cores = []
        for i in range(self.n):
            shape = (self.dims[i],) + (self.r,) * self.deg[i]
            cores.append(jnp.asarray(make(shape), jnp.float32))
        if self.kind == "nonneg" and init == "random":
            cores = [c + scale for c in cores]
        return {"cores": cores}

    def _cores(self, params):
        return _apply_kind(params["cores"], self.kind)

    # ==================================================================
    # contraction engine  (leaves -> root message passing, rooted at 0)
    # ==================================================================
    #
    # Every query below is one contraction of the tree in which each site's
    # physical index is either
    #   * summed / masked   — folded with a weight vector ``w_i`` (ones = plain
    #                          sum; a 0/1 mask = an event; a one-hot = a clamp), or
    #   * open              — kept as an output axis (a marginal target).
    # For ``born`` the contraction is doubled (ket (x) tied to bra (x): the
    # physical index is always shared, so a single letter suffices), giving
    # ``sum_x w(x) psi_ket psi_bra``; for ``nonneg`` it is the single layer
    # ``sum_x w(x) psi``.  ``_run`` returns ``(log_scale, arr, open_ids)`` where
    # ``arr`` carries a global positive factor ``exp(log_scale)`` pulled out for
    # float32 stability (cancels under any normalisation / same-call ratio).

    def _run(self, cores, born, weights, open_sites):
        """Contract the tree; return ``(log_scale, arr, open_ids)`` (see above)."""

        def rec(i, parent):
            children = [nb for nb in self.adj[i] if nb != parent]
            subs = [rec(c, i) for c in children]
            log_scale = 0.0
            for ls, _, _ in subs:
                log_scale = log_scale + ls

            used = set()
            p = _alloc(used)                                   # physical letter
            kbond = {nb: _alloc(used) for nb in self.adj[i]}   # ket bond per nbr
            bbond = {nb: _alloc(used) for nb in self.adj[i]} if born else {}

            inputs, operands = [], []
            inputs.append(p + "".join(kbond[nb] for nb in self.adj[i]))
            operands.append(cores[i])
            if born:
                inputs.append(p + "".join(bbond[nb] for nb in self.adj[i]))
                operands.append(cores[i])

            open_letters, open_ids = [], []
            if i in open_sites:
                open_letters.append(p)
                open_ids.append(i)
            else:
                inputs.append(p)                               # weight factor w_i(x)
                operands.append(weights[i])

            for (ls, carr, cids), c in zip(subs, children):
                col = [_alloc(used) for _ in cids]
                cax = kbond[c] + (bbond[c] if born else "") + "".join(col)
                inputs.append(cax)
                operands.append(carr)
                open_letters.extend(col)
                open_ids.extend(cids)

            if parent is None:
                out = "".join(open_letters)
            else:
                out = kbond[parent] + (bbond[parent] if born else "") + "".join(open_letters)

            arr = jnp.einsum(",".join(inputs) + "->" + out, *operands)
            s = jax.lax.stop_gradient(jnp.max(jnp.abs(arr))) + _EPS
            return log_scale + jnp.log(s), arr / s, open_ids

        return rec(0, None)

    def _contract_full(self, cores, born, weights):
        """Scalar ``log`` of the fully-summed (masked) contraction — no open axes."""
        log_scale, arr, _ = self._run(cores, born, weights, set())
        return log_scale + jnp.log(jnp.clip(arr, a_min=0.0) + _EPS)

    def _contract_open(self, cores, born, weights, open_sites):
        """Unnormalised marginal table over ``sorted(open_sites)`` (drops log_scale).

        The dropped ``exp(log_scale)`` is a global positive factor, so this is only
        valid inside a normalisation or a single-call ratio (never compared across
        calls) — the same contract as :meth:`TensorChain.joint_marginal`.
        """
        _, arr, open_ids = self._run(cores, born, weights, set(open_sites))
        order = [open_ids.index(s) for s in sorted(open_sites)]
        return jnp.transpose(arr, order)

    def _ones_weights(self):
        return [jnp.ones(self.dims[i]) for i in range(self.n)]

    # ---- forward |psi| over a batch of integer configs (single layer) ----
    def _log_abs_psi(self, cores, X):
        """X: (B, n) int -> (B,) log|psi(x)| (log psi for nonneg).

        Batched leaves->root sweep with every physical index clamped to the data
        value; the batch axis rides along as a spectator index on every factor.
        """

        def rec(i, parent):
            children = [nb for nb in self.adj[i] if nb != parent]
            subs = [rec(c, i) for c in children]
            log_scale = jnp.zeros(X.shape[0])
            for ls, _ in subs:
                log_scale = log_scale + ls

            used = set()
            bt = _alloc(used)                                  # batch letter
            bond = {nb: _alloc(used) for nb in self.adj[i]}

            loc = cores[i][X[:, i]]                            # (B,) + bonds
            inputs = [bt + "".join(bond[nb] for nb in self.adj[i])]
            operands = [loc]
            for (ls, carr), c in zip(subs, children):
                inputs.append(bt + bond[c])
                operands.append(carr)
            out = bt + (bond[parent] if parent is not None else "")
            arr = jnp.einsum(",".join(inputs) + "->" + out, *operands)

            if parent is not None:                            # rescale the message
                s = jax.lax.stop_gradient(jnp.max(jnp.abs(arr), axis=1, keepdims=True)) + _EPS
                return log_scale + jnp.log(s[:, 0]), arr / s
            return log_scale, arr                             # root: (B,) scalar

        log_scale, arr = rec(0, None)
        return log_scale + jnp.log(jnp.abs(arr) + _EPS)

    # ---- log normalisation Z ---------------------------------------
    def _log_Z(self, cores):
        return self._contract_full(cores, self.kind == "born", self._ones_weights())

    # ---- public: full-config log prob ------------------------------
    def log_prob_idx(self, params, X):
        cores = self._cores(params)
        X = jnp.asarray(X, jnp.int32)
        factor = 2.0 if self.kind == "born" else 1.0
        return factor * self._log_abs_psi(cores, X) - self._log_Z(cores)

    # ---- exact subset marginal (clamp = one-hot weight) ------------
    def marginal_log_prob_idx(self, params, observed: dict):
        """log p(X_S = x_S) with unobserved sites summed out.

        A clamped site is just a one-hot mask, so this is the masked contraction
        (observed sites one-hot, the rest summed) minus ``log Z``.
        """
        cores = self._cores(params)
        weights = self._ones_weights()
        for s, k in observed.items():
            weights[s] = jax.nn.one_hot(int(k), self.dims[s], dtype=jnp.float32)
        return (self._contract_full(cores, self.kind == "born", weights)
                - self._log_Z(cores))

    # ---- differentiable event / conditional probabilities ----------
    def _log_event_contract(self, cores, masks):
        """log of the (nonnegative) masked tree contraction. masks: {site: (d,)}."""
        weights = self._ones_weights()
        for s, m in masks.items():
            weights[s] = jnp.asarray(m, jnp.float32)
        return self._contract_full(cores, self.kind == "born", weights)

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

    # ---- exact joint marginal over an arbitrary set of sites -------
    def joint_marginal(self, params, sites, masks=None, normalize: bool = True):
        """Exact joint marginal ``p(X_sites)`` over an arbitrary set of sites.

        Same contract and semantics as :meth:`TensorChain.joint_marginal`: axes
        follow ``sorted(sites)`` in original bin order; ``masks`` (``{site: (d,)}``)
        weight *summed* (non-open) sites so a 0/1 mask conditions the joint on an
        event; ``normalize=True`` returns ``p`` (or the event-conditional joint when
        masked), ``normalize=False`` the arbitrarily-scaled contraction (only for a
        single-call ratio — see :meth:`cond_expectation`).
        """
        if isinstance(sites, int):
            sites = (sites,)
        sites = tuple(sorted(int(s) for s in sites))
        if len(set(sites)) != len(sites):
            raise ValueError("joint_marginal needs distinct sites")
        cores = self._cores(params)
        openset = set(sites)
        weights = self._ones_weights()
        if masks is not None:
            for s, m in masks.items():
                if s not in openset:
                    weights[s] = jnp.asarray(m, jnp.float32)
        E = self._contract_open(cores, self.kind == "born", weights, openset)
        return E / jnp.sum(E) if normalize else E

    def site_marginal(self, params, site: int):
        """Exact per-bin mass vector p(X_site = k), shape (d_site,)."""
        return self.joint_marginal(params, (site,))

    def pair_marginal(self, params, i: int, j: int):
        """Exact joint marginal table ``p(X_i, X_j)`` of shape (d_i, d_j)."""
        if i == j:
            raise ValueError("pair_marginal needs two distinct sites")
        M = self.joint_marginal(params, (i, j))       # axes sorted(i, j)
        return M if i < j else M.T

    def cond_expectation(self, params, site, given, f=None):
        """Exact conditional expectation ``E[f(X_site) | given]`` — differentiable."""
        v = self.joint_marginal(params, (site,), masks=given, normalize=False)
        cen = self.disc.bin_centers(site)
        fx = jnp.asarray(cen if f is None else f(cen), jnp.float32)
        return (fx @ v) / (jnp.sum(v) + _EPS)

    def marginal_kl(self, params, sites, ref, direction: str = "forward"):
        """KL between the exact joint marginal over ``sites`` and ``ref``.

        See :meth:`TensorChain.marginal_kl`: ``"forward"`` = ``KL(ref || p)``
        (fit-to-target), ``"reverse"`` = ``KL(p || ref)``.
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

        Separable ``{site: g}`` -> ``E[sum_i g_i(X_i)]``; joint ``(sites, table)``
        -> ``E[g(X_sites)]`` (see :meth:`TensorChain.expectation`).
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

    # ==================================================================
    # cached belief propagation — ALL marginals in one up + one down sweep
    # ==================================================================
    #
    # The single-query :meth:`site_marginal` / :meth:`pair_marginal` each run a
    # *full* tree contraction, so scoring ``C`` constraints costs ``O(C·n)``. On a
    # tree the marginals share structure: one leaves->root pass (upward messages
    # ``up[i]`` = the contraction of ``i``'s whole subtree, one open bond to its
    # parent) plus one root->leaves pass (downward messages ``down[i]`` = the
    # contraction of everything *outside* ``i``'s subtree) caches every directed-edge
    # message in ``O(n)``. Then each site marginal is ``core_i`` contracted with all
    # its incident messages, and each tree-edge pair marginal is the two endpoint
    # cores across their shared bond — all read from the cache. Constraint scoring
    # drops to ``O(n + C)``: a ``prob`` is a sum over a cached site marginal, a mean
    # a dot, a nearest-neighbour coupling a read of a cached edge marginal.
    #
    # Exact only for **nonneg** (sum-product on a tree). Per-message rescaling is
    # exact under normalisation: each message scale is a global constant on the
    # unnormalised marginal, so it cancels (a stop-gradient factor, differentiable).
    # Born would need doubled (r×r) messages; not implemented (use the per-query
    # methods, or the born contraction engine, meanwhile).

    def _rooted_structure(self):
        """(parent, children, preorder) rooted at node 0 — cached (topology is fixed).

        Iterative (no Python recursion), so it is safe for very large ``n`` where a
        path-shaped tree would blow the recursion limit."""
        cached = getattr(self, "_rooted_cache", None)
        if cached is not None:
            return cached
        n, adj = self.n, self.adj
        parent = [-1] * n; order = []; seen = [False] * n
        stack = [0]; seen[0] = True
        while stack:
            u = stack.pop(); order.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; parent[v] = u; stack.append(v)
        children = [[v for v in adj[u] if parent[v] == u] for u in range(n)]
        self._rooted_cache = (parent, children, order)
        return self._rooted_cache

    def _bp_contract(self, core, adj_i, incoming, open_nb=None, open_phys=False):
        """Contract node ``core`` with the ``incoming`` messages ``{nb: (r,)}``,
        leaving ``open_nb``'s bond and/or the physical axis open; rescale by
        ``max|·|`` (stop-grad, exact under normalisation)."""
        used = {_ALPHA[0]}; phys = _ALPHA[0]
        bond = {}
        for nb in adj_i:
            bond[nb] = _alloc(used)
        inp = [phys + "".join(bond[nb] for nb in adj_i)]; ops = [core]
        out = phys if open_phys else ""
        for nb in adj_i:
            if nb == open_nb:
                out += bond[nb]
            else:
                inp.append(bond[nb]); ops.append(incoming[nb])
        arr = jnp.einsum(",".join(inp) + "->" + out, *ops)
        s = jax.lax.stop_gradient(jnp.max(jnp.abs(arr))) + _EPS
        return arr / s

    def _bp_messages(self, cores):
        """All directed-edge messages: ``up[i]`` (i -> parent[i]) and ``down[i]``
        (parent[i] -> i), each a length-``r`` vector. One upward + one downward pass."""
        parent, children, order = self._rooted_structure()
        up = {}
        for i in reversed(order):                       # leaves -> root
            if parent[i] == -1:
                continue
            inc = {c: up[c] for c in children[i]}
            up[i] = self._bp_contract(cores[i], self.adj[i], inc, open_nb=parent[i])
        down = {}
        for i in order:                                 # root -> leaves
            for c in children[i]:
                inc = {nb: (down[i] if nb == parent[i] else up[nb])
                       for nb in self.adj[i] if nb != c}
                down[c] = self._bp_contract(cores[i], self.adj[i], inc, open_nb=c)
        return up, down, parent, children

    def _bp_require_nonneg(self, what):
        if self.kind != "nonneg":
            raise NotImplementedError(
                f"{what} is nonneg-only for now (exact sum-product BP on a tree); "
                "for born use site_marginal / pair_marginal per query.")

    def _node_open(self, cores, up, down, parent, children, i, exclude=None):
        """``core_i`` contracted with every incident message except ``exclude``'s,
        physical axis open (and ``exclude``'s bond open if given)."""
        inc = {nb: (down[i] if nb == parent[i] else up[nb])
               for nb in self.adj[i] if nb != exclude}
        return self._bp_contract(cores[i], self.adj[i], inc,
                                 open_nb=exclude, open_phys=True)

    def all_site_marginals(self, params):
        """Every exact per-site marginal ``[p(X_0), …, p(X_{n-1})]`` in one BP sweep.

        A list of length ``n`` (ragged: entry ``i`` has shape ``(d_i,)``), each
        normalised. Differentiable in ``params``. Equal to
        ``[site_marginal(params, i) for i in range(n)]`` but ``O(n)`` total instead
        of ``O(n)`` *per site* — the scaling primitive for many-constraint fits
        (feed the cache to a batched loss + :func:`~calibrated_response.tn.backends.reusable_adam`).
        Nonneg only.
        """
        self._bp_require_nonneg("all_site_marginals")
        cores = self._cores(params)
        up, down, parent, children = self._bp_messages(cores)
        out = []
        for i in range(self.n):
            v = self._node_open(cores, up, down, parent, children, i, exclude=None)
            out.append(v / jnp.sum(v))
        return out

    def all_edge_marginals(self, params):
        """Every exact nearest-neighbour pair marginal, one per tree edge, in one BP sweep.

        Returns ``{(a, b): p(X_a, X_b)}`` for each edge ``(a, b)`` in :attr:`edges`
        (axes ordered ``(X_a, X_b)`` as stored), each normalised. These are the
        couplings a tree represents directly, so place cross-variable constraints on
        adjacent sites to read them straight from this cache (``O(n)`` for all of
        them). Non-adjacent pairs are not cached — query :meth:`pair_marginal`, or
        re-root the tree so the coupled pair is an edge. Nonneg only.
        """
        self._bp_require_nonneg("all_edge_marginals")
        cores = self._cores(params)
        up, down, parent, children = self._bp_messages(cores)
        out = {}
        for a, b in self.edges:
            Ta = self._node_open(cores, up, down, parent, children, a, exclude=b)  # (d_a, r)
            Tb = self._node_open(cores, up, down, parent, children, b, exclude=a)  # (d_b, r)
            M = jnp.einsum("xr,yr->xy", Ta, Tb)
            out[(a, b)] = M / jnp.sum(M)
        return out

    # ---- arbitrary small-set marginals via Steiner-subtree contraction ----
    #
    # A constraint on an arbitrary set S (a pair, a triple) needs p(X_S). On a tree
    # that is the contraction of the *minimal subtree spanning S* — a path for |S|=2,
    # a "Y" for |S|=3 — with every node on that subtree closed off by its cached
    # off-subtree BP messages. Cost is O(|subtree|·r^3): the r^3 of the node
    # contractions times how many nodes lie between the query variables. So on a
    # bounded-degree latent tree (coupled variables a few hops apart) an arbitrary
    # pair/triple is O(depth·r^3) — not the O(n) of a full :meth:`joint_marginal`
    # sweep — and the one BP sweep is shared across the whole constraint list.
    #
    # (A single ``n``-way latent star is NOT usable: its core has ``n`` bond legs,
    #  i.e. r^n entries. Factor structure must be a *bounded-degree* latent tree,
    #  giving O(log n) paths.)

    def _depths(self):
        cached = getattr(self, "_depth_cache", None)
        if cached is not None:
            return cached
        parent, _, order = self._rooted_structure()
        depth = [0] * self.n
        for v in order:                                 # preorder: parent before child
            depth[v] = 0 if parent[v] == -1 else depth[parent[v]] + 1
        self._depth_cache = depth
        return depth

    def _path_nodes(self, u, v):
        """All nodes on the unique tree path ``u..v`` inclusive (via parent pointers)."""
        parent = self._rooted_structure()[0]
        depth = self._depths()
        pu, pv = [], []
        while depth[u] > depth[v]:
            pu.append(u); u = parent[u]
        while depth[v] > depth[u]:
            pv.append(v); v = parent[v]
        while u != v:
            pu.append(u); pv.append(v); u = parent[u]; v = parent[v]
        return pu + [u] + pv[::-1]

    def _steiner_nodes(self, sites):
        """Nodes of the minimal subtree spanning ``sites`` (union of pairwise paths)."""
        sl = list(sites)
        T = set(sl)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                T.update(self._path_nodes(sl[i], sl[j]))
        return T

    def _subtree_contract(self, cores, up, down, parent, children, sites):
        """Contract the Steiner subtree spanning ``sites``, closing its boundary with
        the cached BP messages. Returns the (normalised) joint over ``sites``, axes in
        sorted-``sites`` order. Reuses one message cache across many calls."""
        sites = tuple(sorted(set(int(s) for s in sites)))
        site_set = set(sites)
        T = self._steiner_nodes(sites)
        Tadj = {v: [w for w in self.adj[v] if w in T] for v in T}

        def msg(w, v):                                  # cached message w -> v
            return down[v] if w == parent[v] else up[w]

        def rec(i, tparent):
            used = set()
            p = _alloc(used)
            bond = {nb: _alloc(used) for nb in self.adj[i]}
            inputs = [p + "".join(bond[nb] for nb in self.adj[i])]
            operands = [cores[i]]
            open_letters, open_ids = [], []
            if i in site_set:                           # keep this physical axis open
                open_letters.append(p); open_ids.append(i)
            for nb in self.adj[i]:
                if nb == tparent:
                    continue                            # leave bond[nb] open (up to T-parent)
                if nb in Tadj[i]:                       # in-subtree child: recurse
                    arr_c, ids_c = rec(nb, i)
                    col = [_alloc(used) for _ in ids_c]
                    inputs.append(bond[nb] + "".join(col)); operands.append(arr_c)
                    open_letters.extend(col); open_ids.extend(ids_c)
                else:                                   # boundary: fold cached message
                    inputs.append(bond[nb]); operands.append(msg(nb, i))
            out = ("" if tparent is None else bond[tparent]) + "".join(open_letters)
            arr = jnp.einsum(",".join(inputs) + "->" + out, *operands)
            s = jax.lax.stop_gradient(jnp.max(jnp.abs(arr))) + _EPS
            return arr / s, open_ids

        arr, open_ids = rec(sites[0], None)
        arr = jnp.transpose(arr, [open_ids.index(s) for s in sites])
        return arr / jnp.sum(arr)

    def subtree_marginal(self, params, sites):
        """Exact joint marginal ``p(X_sites)`` over an ARBITRARY small set of sites.

        Same result as :meth:`joint_marginal` (axes follow ``sorted(sites)``), but
        computed by contracting only the Steiner subtree spanning ``sites`` over the
        cached BP messages — ``O(|subtree|·r^3)`` after one ``O(n·r^3)`` BP pass,
        versus ``joint_marginal``'s full ``O(n)`` sweep *per call*. Use
        :meth:`pair_marginals` / :meth:`subtree_marginals` to amortise the BP pass
        across a whole constraint list. Nonneg only.
        """
        self._bp_require_nonneg("subtree_marginal")
        cores = self._cores(params)
        up, down, parent, children = self._bp_messages(cores)
        return self._subtree_contract(cores, up, down, parent, children, sites)

    def subtree_marginals(self, params, site_sets):
        """Marginals for many arbitrary sets, sharing ONE BP pass.

        ``site_sets`` is an iterable of site tuples; returns a list of tables aligned
        with it (each over ``sorted(set)``). Cost ``O(n·r^3 + Σ |subtree|·r^3)``."""
        self._bp_require_nonneg("subtree_marginals")
        cores = self._cores(params)
        up, down, parent, children = self._bp_messages(cores)
        return [self._subtree_contract(cores, up, down, parent, children, s)
                for s in site_sets]

    def pair_marginals(self, params, pairs):
        """``{(i, j): p(X_i, X_j)}`` for ARBITRARY pairs (not just tree edges), one
        BP pass shared across all of them (axes ``(X_min, X_max)``). Nonneg only."""
        self._bp_require_nonneg("pair_marginals")
        cores = self._cores(params)
        up, down, parent, children = self._bp_messages(cores)
        out = {}
        for (i, j) in pairs:
            out[(i, j)] = self._subtree_contract(cores, up, down, parent, children, (i, j))
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

    # ---- cached scoring: one BP sweep, then read every constraint off it ----
    def _apply_factors(self, joint, sites, factors, keep):
        """From a ``joint`` whose axes are ``sorted(sites)``, multiply in the per-site
        vectors ``factors`` (masks / tables / value functions) and sum every axis
        *not* in ``keep``. Scalar if ``keep`` is empty; else axes ``sorted(keep)``."""
        sites = list(sites)
        used = set()
        letter = {s: _alloc(used) for s in sites}
        inp = ["".join(letter[s] for s in sites)]
        ops = [joint]
        for s, v in factors.items():
            inp.append(letter[s]); ops.append(jnp.asarray(v, jnp.float32))
        out = "".join(letter[s] for s in sorted(keep))
        return jnp.einsum(",".join(inp) + "->" + out, *ops)

    def _constraint_site_sets(self, constraints):
        """Every site-set whose joint marginal the scorer will request — mirrors the
        branches of :meth:`_score_with` exactly, so the batched provider caches all of
        them. Used to build a :class:`~calibrated_response.tn.steiner.SteinerMarginals`."""
        from .steiner import _canon
        sets = []
        for cst in constraints:
            kind = cst[0]
            if kind == "prob":
                sets.append(_canon(cst[1]))
            elif kind == "cond":
                sets.append(_canon(set(cst[1]) | set(cst[2])))
            elif kind == "kl":
                sets.append(_canon(cst[1]))
            elif kind == "expect":
                values = cst[1]
                if isinstance(values, tuple):
                    sets.append(_canon(values[0]))
                else:
                    sets.extend(_canon(site) for site in values)
            elif kind == "cond_expect":
                sets.append(_canon({cst[1]} | set(cst[2])))
            else:
                raise ValueError(f"unknown constraint kind {kind!r}")
        return sets

    def _cached_constraint_scores(self, cores, constraints):
        """Score every constraint off ONE shared BP sweep (nonneg only).

        Same numbers as scoring each constraint with its own full contraction, but
        the compile graph is ``O(n)`` (one up+down pass) plus one small
        Steiner-subtree contraction per constraint — not ``O(C·n)`` independent full
        contractions. Per-constraint Steiner reference; the batched provider in
        :meth:`constraint_loss` gives identical numbers with an ``O(1)``-in-``C`` graph.
        """
        up, down, parent, children = self._bp_messages(cores)

        def joint_over(sites):                       # normalised p(X_sites), axes sorted
            return self._subtree_contract(cores, up, down, parent, children, sites)

        return self._score_with(constraints, joint_over)

    def _score_with(self, constraints, joint_over):
        """The constraint SSE, parameterised by a ``joint_over(sites) -> p(X_sites)``
        provider (axes in sorted-site order). Shared by the per-constraint Steiner path
        and the batched :class:`SteinerMarginals` path — the only difference between them
        is how ``joint_over`` is computed."""
        tot = 0.0
        for cst in constraints:
            kind = cst[0]
            if kind == "prob":
                _, ev, tg = cst
                sites = tuple(sorted(ev))
                val = self._apply_factors(joint_over(sites), sites, dict(ev), set())
                tot = tot + (val - tg) ** 2
            elif kind == "cond":
                _, ev, gv, tg = cst
                sites = tuple(sorted(set(ev) | set(gv)))
                P = joint_over(sites)
                merged = dict(gv)
                for s, m in ev.items():              # shared site -> product mask
                    merged[s] = merged[s] * m if s in merged else m
                num = self._apply_factors(P, sites, merged, set())
                den = self._apply_factors(P, sites, dict(gv), set())
                tot = tot + (num / (den + _EPS) - tg) ** 2
            elif kind == "kl":
                sites, ref = cst[1], cst[2]
                w = cst[3] if len(cst) > 3 else 1.0
                st = (sites,) if isinstance(sites, int) else tuple(sorted(sites))
                P = joint_over(st)
                r = jnp.asarray(ref, jnp.float32); r = r / jnp.sum(r)
                tot = tot + w * jnp.sum(r * (jnp.log(r + _EPS) - jnp.log(P + _EPS)))
            elif kind == "expect":
                values, tg = cst[1], cst[2]
                w = cst[3] if len(cst) > 3 else 1.0
                if isinstance(values, tuple):
                    sites, table = values
                    st = (sites,) if isinstance(sites, int) else tuple(sorted(sites))
                    ev = jnp.sum(joint_over(st) * jnp.asarray(table, jnp.float32))
                else:
                    ev = 0.0
                    for site, g in values.items():
                        ev = ev + jnp.sum(joint_over((site,)) * jnp.asarray(g, jnp.float32))
                tot = tot + w * (ev - tg) ** 2
            elif kind == "cond_expect":
                site, given, tg = cst[1], cst[2], cst[3]
                w = cst[4] if len(cst) > 4 else 1.0
                sites = tuple(sorted({site} | set(given)))
                P = joint_over(sites)
                cen = jnp.asarray(self.disc.bin_centers(site), jnp.float32)
                num = self._apply_factors(P, sites, {**given, site: cen}, set())
                den = self._apply_factors(P, sites, dict(given), set())
                tot = tot + w * (num / (den + _EPS) - tg) ** 2
            else:
                raise ValueError(f"unknown constraint kind {kind!r}")
        return tot

    def _direct_constraint_scores(self, p, constraints):
        """Per-constraint full contraction — the born path (and the reference the
        cached scorer is verified against)."""
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
        return tot

    def constraint_loss(self, constraints, weight_reg=0.0):
        """Squared-error loss on marginal / conditional constraints (spec §6).

        Grammar (identical to :meth:`TensorChain.constraint_loss`):
            ("prob", event_dict, target)
            ("cond", event_dict, given_dict, target)
            ("kl",   sites, ref[, weight])
            ("expect", values, target[, weight])
            ("cond_expect", site, given, target[, weight])

        For ``nonneg`` the whole constraint list is scored off a single cached BP sweep
        through a batched :class:`~calibrated_response.tn.steiner.SteinerMarginals`
        provider, whose compile graph is ``O(n)`` (the sweep) plus ``O(1)`` in the number
        of constraints — it stays tractable into the thousands of constraints where the
        per-constraint graph walls. ``born`` falls back to a full contraction per
        constraint. All three (born / per-constraint Steiner / batched) agree numerically.
        """
        if self.kind == "nonneg":
            from .steiner import SteinerMarginals, _canon
            provider = SteinerMarginals(self, self._constraint_site_sets(constraints))

            def loss(p):
                cores = self._cores(p)
                joints = provider(cores)                 # owns the batched BP sweep
                tot = self._score_with(constraints, lambda s: joints[_canon(s)])
                if weight_reg:
                    tot = tot + weight_reg * sum(jnp.mean(c ** 2) for c in p["cores"])
                return tot
            return loss

        def loss(p):
            tot = self._direct_constraint_scores(p, constraints)
            if weight_reg:
                tot = tot + weight_reg * sum(jnp.mean(c ** 2) for c in p["cores"])
            return tot
        return loss

    def fit(self, X_continuous, backend: str = "adam", seed: int = 0, **kw):
        """Convenience: fit by NLL to a continuous dataset (discretised first)."""
        Xi = self.disc.to_index(np.asarray(X_continuous))
        return self.optimize(self.nll_loss(Xi), backend=backend, seed=seed, **kw)

    def nll(self, params, Xi):
        return float(-jnp.mean(self.log_prob_idx(params, Xi)))
