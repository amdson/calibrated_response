"""Batched Steiner-subtree marginals for many site-sets sharing ONE belief-propagation
sweep — the scaling primitive behind :meth:`TensorTree.constraint_loss` on large trees.

A constraint on a small site-set ``S`` needs ``p(X_S)``, the contraction of the minimal
subtree spanning ``S`` closed off by the cached BP messages. Doing that per constraint
emits a distinct einsum chain each time, so the *compile graph* grows ``O(C · depth)`` and
walls (OOM) in the thousands of constraints. This module collapses the whole list into a
graph that is **O(depth) in the tree size and O(1) in the constraint count**:

  * :class:`BPPlan` runs the up/down message sweep **level-synchronously** — all nodes at
    one tree depth sharing a (degree, parent-slot) signature are contracted in ONE batched
    einsum, so the whole sweep is ``O(depth)`` graph ops instead of ``O(n)``.
  * every Steiner arm is walked as a fixed-length chain of ``(r, r)`` transfer matrices
    (identity-padded to a common length), so all constraints of a given arity share ONE
    transfer table, ONE ``lax.scan`` of gathered matmuls, and <= 6 batched combine einsums.

Only sizes 1/2/3 on a max-degree-3 tree with uniform leaf bin-dim are batched. Everything
else (non-leaf site, ``|S| > 3``, higher-degree tree, ragged dims, non-binary sweep) falls
back transparently to :meth:`TensorTree._subtree_contract` / :meth:`_bp_messages` — always
correct, just not accelerated.

Correctness is defined by agreement with those fallbacks / :meth:`TensorTree.subtree_marginals`;
identity padding and per-message rescaling are exact under the final normalisation.
"""
from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

_EPS = 1e-30
_BL = ["A", "B", "C"]                                 # med bond letters (combine)
_OL = ["x", "y", "z"]                                 # output (physical) letters (combine)
_bl = "abc"                                           # sweep bond letters


def _canon(sites):
    """Canonical key for a site-set: sorted unique ints."""
    if isinstance(sites, int):
        return (int(sites),)
    return tuple(sorted(set(int(x) for x in sites)))


def _rescale(x):
    """Per-row max-normalise (stop-grad); exact under later sum-normalisation, keeps the
    depth-long message products in float32 range — mirrors ``TensorTree._bp_contract``."""
    s = jax.lax.stop_gradient(jnp.max(jnp.abs(x), axis=-1, keepdims=True)) + _EPS
    return x / s


class BPPlan:
    """Static level-synchronous schedule for the up/down BP sweep on a fixed tree.

    ``run(cores) -> (UpA, DownA)`` returns the directed-edge messages as ``(n, r)`` arrays
    (``UpA[i]`` = message ``i -> parent[i]``; ``DownA[i]`` = ``parent[i] -> i``), consistent
    with the node-0 rooting the rest of the class uses, in ``O(depth)`` batched einsums.
    ``ok`` is False for trees this schedule doesn't cover (degree > 3, ragged leaf dims);
    the caller then uses the per-node :meth:`TensorTree._bp_messages`.
    """

    def __init__(self, m):
        self.m = m
        self.r = m.r
        parent, children, _ = m._rooted_structure()
        self.parent, self.children = parent, children
        n = m.n
        self.root = next(v for v in range(n) if parent[v] == -1)
        leaf_dims = {m.dims[v] for v in range(n) if len(children[v]) == 0}
        self.ok = (n > 0 and max(m.deg) <= 3 and len(leaf_dims) <= 1)
        if not self.ok:
            return
        depth = m._depths()

        def up_group(nodes):                          # (deg, parent-slot) -> one einsum
            groups = defaultdict(list)
            for v in nodes:
                groups[(m.deg[v], m.adj[v].index(parent[v]))].append(v)
            out = []
            for (dv, pv), vs in groups.items():
                folds = [s for s in range(dv) if s != pv]
                es = ("v z" + _bl[:dv] + "".join(",v" + _bl[s] for s in folds)
                      + "->v" + _bl[pv]).replace(" ", "")
                ids = np.array(vs, np.int64)
                specs = [("U", np.array([m.adj[v][s] for v in vs], np.int64)) for s in folds]
                out.append(dict(es=es, core=ids, tgt=ids, specs=specs))
            return out

        # UP: every non-root node, deepest depth first (children are always one level deeper)
        by_d = defaultdict(list)
        for v in range(n):
            if v != self.root:
                by_d[depth[v]].append(v)
        self.up_levels = [up_group(by_d[d]) for d in sorted(by_d, reverse=True)]

        # DOWN: root's children first (no parent term), then receivers by increasing depth
        self.down_root = self._down_group(
            [c for c in children[self.root]], is_root=True)
        by_d2 = defaultdict(list)
        for c in range(n):
            v = parent[c]
            if v != -1 and v != self.root:
                by_d2[depth[c]].append(c)
        self.down_levels = [self._down_group(by_d2[d], is_root=False)
                            for d in sorted(by_d2)]

    def _down_group(self, recv, is_root):
        """Group down-message receivers ``c`` by parent shape into batched einsums.
        ``down[c]`` folds ``down[parent[c]]`` (unless the parent is the root) and the
        up-messages of ``c``'s siblings, leaving ``c``'s bond open."""
        m, parent = self.m, self.parent
        groups = defaultdict(list)
        for c in recv:
            v = parent[c]
            pv = -1 if is_root else m.adj[v].index(parent[v])
            groups[(m.deg[v], pv, m.adj[v].index(c))].append(c)
        out = []
        for (dv, pv, t), cs in groups.items():
            others = [s for s in range(dv) if s != t and s != pv]
            parts = ["v z" + _bl[:dv]]
            specs = []
            if not is_root:
                parts.append("v" + _bl[pv]); specs.append(("D", np.array([parent[c] for c in cs], np.int64)))
            for s in others:
                parts.append("v" + _bl[s])
                specs.append(("U", np.array([m.adj[parent[c]][s] for c in cs], np.int64)))
            es = (",".join(parts) + "->v" + _bl[t]).replace(" ", "")
            core = np.array([parent[c] for c in cs], np.int64)
            out.append(dict(es=es, core=core, tgt=np.array(cs, np.int64), specs=specs))
        return out

    def _apply(self, cores, g, UpA, DownA):
        LC = jnp.stack([cores[int(i)] for i in g["core"]])
        ops = [LC]
        for src, ids in g["specs"]:
            ops.append((UpA if src == "U" else DownA)[jnp.asarray(ids)])
        return g["tgt"], _rescale(jnp.einsum(g["es"], *ops))

    def run(self, cores):
        n, r = self.m.n, self.r
        UpA = jnp.zeros((n, r), jnp.float32)
        DownA = jnp.zeros((n, r), jnp.float32)
        for lvl in self.up_levels:
            for g in lvl:
                tgt, val = self._apply(cores, g, UpA, DownA)
                UpA = UpA.at[jnp.asarray(tgt)].set(val)
        for g in self.down_root:
            tgt, val = self._apply(cores, g, UpA, DownA)
            DownA = DownA.at[jnp.asarray(tgt)].set(val)
        for lvl in self.down_levels:
            for g in lvl:
                tgt, val = self._apply(cores, g, UpA, DownA)
                DownA = DownA.at[jnp.asarray(tgt)].set(val)
        return UpA, DownA


_PAD, _D3, _D2 = 0, 1, 2                              # per-step transfer kinds


class SteinerMarginals:
    """Precompute static index arrays for a fixed collection of site-sets, then map
    ``cores -> {site_tuple: p(X_site_tuple)}`` in an ``O(depth)`` graph (batched sizes
    1/2/3, fallback for the rest). Owns the BP sweep (:class:`BPPlan`)."""

    def __init__(self, m, site_sets):
        self.m = m
        self.bp = BPPlan(m)
        uniq = sorted({_canon(s) for s in site_sets}, key=lambda s: (len(s), s))

        deg = m.deg
        max_deg_ok = max(deg) <= 3 if m.n else True
        leaf_dims = {m.dims[v] for v in range(m.n) if deg[v] == 1}
        self.NB = next(iter(leaf_dims)) if len(leaf_dims) == 1 else None

        def batchable_leaf(s):
            return (max_deg_ok and self.NB is not None
                    and all(deg[v] == 1 and m.dims[v] == self.NB for v in s))

        self.singles, self.pairs, self.triples, self.fallback = [], [], [], []
        for s in uniq:
            if batchable_leaf(s) and len(s) == 1:
                self.singles.append(s)
            elif batchable_leaf(s) and len(s) == 2:
                self.pairs.append(s)
            elif batchable_leaf(s) and len(s) == 3:
                self.triples.append(s)
            else:
                self.fallback.append(s)

        self.nodes3 = [v for v in range(m.n) if deg[v] == 3]
        self.nodes2 = [v for v in range(m.n) if deg[v] == 2]
        self.idx3 = {v: k for k, v in enumerate(self.nodes3)}
        self.idx2 = {v: k for k, v in enumerate(self.nodes2)}
        if self.nodes3:
            self._build_msg_slots()
        if self.singles:
            self._build_singles()
        if self.pairs:
            self._build_pairs()
        if self.triples:
            self._build_triples()

    # ---- static index construction (numpy, once) -----------------------------
    def _parent(self):
        return self.m._rooted_structure()[0]

    def _build_msg_slots(self):
        m, parent = self.m, self._parent()
        K = len(self.nodes3)
        self.msg_src = np.zeros((K, 3), np.int64)
        self.msg_down = np.zeros((K, 3), bool)
        for k, v in enumerate(self.nodes3):
            for s, w in enumerate(m.adj[v]):
                if w == parent[v]:
                    self.msg_src[k, s], self.msg_down[k, s] = v, True     # down[v]
                else:
                    self.msg_src[k, s], self.msg_down[k, s] = w, False    # up[w]

    def _step(self, v, pin, pout):
        adjv = self.m.adj[v]
        ipos, opos = adjv.index(pin), adjv.index(pout)
        if self.m.deg[v] == 3:
            return (_D3, self.idx3[v], 3 - ipos - opos, ipos > opos)
        return (_D2, self.idx2[v], 0, ipos > opos)

    def _arm_steps(self, path):
        return [self._step(path[t], path[t - 1], path[t + 1])
                for t in range(1, len(path) - 1)]

    def _median(self, i, j, k):
        pij, pik = self.m._path_nodes(i, j), self.m._path_nodes(i, k)
        t = 0
        while t < len(pij) and t < len(pik) and pij[t] == pik[t]:
            t += 1
        return pij[t - 1]

    def _build_singles(self):
        m, parent = self.m, self._parent()
        self.single_leaf = np.array([s[0] for s in self.singles], np.int64)
        src, down = [], []
        for s in self.singles:
            leaf = s[0]; nb = m.adj[leaf][0]
            if nb == parent[leaf]:
                src.append(leaf); down.append(True)
            else:
                src.append(nb); down.append(False)
        self.single_src = np.array(src, np.int64)
        self.single_down = np.array(down, bool)

    def _pack_grid(self, step_lists):
        L = max((len(s) for s in step_lists), default=0)
        R = len(step_lists)
        kind = np.zeros((R, L), np.int64); idx = np.zeros((R, L), np.int64)
        off = np.zeros((R, L), np.int64); tr = np.zeros((R, L), bool)
        for r, steps in enumerate(step_lists):
            for t, (kd, ix, of, trv) in enumerate(steps):
                kind[r, t], idx[r, t], off[r, t], tr[r, t] = kd, ix, of, trv
        return kind, idx, off, tr, L

    def _build_pairs(self):
        step_lists, far = [], []
        for (i, j) in self.pairs:
            path = self.m._path_nodes(i, j)
            step_lists.append(self._arm_steps(path)); far.append(j)
        self.pair_far = np.array(far, np.int64)
        (self.pk_kind, self.pk_idx, self.pk_off,
         self.pk_tr, self.pair_L) = self._pack_grid(step_lists)

    def _build_triples(self):
        arm_steps, meds, medpos = [], [], []
        for t in self.triples:
            med = self._median(*t); meds.append(med)
            mp = []
            for x in t:
                path = self.m._path_nodes(x, med)
                arm_steps.append(self._arm_steps(path))
                mp.append(self.m.adj[med].index(path[-2]))
            medpos.append(tuple(mp))
        self.tri_med_k = np.array([self.idx3[md] for md in meds], np.int64)
        (self.tk_kind, self.tk_idx, self.tk_off,
         self.tk_tr, self.tri_L) = self._pack_grid(arm_steps)
        groups = defaultdict(list)
        for c, mp in enumerate(medpos):
            groups[mp].append(c)
        self.tri_groups = dict(groups)

    # ---- runtime (jnp) -------------------------------------------------------
    def __call__(self, cores):
        m, r = self.m, self.m.r
        parent, children, _ = m._rooted_structure()

        if self.bp.ok:
            UpA, DownA = self.bp.run(cores)
            up_d = down_d = None
        else:                                          # non-binary / unsupported: dict sweep
            up, down, parent, children = m._bp_messages(cores)
            zr = jnp.zeros(r, jnp.float32)
            UpA = jnp.stack([up.get(i, zr) for i in range(m.n)])
            DownA = jnp.stack([down.get(i, zr) for i in range(m.n)])
            up_d, down_d = up, down

        out = {}
        if self.fallback:                              # per-set Steiner contraction (dicts)
            if up_d is None:
                up_d = {i: UpA[i] for i in range(m.n)}
                down_d = {i: DownA[i] for i in range(m.n)}
            for s in self.fallback:
                out[s] = m._subtree_contract(cores, up_d, down_d, parent, children, s)
        if not (self.singles or self.pairs or self.triples):
            return out

        def msg_of(src, is_down):
            return jnp.where(jnp.asarray(is_down)[..., None],
                             DownA[jnp.asarray(src)], UpA[jnp.asarray(src)])

        if self.singles:
            LC = jnp.stack([cores[int(v)] for v in self.single_leaf])     # (S,NB,r)
            mv = msg_of(self.single_src, self.single_down)
            marg = jnp.einsum("sxr,sr->sx", LC, mv)
            marg = marg / jnp.sum(marg, axis=1, keepdims=True)
            for i, s in enumerate(self.singles):
                out[s] = marg[i]

        if self.pairs or self.triples:
            Tf3, Tf2 = self._transfer_tables(cores, msg_of, r)
        if self.pairs:
            self._eval_pairs(cores, Tf3, Tf2, out)
        if self.triples:
            self._eval_triples(cores, Tf3, Tf2, out)
        return out

    def _transfer_tables(self, cores, msg_of, r):
        Tf3 = Tf2 = None
        if self.nodes3:
            LC3 = jnp.stack([cores[v].reshape(r, r, r) for v in self.nodes3])
            msg = msg_of(self.msg_src, self.msg_down)                     # (K3,3,r)
            Tf3 = jnp.stack([
                jnp.einsum("kabc,ka->kbc", LC3, msg[:, 0]),               # fold slot 0
                jnp.einsum("kabc,kb->kac", LC3, msg[:, 1]),               # fold slot 1
                jnp.einsum("kabc,kc->kab", LC3, msg[:, 2]),               # fold slot 2
            ], axis=1)
        if self.nodes2:
            Tf2 = jnp.stack([cores[v].reshape(r, r) for v in self.nodes2])
        return Tf3, Tf2

    def _gather_M(self, Tf3, Tf2, kind, idx, off, tr, r):
        eye = jnp.eye(r, dtype=jnp.float32)[None]
        M = jnp.broadcast_to(eye, (kind.shape[0], r, r))
        if Tf3 is not None:
            M3 = Tf3[jnp.asarray(idx), jnp.asarray(off)]
            M3 = jnp.where(jnp.asarray(tr)[:, None, None], jnp.swapaxes(M3, 1, 2), M3)
            M = jnp.where((jnp.asarray(kind) == _D3)[:, None, None], M3, M)
        if Tf2 is not None:
            M2 = Tf2[jnp.asarray(idx)]
            M2 = jnp.where(jnp.asarray(tr)[:, None, None], jnp.swapaxes(M2, 1, 2), M2)
            M = jnp.where((jnp.asarray(kind) == _D2)[:, None, None], M2, M)
        return M

    def _scan_arms(self, A, kind, idx, off, tr, Tf3, Tf2, rows, L, r):
        M = self._gather_M(Tf3, Tf2, kind.reshape(-1), idx.reshape(-1),
                           off.reshape(-1), tr.reshape(-1), r).reshape(rows, L, r, r)
        M = jnp.moveaxis(M, 1, 0)
        return jax.lax.scan(lambda A, Ms: (jnp.einsum("rni,rij->rnj", A, Ms), None), A, M)[0]

    def _eval_pairs(self, cores, Tf3, Tf2, out):
        r = self.m.r; P = len(self.pairs)
        A = jnp.stack([cores[i] for (i, _) in self.pairs])
        A = self._scan_arms(A, self.pk_kind, self.pk_idx, self.pk_off, self.pk_tr,
                            Tf3, Tf2, P, self.pair_L, r)
        Far = jnp.stack([cores[int(j)] for j in self.pair_far])
        J = jnp.einsum("pxr,pyr->pxy", A, Far)
        J = J / jnp.sum(J, axis=(1, 2), keepdims=True)
        for p, s in enumerate(self.pairs):
            out[s] = J[p]

    def _eval_triples(self, cores, Tf3, Tf2, out):
        r = self.m.r; T = len(self.triples)
        leaves = [x for t in self.triples for x in t]
        A = jnp.stack([cores[x] for x in leaves])
        A = self._scan_arms(A, self.tk_kind, self.tk_idx, self.tk_off, self.tk_tr,
                            Tf3, Tf2, 3 * T, self.tri_L, r).reshape(T, 3, -1, r)
        LC3 = jnp.stack([cores[v].reshape(r, r, r) for v in self.nodes3])
        med_k = jnp.asarray(self.tri_med_k)
        for mp, idxs in self.tri_groups.items():
            ii = jnp.asarray(idxs)
            terms = ["cABC"]; ops = [LC3[med_k[ii]]]
            for rk in range(3):
                terms.append("c" + _OL[rk] + _BL[mp[rk]]); ops.append(A[ii][:, rk])
            J = jnp.einsum(",".join(terms) + "->cxyz", *ops)
            J = J / jnp.sum(J, axis=(1, 2, 3), keepdims=True)
            for g, c in enumerate(idxs):
                out[self.triples[c]] = J[g]
