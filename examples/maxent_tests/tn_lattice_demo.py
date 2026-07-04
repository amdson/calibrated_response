"""3x3 lattice Bayes net — a harder test with collider nodes.

Nodes are a 3x3 grid of continuous variables on [0,100], thresholded at 50 (a
binary H/L abstraction). Edges go right and down, so every interior node has two
parents => it is a **collider**. A collider's CPD P(child | p1, p2) is a genuine
3-way interaction: it cannot be pinned by pairwise constraints, only by
**triplet** constraints P(child>50 | p1 vs 50, p2 vs 50) over all four parent
states. The tensor chain expresses these directly as ``cond_prob`` with a
two-site ``given`` (verified exact).

    (0,0) -> (0,1) -> (0,2)
      |        |        |
      v        v        v
    (1,0) -> (1,1) -> (1,2)
      |        |        |
      v        v        v
    (2,0) -> (2,1) -> (2,2)

Colliders (two parents): (1,1),(1,2),(2,1),(2,2). We give them AND / OR gates
(non-additive, so genuinely requiring the triplet). Everything is threshold-based
so the exact marginals are computable by enumerating the 2^9 binary net.
"""

from __future__ import annotations

import itertools

import numpy as np

from calibrated_response.tn import TensorChain, ContinuousVar

# ---- structure -----------------------------------------------------------
NODES = [(r, c) for r in range(3) for c in range(3)]     # row-major
IDX = {node: r * 3 + c for r, c in NODES for node in [(r, c)]}
IDX = {(r, c): r * 3 + c for r in range(3) for c in range(3)}


def parents(node):
    r, c = node
    ps = []
    if c > 0:
        ps.append((r, c - 1))     # left
    if r > 0:
        ps.append((r - 1, c))     # up
    return ps


COLLIDER_GATE = {(1, 1): "AND", (1, 2): "OR", (2, 1): "OR", (2, 2): "AND"}
ROOT_HIGH = 0.6
FOLLOW_HI, FOLLOW_LO = 0.85, 0.15     # single-parent: follow the parent
AND_HI, AND_LO = 0.90, 0.15           # AND gate: high mainly when both high
OR_HI, OR_LO = 0.85, 0.10             # OR gate: low mainly when both low


def cpd_high(node, parent_states):
    """P(node = High | parent_states), parent_states a tuple of bools in parents() order."""
    ps = parents(node)
    if not ps:
        return ROOT_HIGH
    if len(ps) == 1:
        return FOLLOW_HI if parent_states[0] else FOLLOW_LO
    p1, p2 = parent_states
    if COLLIDER_GATE[node] == "AND":
        return AND_HI if (p1 and p2) else AND_LO
    return OR_LO if (not p1 and not p2) else OR_HI      # OR gate


# ---- analytic joint via 2^9 enumeration ----------------------------------
def analytic_joint():
    """Full binary joint P(bits) as a (2,)*9 array; index 1 == 'High' (>50)."""
    J = np.zeros((2,) * 9)
    for bits in itertools.product([0, 1], repeat=9):
        val = {node: bool(bits[IDX[node]]) for node in NODES}
        p = 1.0
        for node in NODES:
            ph = cpd_high(node, tuple(val[pp] for pp in parents(node)))
            p *= ph if val[node] else (1.0 - ph)
        J[bits] = p
    return J / J.sum()


def analytic_marginals(J=None):
    J = analytic_joint() if J is None else J
    return np.array([J.sum(axis=tuple(k for k in range(9) if k != s))[1] for s in range(9)])


def analytic_family_table(J, sites):
    """Joint threshold table over ``sites`` (list of site indices), shape (2,)*len."""
    keep = tuple(sites)
    drop = tuple(k for k in range(9) if k not in keep)
    T = J.sum(axis=drop)
    # reorder axes to match `sites` order
    order = np.argsort(np.argsort(keep))
    return np.transpose(T, axes=order)


# ---- constraint builders -------------------------------------------------
def _masks(model):
    up = lambda s: {s: model.threshold_mask(s, 50.0, above=True)}
    dn = lambda s: {s: model.threshold_mask(s, 50.0, above=False)}
    return up, dn


def lattice_constraints_cpd(model):
    """Conditional-CPD constraints: root prior + P(child | parents) for every node
    (colliders => triplet *conditionals*). Pins each CPD but NOT the parents' joint,
    so collider marginals stay under-determined."""
    up, dn = _masks(model)
    cons = []
    for node in NODES:
        s = IDX[node]
        ps = parents(node)
        if not ps:
            cons.append(("prob", up(s), cpd_high(node, ())))
        elif len(ps) == 1:
            p = IDX[ps[0]]
            cons.append(("cond", up(s), up(p), cpd_high(node, (True,))))
            cons.append(("cond", up(s), dn(p), cpd_high(node, (False,))))
        else:
            p1, p2 = IDX[ps[0]], IDX[ps[1]]
            for b1 in (True, False):
                for b2 in (True, False):
                    g = {**(up(p1) if b1 else dn(p1)), **(up(p2) if b2 else dn(p2))}
                    cons.append(("cond", up(s), g, cpd_high(node, (b1, b2))))
    return cons


def lattice_constraints_family(model, J=None):
    """Family-*marginal* constraints: for each node, the joint threshold marginal
    over {node} ∪ parents (root=1-site, single-parent=pair, collider=**triplet**).
    These pin both the CPD and the parents' joint, so they reproduce the net."""
    up, dn = _masks(model)
    J = analytic_joint() if J is None else J
    cons = []
    for node in NODES:
        fam = [IDX[node]] + [IDX[p] for p in parents(node)]
        T = analytic_family_table(J, fam)                 # (2,)*len(fam)
        for bits in itertools.product([0, 1], repeat=len(fam)):
            ev = {}
            for site, b in zip(fam, bits):
                ev.update(up(site) if b else dn(site))
            cons.append(("prob", ev, float(T[bits])))
    return cons


def make_model(bins=8, bond_dim=16, kind="born"):
    vars = [ContinuousVar(f"{r}{c}", 0, 100, bins) for r, c in NODES]
    return TensorChain(vars, bond_dim=bond_dim, kind=kind)


# ---- verification --------------------------------------------------------
def constraint_residuals(model, params, cons):
    errs = []
    for cst in cons:
        if cst[0] == "prob":
            _, ev, tg = cst
            errs.append(abs(float(model.event_prob(params, ev)) - tg))
        else:
            _, ev, gv, tg = cst
            errs.append(abs(float(model.cond_prob(params, ev, gv)) - tg))
    return float(np.max(errs)), float(np.mean(errs))


def report(model, params, cons, tag):
    marg = analytic_marginals()
    print(f"\n=== {tag} ===")
    mx, mn = constraint_residuals(model, params, cons)
    print(f"  constraint residual: max {mx:.4f}  mean {mn:.4f}  ({len(cons)} constraints)")
    print("  propagated P(node>50): node  model  analytic  err")
    errs = []
    for node in NODES:
        s = IDX[node]
        pm = model.prob_gt(params, s, 50.0)
        pa = marg[s]
        errs.append(abs(pm - pa))
        print(f"    {node}   {pm:.3f}   {pa:.3f}   {abs(pm-pa):.3f}")
    print(f"  max |model-analytic| over nodes = {max(errs):.3f}")
    return max(errs)


def fit(model, cons, steps=2500, lr=3e-2, curv=2.0):
    from calibrated_response.tn import losses as L
    loss = L.combined_loss(model, cons, [("curvature", curv)])
    params, hist = model.optimize(loss, backend="adam", steps=steps, lr=lr,
                                  init=model.init_params(init="uniform"))
    return params, hist


def main():
    J = analytic_joint()
    # attempt 1: conditional CPDs (colliders as triplet conditionals)
    m1 = make_model(bins=8, bond_dim=16, kind="born")
    c1 = lattice_constraints_cpd(m1)
    p1, h1 = fit(m1, c1)
    print(f"[cpd] final loss = {h1[-1]:.2e}")
    e1 = report(m1, p1, c1, "attempt 1: collider CPD *conditionals*")

    # attempt 2: family marginals (colliders as triplet *marginals*)
    m2 = make_model(bins=8, bond_dim=16, kind="born")
    c2 = lattice_constraints_family(m2, J)
    p2, h2 = fit(m2, c2)
    print(f"\n[family] final loss = {h2[-1]:.2e}")
    e2 = report(m2, p2, c2, "attempt 2: collider triplet *marginals*")

    print(f"\nmax node error:  CPD-conditionals {e1:.3f}   triplet-marginals {e2:.3f}")


if __name__ == "__main__":
    main()
