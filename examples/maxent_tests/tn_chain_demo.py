"""Fit a tensor-chain to the A->B->C chained-Bayes problem by CONSTRAINT MATCHING.

No dataset. We minimise a squared-error loss on the marginal/conditional
*constraints* of ``case_chain_propagation`` (spec §6), exactly the local
probabilities that define the problem:

    P(A>50)=0.70
    P(B>50|A>50)=0.80   P(B<50|A<50)=0.90
    P(C>50|B>50)=0.80   P(C<50|B<50)=0.90

The *propagated* marginals P(B>50)=0.59, P(C>50)=0.513 are NOT fit directly;
they should emerge (law of total probability) if the model satisfies the local
constraints. Verification reads them exactly off the network + checks positive
correlation via Born-rule samples.

Run:  JAX_PLATFORMS=cpu python -m examples.maxent_tests.tn_chain_demo
"""

from __future__ import annotations

import numpy as np

from calibrated_response.tn import TensorChain, ContinuousVar

A, B, C = 0, 1, 2
EXPECTED = {"A": 0.70, "B": 0.59, "C": 0.513}   # analytic propagated marginals


def chain_constraints(model: TensorChain, n_var=3):
    """The 5 local constraints of case_chain_propagation as (kind, ...) tuples."""
    up = lambda s: {s: model.threshold_mask(s, 50.0, above=True)}
    dn = lambda s: {s: model.threshold_mask(s, 50.0, above=False)}
    var = list(range(n_var))
    constraints = [("prob", up(var[0]), 0.70)]
    for i in range(1, n_var):
        constraints.append(("cond", up(var[i]), up(var[i - 1]), 0.70))
        constraints.append(("cond", dn(var[i]), dn(var[i - 1]), 0.95))
    return constraints


def report(model, params, tag):
    n_var = len(model.vars)
    names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:n_var]
    print(f"\n=== {tag} ===")
    print("  constraints (model vs target):")
    for cst in chain_constraints(model, n_var=n_var):
        if cst[0] == "prob":
            _, ev, tg = cst
            v = float(model.event_prob(params, ev))
            print(f"    P(A>50)               = {v:.3f}   (target {tg:.2f})")
        else:
            _, ev, gv, tg = cst
            v = float(model.cond_prob(params, ev, gv))
            es = "B" if B in ev else "C"
            side = ">" if list(ev.values())[0][-1] > 0.5 else "<"  # rough label
            print(f"    P(cond) {es}|prev          = {v:.3f}   (target {tg:.2f})")
    print("  propagated marginals (NOT fit directly), exact from network:")
    for i, nm in enumerate(names):
        print(f"    P({nm}>50) = {model.prob_gt(params, i, 50.0):.3f}   (target {EXPECTED[nm]:.3f})")
    s = model.sample(params, 20000, seed=1)
    cab = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
    cbc = float(np.corrcoef(s[:, 1], s[:, 2])[0, 1])
    cac = float(np.corrcoef(s[:, 0], s[:, 2])[0, 1])
    print(f"  sampled correlations: AB={cab:+.3f} BC={cbc:+.3f} AC={cac:+.3f} "
          f"{'ok' if min(cab, cbc, cac) > 0 else 'FAIL'}")


# ---- optional: sample the true generative process, for reference only ----
def generate_chain(N=20000, seed=0):
    rng = np.random.default_rng(seed)
    def draw(pr, n):
        hi = rng.random(n) < pr
        return np.where(hi, rng.uniform(50, 100, n), rng.uniform(0, 50, n))
    a = draw(0.70, N)
    b = np.where(a > 50, draw(0.80, N), draw(0.10, N))
    c = np.where(b > 50, draw(0.80, N), draw(0.10, N))
    return np.stack([a, b, c], axis=1)


def main():
    vars = [ContinuousVar("A", 0, 100, 20),
            ContinuousVar("B", 0, 100, 20),
            ContinuousVar("C", 0, 100, 20)]

    model = TensorChain(vars, bond_dim=6, kind="born")
    loss = model.constraint_loss(chain_constraints(model), weight_reg=1e-3)
    params, hist = model.optimize(loss, backend="adam", steps=2000, lr=3e-2, seed=0)
    print(f"final constraint loss = {hist[-1]:.2e}")
    report(model, params, "Born machine trained on constraints")


if __name__ == "__main__":
    main()
