"""Mixture-base vs standard-Gaussian base on a bimodal maxent target.

The conflict2 scenario from nll_conflict_demo (two clashing expect_nll
estimates, E[x]=10 vs E[x]=45) has a bimodal maxent solution.  A flow with a
unimodal N(0, I) base must tear the base apart with the invertible map to
represent it — extreme Jacobians, ugly optimization.  With
``n_components=K`` the base is a uniform-weight Gaussian mixture with
learnable means/scales, so bimodality comes from component placement and the
map stays gentle.  Entropy and log_prob stay exact.

Note the conflict2 optimum is *edge*-bimodal (sd 41 > uniform ceiling 28.9
means mass at 0 and 100) — and edge modes are cheap for a plain flow (the
sigmoid squash saturates).  The discriminating stress is **interior** modes:
``interior2`` constrains E[sin(3 pi x / 100)] ~ 0.9, whose maxent solution
p(x) ~ exp(lam sin(...)) has equal interior peaks at x = 16.7 and 83.3 with a
deep valley at 50.  A monotone-map flow must contort to split an interior
valley; a mixture base just parks one component per peak.

Grid: {agree, conflict2, interior2} x {K=1, K=4}.  agree is the unimodal
control — the mixture should not hurt it.

Run:  python examples/mixture_base_demo.py
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from calibrated_response.maxent_sampler.flow_model import FlowSamplerModel
from calibrated_response.maxent_sampler.model import moment, soft_gt
from calibrated_response.tn.discretize import ContinuousVar

K_OBS = 8.0    # effective observation count per estimate
TAU = 3.0      # noise-sd floor, in x units
BETA = 0.5     # beta-NLL exponent

FIT_KW = dict(steps=1500, n_samples=2048, lr=2e-3, grad_clip=5.0)

mx = moment(0, 1)
gx275 = soft_gt(0, 27.5)
gy = soft_gt(1, 50.0)


def fsin(x):
    """Peaks at x = 16.7 and 83.3, valley at 50 (and at the edges)."""
    return jnp.sin(3.0 * jnp.pi * x[:, 0] / 100.0)


SCENARIOS = {
    "agree": [
        ("expect_nll", mx, 27.5, K_OBS, TAU, BETA),
        ("prob_nll", gx275, 0.5, K_OBS),
        ("cond_expect_nll", mx, gy, 27.5, K_OBS, TAU, BETA),
    ],
    "conflict2": [
        ("expect_nll", mx, 10.0, K_OBS, TAU, BETA),
        ("expect_nll", mx, 45.0, K_OBS, TAU, BETA),
    ],
    "interior2": [
        ("expect", fsin, 0.92, 200.0),
    ],
}


def fit(constraints, n_components, seed=0):
    m = FlowSamplerModel([ContinuousVar("x", 0.0, 100.0, 32),
                          ContinuousVar("y", 0.0, 100.0, 32)],
                         n_layers=6, hidden=32, n_components=n_components)
    loss = m.constraint_loss(constraints, entropy_reg=1.0,
                             n_samples=FIT_KW["n_samples"])
    params, hist = m.optimize(
        loss, seed=seed,
        **{k: v for k, v in FIT_KW.items() if k != "n_samples"})
    return m, params, hist


PROBES = np.array([[16.7, 50.0], [50.0, 50.0], [83.3, 50.0]])


def report(m, params, n=40000, seed=1):
    x = m.sample(params, n, seed=seed)
    lp = m.log_prob(params, PROBES)
    return dict(
        mean=float(np.mean(x[:, 0])),
        sd=float(np.std(x[:, 0])),
        entropy=m.entropy(params),
        p_lo=float(np.mean(x[:, 0] < 50.0)),          # mode balance
        p_mid=float(np.mean((x[:, 0] > 30.0) & (x[:, 0] < 70.0))),
        lp_pk1=float(lp[0]), lp_val=float(lp[1]), lp_pk2=float(lp[2]),
    )


def main():
    rows = {}
    for scen, csts in SCENARIOS.items():
        for nc in (1, 4):
            name = f"{scen}-K{nc}"
            print(f"fitting {name} ...", flush=True)
            m, params, hist = fit(csts, nc)
            r = report(m, params)
            r["loss"] = float(np.mean(hist[-50:]))
            rows[name] = r
            print(f"  loss(avg last 50) {r['loss']:.3f}  ->  {r}", flush=True)

    hdr = (f"{'run':<14}{'mean(x)':>9}{'sd(x)':>8}{'loss':>9}{'H':>8}"
           f"{'P(x<50)':>9}{'P(mid)':>8}"
           f"{'lq(16.7)':>10}{'lq(50)':>8}{'lq(83.3)':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:<14}{r['mean']:>9.2f}{r['sd']:>8.2f}{r['loss']:>9.3f}"
              f"{r['entropy']:>8.2f}{r['p_lo']:>9.3f}{r['p_mid']:>8.3f}"
              f"{r['lp_pk1']:>10.3f}{r['lp_val']:>8.3f}{r['lp_pk2']:>10.3f}")

    c1, c4 = rows["conflict2-K1"], rows["conflict2-K4"]
    a1, a4 = rows["agree-K1"], rows["agree-K4"]
    i1, i4 = rows["interior2-K1"], rows["interior2-K4"]
    checks = [
        ("mixture keeps conflict width (sd > 30)", c4["sd"] > 30.0),
        ("mixture conflict mean near pooled 27.5",
         abs(c4["mean"] - 27.5) < 8.0),
        ("mixture no worse on conflict loss",
         c4["loss"] < c1["loss"] + 2.0),
        ("mixture does not hurt unimodal control (sd within 40%)",
         a4["sd"] < 1.4 * a1["sd"]),
        ("unimodal control loss comparable", a4["loss"] < a1["loss"] + 1.0),
        ("interior2-K4 has both interior modes populated",
         0.2 < i4["p_lo"] < 0.8),
        ("interior2-K4 density dips at the valley (x = 50)",
         i4["lp_val"] < min(i4["lp_pk1"], i4["lp_pk2"]) - 0.5),
        ("interior2 mixture loss no worse than standard base",
         i4["loss"] < i1["loss"] + 0.5),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
