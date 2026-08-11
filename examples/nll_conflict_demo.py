"""Two-variable demo of the synthetic-likelihood ("*_nll") constraint kinds.

The question under test: do the NLL kinds turn *disagreement between
estimates* into predictive width, where the squared-error kinds silently
average?  x in [0, 100] is the variable of interest; y in [0, 100] is an
unconstrained conditioning site (its marginal stays ~uniform, so
P(y > 50) ~ 0.5 and E[x | y > 50] targets x through a half-batch).

Scenarios (K, TAU, BETA shared):

    single       E[x] = 27.5                                   (one expect_nll)
    agree        E[x] = 27.5,  P(x > 27.5) = 0.5,  E[x | y > 50] = 27.5
                 -- three estimate types, mutually consistent
    conflict2    E[x] = 10  vs  E[x] = 45                      (direct clash)
    conflict3    E[x] = 10,  E[x | y > 50] = 45,  P(x > 45) = 0.7,
                 P(y > 50) = 0.5 -- mixed-type clash; pinning P(y > 50) blocks
                 the escape of starving the condition to satisfy the
                 conditional cheaply via correlation
    agree_sq /   E[x] = 27.5 twice  vs  E[x] = 10 & E[x] = 45, both with the
    conflict_sq  OLD squared "expect" kind (weight = 1/(2*5**2)) -- baseline

Expected ordering of fitted sd(x):

    sd(agree) < sd(single)  <<  sd(conflict2), sd(conflict3)

while the squared-loss baseline is *insensitive*: sd(conflict_sq) ~
sd(agree_sq) (whatever width entropy buys, disagreement neither widens nor
agreement tightens it).  Uniform-on-[0,100] sd = 28.9 is the ceiling.

Run:  python examples/nll_conflict_demo.py
"""
from __future__ import annotations

import numpy as np

from calibrated_response.maxent_sampler.flow_model import FlowSamplerModel
from calibrated_response.maxent_sampler.model import moment, soft_gt, hard_gt
from calibrated_response.tn.discretize import ContinuousVar

K = 8.0        # effective observation count per estimate
TAU = 3.0      # noise-sd floor, in x units (3% of span)
BETA = 0.5     # β-NLL exponent

FIT_KW = dict(steps=1500, n_samples=2048, lr=2e-3, grad_clip=5.0, seed=0)

mx = moment(0, 1)                    # E[x]
gy = soft_gt(1, 50.0)                # soft 1[y > 50] (loss side)
gx275 = soft_gt(0, 27.5)
gx45 = soft_gt(0, 45.0)

SCENARIOS = {
    "single": [
        ("expect_nll", mx, 27.5, K, TAU, BETA),
    ],
    "agree": [
        ("expect_nll", mx, 27.5, K, TAU, BETA),
        ("prob_nll", gx275, 0.5, K),
        ("cond_expect_nll", mx, gy, 27.5, K, TAU, BETA),
    ],
    "conflict2": [
        ("expect_nll", mx, 10.0, K, TAU, BETA),
        ("expect_nll", mx, 45.0, K, TAU, BETA),
    ],
    "conflict3": [
        ("expect_nll", mx, 10.0, K, TAU, BETA),
        ("cond_expect_nll", mx, gy, 45.0, K, TAU, BETA),
        ("prob_nll", gx45, 0.7, K),
        ("prob_nll", gy, 0.5, K),
    ],
    "agree_sq": [
        ("expect", mx, 27.5, 1.0 / (2.0 * 5.0 ** 2)),
        ("expect", mx, 27.5, 1.0 / (2.0 * 5.0 ** 2)),
    ],
    "conflict_sq": [
        ("expect", mx, 10.0, 1.0 / (2.0 * 5.0 ** 2)),
        ("expect", mx, 45.0, 1.0 / (2.0 * 5.0 ** 2)),
    ],
}


def fit_scenario(constraints, seed=0):
    m = FlowSamplerModel([ContinuousVar("x", 0.0, 100.0, 32),
                          ContinuousVar("y", 0.0, 100.0, 32)],
                         n_layers=6, hidden=32)
    loss = m.constraint_loss(constraints, entropy_reg=1.0, **{
        k: v for k, v in FIT_KW.items() if k == "n_samples"})
    params, hist = m.optimize(
        loss, seed=seed,
        **{k: v for k, v in FIT_KW.items() if k not in ("n_samples", "seed")})
    return m, params, hist


def report(m, params, n=40000, seed=1):
    x = m.sample(params, n, seed=seed)
    inb = x[:, 1] > 50.0
    return dict(
        mean=float(np.mean(x[:, 0])),
        sd=float(np.std(x[:, 0])),
        p_gt_275=float(np.mean(x[:, 0] > 27.5)),
        p_gt_45=float(np.mean(x[:, 0] > 45.0)),
        cond_mean=float(np.mean(x[inb, 0])) if inb.any() else float("nan"),
        p_y_gt_50=float(np.mean(inb)),
    )


def main():
    rows = {}
    for name, csts in SCENARIOS.items():
        print(f"fitting {name} ...", flush=True)
        m, params, hist = fit_scenario(csts)
        rows[name] = report(m, params)
        print(f"  final loss {hist[-1]:.3f}  ->  {rows[name]}", flush=True)

    hdr = f"{'scenario':<12}{'mean(x)':>9}{'sd(x)':>8}{'P(x>27.5)':>11}" \
          f"{'P(x>45)':>9}{'E[x|y>50]':>11}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:<12}{r['mean']:>9.2f}{r['sd']:>8.2f}"
              f"{r['p_gt_275']:>11.3f}{r['p_gt_45']:>9.3f}"
              f"{r['cond_mean']:>11.2f}")
    print(f"\n(uniform-on-[0,100] sd = {100 / np.sqrt(12):.1f} is the ceiling)")

    sd = {k: r["sd"] for k, r in rows.items()}
    checks = [
        ("agree tighter than single", sd["agree"] < sd["single"]),
        ("direct conflict widens vs single (>1.5x)",
         sd["conflict2"] > 1.5 * sd["single"]),
        ("mixed-type conflict widens vs single (>1.5x)",
         sd["conflict3"] > 1.5 * sd["single"]),
        ("squared-loss width insensitive to conflict (ratio < 1.3)",
         sd["conflict_sq"] < 1.3 * sd["agree_sq"]),
        ("NLL width responds to conflict (>3x agree)",
         sd["conflict2"] > 3.0 * sd["agree"]),
        ("conflict2 mean near pooled average",
         abs(rows["conflict2"]["mean"] - 27.5) < 8.0),
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
