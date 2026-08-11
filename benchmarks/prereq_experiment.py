"""Payload for the prerequisite-event maxent-recovery sweep (git-pullable).

The Colab notebook (``benchmarks/colab_prereq_sweep.ipynb``) is a thin shell:
it clones/pulls the repo and calls :func:`run_sweep` / :func:`plot_results`
from here, so iterating on the experiment is a ``git pull`` — no notebook
re-upload.

The experiment: an event ``e`` with ``m`` independent prerequisites
``a_1..a_m`` (all sites on [0, 1], proposition = ``x > 0.5``).  Elicited
constraints are only

* ``P(a_i) = p`` for each prerequisite (default 0.7), and
* ``P(e and not a_i) ~ 0`` for each — the implications ``e -> a_i``.

Nothing is elicited about ``P(e | all prerequisites met)``, so the true
max-entropy joint has ``P(e | C) = 0.5`` exactly.  NOTE the prerequisites
are **not** independent at the optimum: the Gibbs family is star-shaped
(features ``1[a_i]`` and ``1[e and not a_i]`` all couple through ``e``), so
the e-branch adds mass at the all-ones corner, marginally correlating the
``a_i`` and pushing ``P(C)`` above ``p^m``.  :func:`maxent_reference`
computes the exact binary-reduction solution (symmetric site weight ``w``
by bisection); all "ratio"/deviation metrics are measured against it.
Previously fitted models collapsed ``P(e | C)`` far below 0.5 — the
implication constraints squash ``e`` globally and the flow fails to carve
the corner back out.  The sweep asks whether a Gaussian-mixture base
(``FlowSamplerModel(n_components=K)``) recovers it, across the number of
prerequisites ``m`` and mixture size ``K`` (including ``n_layers=0``:
no coupling layers at all, a pure squashed-GMM engine).

Headline metrics per fit:

* ``p_e_given_c``   — want 0.5 (THE number that was previously wrong)
* ``p_c_ratio``     — P(C) / maxent P(C), want 1.0 (no condition-mass drift)
* ``p_e_given_notc``— want ~0 (implications hold)
* ``marg_err``      — max_i |P(a_i) - p|, want 0
* ``max_corr``      — max off-diagonal |corr| among prerequisite
  indicators; the maxent target is ``corr_true`` (positive!), not 0
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

from calibrated_response.maxent_sampler.flow_model import FlowSamplerModel
from calibrated_response.maxent_sampler.model import soft_gt, soft_lt
from calibrated_response.tn.discretize import ContinuousVar

P_PREREQ = 0.7      # elicited P(a_i)
K_OBS = 32.0        # pseudo-count strength of the P(a_i) estimates
K_HARD = 256.0      # pseudo-count strength of the implication constraints
P_IMPOSSIBLE = 1e-3 # "e without a_i" target (prob_nll clips near 0 anyway)
VIOL_SHARP = 800.0  # sharpness of the violation feature's indicators

# WHY the extra sharpness: scoring the implication with default-sharpness
# (50) indicators makes the EXACT maxent solution register soft violation
# ~0.012-0.015 (sigmoid-width mass near the thresholds) -- 12-15x the
# P_IMPOSSIBLE target, so at K_HARD the objective itself preferred the
# vacuous e-suppressed joint over the true one (verified by direct scoring,
# all m).  The leakage floor scales ~1/sharpness; at 800 it is ~0.0007,
# below the target, and direct scoring prefers the true joint at every m.
# (Threshold *margins* were tried first and are WRONG: they open a dead band
# -- e.g. e in (0.5, 0.6) -- that hard-threshold evaluation counts as a
# violation but the loss cannot see, and entropy happily fills it.)


def maxent_reference(n_prereq: int, p_prereq: float = P_PREREQ):
    """Exact maxent solution of the binary reduction (hard implications).

    Unnormalized weights: ``e=0`` states get ``prod_i w^{a_i}``; the only
    ``e=1`` state is all-ones with weight ``prod_i w``.  The symmetric site
    weight ``w`` is set by ``P(a_i) = p`` (bisection; the marginal is
    monotone in ``w``).  Exact up to soft-indicator smoothing and the
    finite-k implication target."""
    m = n_prereq

    def marg(w):
        Z = (1.0 + w) ** m + w ** m
        return (w * (1.0 + w) ** (m - 1) + w ** m) / Z

    lo, hi = 1e-9, 1e9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if marg(mid) < p_prereq:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)
    Z = (1.0 + w) ** m + w ** m
    p_c = 2.0 * w ** m / Z
    p_e = w ** m / Z
    if m > 1:
        p11 = (w * w * (1.0 + w) ** (m - 2) + w ** m) / Z
        corr = (p11 - p_prereq ** 2) / (p_prereq * (1.0 - p_prereq))
    else:
        corr = 0.0
    return dict(w=w, p_c=p_c, p_e=p_e, corr=corr)


def st_ind(site: int, threshold: float, sharpness: float = 50.0,
           direction: str = "gt"):
    """Straight-through indicator: forward = exact hard 1[..], backward =
    the soft sigmoid's gradient.  In a product of two ST features each
    factor's gradient is gated by the OTHER factor's *hard* value, so e.g.
    samples inside the all-met corner feel zero implication gradient."""
    def f(x):
        v = x[:, site]
        d = (v - threshold) if direction == "gt" else (threshold - v)
        s = jax.nn.sigmoid(sharpness * d)
        h = (d > 0).astype(jnp.float32)
        return s + jax.lax.stop_gradient(h - s)
    return f


def build_constraints(n_prereq: int, p_prereq: float = P_PREREQ,
                      k_obs: float = K_OBS, k_hard: float = K_HARD,
                      viol_mode: str = "st"):
    """Sites 0..m-1 are prerequisites, site m is the event.

    ``viol_mode="st"`` (default) scores the implications with
    straight-through indicators: the forward violation probability is EXACT
    (no soft-leakage floor at any sharpness -- see the VIOL_SHARP note), and
    gradients keep default-sharpness support.  ``"soft"`` keeps the
    sharpness-``VIOL_SHARP`` sigmoid scoring for A/B."""
    if viol_mode == "st":
        g_e = st_ind(n_prereq, 0.5, direction="gt")
        not_as = [st_ind(i, 0.5, direction="lt") for i in range(n_prereq)]
    else:
        g_e = soft_gt(n_prereq, 0.5, VIOL_SHARP)
        not_as = [soft_lt(i, 0.5, VIOL_SHARP) for i in range(n_prereq)]
    csts = []
    for i in range(n_prereq):
        csts.append(("prob_nll", soft_gt(i, 0.5), p_prereq, k_obs))

        def violated(x, not_a=not_as[i]):  # e holds but prerequisite i doesn't
            return g_e(x) * not_a(x)
        csts.append(("prob_nll", violated, P_IMPOSSIBLE, k_hard))
    return csts


def fit_and_measure(n_prereq=4, n_components=1, n_layers=6, hidden=64,
                    base_spread=1.0, p_prereq=P_PREREQ, steps=3000,
                    n_samples=4096, eval_samples=200_000, seed=0,
                    warmup_steps=0, warmup_k_hard=8.0, viol_mode="st"):
    """Fit one model and measure recovery of the analytic maxent solution.

    ``warmup_steps > 0`` anneals the implications: that many steps at
    ``k_hard=warmup_k_hard`` first, then ``steps`` at full K_HARD from the
    warm-started params.  Rationale: at init e is ~uniform, so soft violation
    starts ~0.1 and the full-strength implication term (loss ~100) crushes e
    globally in the first ~100 steps; the vacuous basin is then an attractor
    (verified: longer single-phase training DECREASES P(e|C)).  A weak-k
    phase lets the corner structure form before the wall goes up."""
    sites = ([ContinuousVar(f"a{i}", 0.0, 1.0, 16) for i in range(n_prereq)]
             + [ContinuousVar("e", 0.0, 1.0, 16)])
    m = FlowSamplerModel(sites, n_layers=n_layers, hidden=hidden,
                         n_components=n_components, base_spread=base_spread)
    npar = int(m.net.n_params)
    loss = m.constraint_loss(
        build_constraints(n_prereq, p_prereq, viol_mode=viol_mode),
        entropy_reg=1.0, n_samples=n_samples)
    t0 = time.time()
    if warmup_steps:
        weak = m.constraint_loss(
            build_constraints(n_prereq, p_prereq, k_hard=warmup_k_hard,
                              viol_mode=viol_mode),
            entropy_reg=1.0, n_samples=n_samples)
        p0, _ = m.optimize(weak, seed=seed, steps=warmup_steps, lr=2e-3,
                           grad_clip=5.0)
        params, hist = m.optimize(loss, seed=seed + 1, steps=steps, lr=1e-3,
                                  grad_clip=5.0, init=p0)
    else:
        params, hist = m.optimize(loss, seed=seed, steps=steps, lr=2e-3,
                                  grad_clip=5.0)
    fit_s = time.time() - t0

    x = m.sample(params, eval_samples, seed=seed + 1)
    A = x[:, :n_prereq] > 0.5                    # (N, m) prerequisite truth
    E = x[:, n_prereq] > 0.5
    C = A.all(axis=1)
    marg = A.mean(axis=0)
    corr = np.corrcoef(A.T) if n_prereq > 1 else np.eye(1)
    off = corr[~np.eye(n_prereq, dtype=bool)] if n_prereq > 1 else np.array([0.0])
    ref = maxent_reference(n_prereq, p_prereq)
    return dict(
        n_prereq=int(n_prereq), n_components=int(n_components),
        n_layers=int(n_layers), hidden=int(hidden), seed=int(seed),
        warmup_steps=int(warmup_steps), viol_mode=viol_mode,
        params=npar, fit_seconds=round(fit_s, 1),
        entropy=float(m.entropy(params)),
        final_loss=float(np.mean(hist[-50:])),
        p_c=float(C.mean()), p_c_true=float(ref["p_c"]),
        p_e_true=float(ref["p_e"]), corr_true=float(ref["corr"]),
        p_c_ratio=float(C.mean() / ref["p_c"]),
        p_e=float(E.mean()),
        p_e_given_c=float(E[C].mean()) if C.any() else float("nan"),
        p_e_given_notc=float(E[~C].mean()) if (~C).any() else float("nan"),
        marg_err=float(np.max(np.abs(marg - p_prereq))),
        max_corr=float(np.max(np.abs(off))),
    )


def default_configs():
    """m x K grid (affine 6x64, ST violations), annealed variants, the
    soft-violation A/B at K=1, and a flowless GMM arm."""
    cfgs = []
    for m in (2, 4, 6):
        for K in (1, 4, 16, 64):
            cfgs.append(dict(n_prereq=m, n_components=K, n_layers=6, hidden=64))
        for K in (1, 16):                        # annealed k_hard (two-phase)
            cfgs.append(dict(n_prereq=m, n_components=K, n_layers=6, hidden=64,
                             warmup_steps=1000))
        for wu in (0, 1000):                     # sharp-soft scoring A/B
            cfgs.append(dict(n_prereq=m, n_components=1, n_layers=6, hidden=64,
                             warmup_steps=wu, viol_mode="soft"))
        for K in (16, 64):                       # squashed-GMM: no coupling layers
            cfgs.append(dict(n_prereq=m, n_components=K, n_layers=0, hidden=64))
    return cfgs


def _key(row):
    return (row["n_prereq"], row["n_layers"], row["hidden"],
            row["n_components"], row.get("warmup_steps", 0),
            row.get("viol_mode", "soft"), row["seed"])


def run_sweep(configs=None, seeds=(0, 1, 2), steps=3000, n_samples=4096,
              eval_samples=200_000, out_path="results/prereq_sweep.jsonl"):
    """Fit every (config x seed); append rows to ``out_path`` (resumable)."""
    configs = configs or default_configs()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(_key(json.loads(line)))
            except Exception:
                pass
    rows, t0 = [], time.time()
    for cfg in configs:
        for s in seeds:
            probe = {**cfg, "seed": s}
            if _key({**dict(hidden=64), **probe}) in done:
                continue
            row = fit_and_measure(steps=steps, n_samples=n_samples,
                                  eval_samples=eval_samples, seed=s, **cfg)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            wu = f" wu{row['warmup_steps']}" if row["warmup_steps"] else ""
            print(f"m={row['n_prereq']} L{row['n_layers']:<2} "
                  f"K{row['n_components']:<3}{wu} seed{s} | "
                  f"P(e|C)={row['p_e_given_c']:.3f} (want 0.500) "
                  f"P(C)ratio={row['p_c_ratio']:.3f} "
                  f"P(e|~C)={row['p_e_given_notc']:.4f} "
                  f"margerr={row['marg_err']:.3f} "
                  f"corr={row['max_corr']:.3f} (want {row['corr_true']:.3f}) "
                  f"({time.time() - t0:.0f}s)")
    return [json.loads(l) for l in open(out_path)]


def plot_results(rows_or_path="results/prereq_sweep.jsonl",
                 save="results/prereq_sweep.png"):
    """P(e|C) recovery and P(C) drift vs. mixture size, one line per m."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if isinstance(rows_or_path, str):
        rows = [json.loads(l) for l in open(rows_or_path)]
    else:
        rows = list(rows_or_path)
    # recompute the ratio against the analytic reference so rows logged by
    # older payload versions (which used p^m as the target) plot correctly
    refs = {m: maxent_reference(m) for m in {r["n_prereq"] for r in rows}}
    for r in rows:
        r["p_c_ratio"] = r["p_c"] / refs[r["n_prereq"]]["p_c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {2: "tab:blue", 4: "tab:orange", 6: "tab:red"}
    families = [  # (predicate, style, suffix)
        (lambda r: r["n_layers"] > 0 and not r.get("warmup_steps", 0),
         dict(marker="o", ls="-"), ""),
        (lambda r: r["n_layers"] > 0 and r.get("warmup_steps", 0),
         dict(marker="^", ls="-.", alpha=0.8), " (annealed)"),
        (lambda r: r["n_layers"] == 0,
         dict(marker="s", ls="--", alpha=0.6), " (GMM, L=0)"),
    ]
    for pred, style, suffix in families:
        for m in sorted({r["n_prereq"] for r in rows}):
            by = defaultdict(list)
            for r in rows:
                if r["n_prereq"] == m and pred(r):
                    by[r["n_components"]].append(r)
            if not by:
                continue
            Ks = sorted(by)
            mean = lambda key: np.array([np.mean([r[key] for r in by[K]]) for K in Ks])
            sd = lambda key: np.array([np.std([r[key] for r in by[K]]) for K in Ks])
            label = f"m={m}{suffix}"
            c = colors.get(m)
            ax1.errorbar(Ks, mean("p_e_given_c"), yerr=sd("p_e_given_c"),
                         lw=2, capsize=3, color=c, label=label, **style)
            ax2.errorbar(Ks, mean("p_c_ratio"), yerr=sd("p_c_ratio"),
                         lw=2, capsize=3, color=c, label=label, **style)
    for ax, target, name in ((ax1, 0.5, "P(e | all prerequisites met)"),
                             (ax2, 1.0, "P(C) / p^m")):
        ax.axhline(target, color="gray", ls=":", label=f"maxent target ({target})")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_components (mixture base size)")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
    ax1.set_ylim(0, 0.6)
    ax1.set_title("Corner recovery (was collapsing toward 0)")
    ax2.set_title("Condition-mass drift (maxent wants 1.0)")
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, dpi=130)
    return fig
