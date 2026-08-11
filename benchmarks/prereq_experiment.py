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
max-entropy joint has independent Bernoulli(p) prerequisites and ``e``
uniform inside the all-met corner: ``P(e | C) = 0.5`` and ``P(C) = p^m``.
Previously fitted models collapsed ``P(e | C)`` far below 0.5 — the
implication constraints squash ``e`` globally and the flow fails to carve
the corner back out.  The sweep asks whether a Gaussian-mixture base
(``FlowSamplerModel(n_components=K)``) recovers it, across the number of
prerequisites ``m`` and mixture size ``K`` (including ``n_layers=0``:
no coupling layers at all, a pure squashed-GMM engine).

Headline metrics per fit:

* ``p_e_given_c``   — want 0.5 (THE number that was previously wrong)
* ``p_c_ratio``     — P(C) / p^m, want 1.0 (no condition-mass drift)
* ``p_e_given_notc``— want ~0 (implications hold)
* ``marg_err``      — max_i |P(a_i) - p|, want 0
* ``max_corr``      — max off-diagonal |corr| among prerequisite
  indicators, want 0 (independence is the maxent structure)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from calibrated_response.maxent_sampler.flow_model import FlowSamplerModel
from calibrated_response.maxent_sampler.model import soft_gt
from calibrated_response.tn.discretize import ContinuousVar

P_PREREQ = 0.7      # elicited P(a_i)
K_OBS = 32.0        # pseudo-count strength of the P(a_i) estimates
K_HARD = 256.0      # pseudo-count strength of the implication constraints
P_IMPOSSIBLE = 1e-3 # "e without a_i" target (prob_nll clips near 0 anyway)


def build_constraints(n_prereq: int, p_prereq: float = P_PREREQ,
                      k_obs: float = K_OBS, k_hard: float = K_HARD):
    """Sites 0..m-1 are prerequisites, site m is the event."""
    g_e = soft_gt(n_prereq, 0.5)
    csts = []
    for i in range(n_prereq):
        g_a = soft_gt(i, 0.5)
        csts.append(("prob_nll", g_a, p_prereq, k_obs))

        def violated(x, g_a=g_a):        # e holds but prerequisite i doesn't
            return g_e(x) * (1.0 - g_a(x))
        csts.append(("prob_nll", violated, P_IMPOSSIBLE, k_hard))
    return csts


def fit_and_measure(n_prereq=4, n_components=1, n_layers=6, hidden=64,
                    base_spread=1.0, p_prereq=P_PREREQ, steps=3000,
                    n_samples=4096, eval_samples=200_000, seed=0):
    """Fit one model and measure recovery of the analytic maxent solution."""
    sites = ([ContinuousVar(f"a{i}", 0.0, 1.0, 16) for i in range(n_prereq)]
             + [ContinuousVar("e", 0.0, 1.0, 16)])
    m = FlowSamplerModel(sites, n_layers=n_layers, hidden=hidden,
                         n_components=n_components, base_spread=base_spread)
    npar = int(m.net.n_params)
    csts = build_constraints(n_prereq, p_prereq)
    loss = m.constraint_loss(csts, entropy_reg=1.0, n_samples=n_samples)
    t0 = time.time()
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
    p_c_true = p_prereq ** n_prereq
    return dict(
        n_prereq=int(n_prereq), n_components=int(n_components),
        n_layers=int(n_layers), hidden=int(hidden), seed=int(seed),
        params=npar, fit_seconds=round(fit_s, 1),
        entropy=float(m.entropy(params)),
        final_loss=float(np.mean(hist[-50:])),
        p_c=float(C.mean()), p_c_true=float(p_c_true),
        p_c_ratio=float(C.mean() / p_c_true),
        p_e=float(E.mean()),
        p_e_given_c=float(E[C].mean()) if C.any() else float("nan"),
        p_e_given_notc=float(E[~C].mean()) if (~C).any() else float("nan"),
        marg_err=float(np.max(np.abs(marg - p_prereq))),
        max_corr=float(np.max(np.abs(off))),
    )


def default_configs():
    """m x K grid (affine 6x64) plus a flowless pure-GMM arm."""
    cfgs = []
    for m in (2, 4, 6):
        for K in (1, 4, 16, 64):
            cfgs.append(dict(n_prereq=m, n_components=K, n_layers=6, hidden=64))
        for K in (16, 64):                       # squashed-GMM: no coupling layers
            cfgs.append(dict(n_prereq=m, n_components=K, n_layers=0, hidden=64))
    return cfgs


def _key(row):
    return (row["n_prereq"], row["n_layers"], row["hidden"],
            row["n_components"], row["seed"])


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
            print(f"m={row['n_prereq']} L{row['n_layers']:<2} "
                  f"K{row['n_components']:<3} seed{s} | "
                  f"P(e|C)={row['p_e_given_c']:.3f} (want 0.500) "
                  f"P(C)ratio={row['p_c_ratio']:.3f} "
                  f"P(e|~C)={row['p_e_given_notc']:.4f} "
                  f"margerr={row['marg_err']:.3f} corr={row['max_corr']:.3f} "
                  f"({time.time() - t0:.0f}s)")
    return [json.loads(l) for l in open(out_path)]


def plot_results(rows_or_path="results/prereq_sweep.jsonl",
                 save="results/prereq_sweep.png"):
    """P(e|C) recovery and P(C) drift vs. mixture size, one line per m."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    rows = rows_or_path
    if isinstance(rows_or_path, str):
        rows = [json.loads(l) for l in open(rows_or_path)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {2: "tab:blue", 4: "tab:orange", 6: "tab:red"}
    for flowless in (False, True):
        for m in sorted({r["n_prereq"] for r in rows}):
            by = defaultdict(list)
            for r in rows:
                if r["n_prereq"] == m and (r["n_layers"] == 0) == flowless:
                    by[r["n_components"]].append(r)
            if not by:
                continue
            Ks = sorted(by)
            mean = lambda key: np.array([np.mean([r[key] for r in by[K]]) for K in Ks])
            sd = lambda key: np.array([np.std([r[key] for r in by[K]]) for K in Ks])
            style = dict(marker="s", ls="--", alpha=0.6) if flowless else dict(marker="o", ls="-")
            label = f"m={m}" + (" (GMM, L=0)" if flowless else "")
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
