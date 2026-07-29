"""Payload for the flow-capacity / derived-tail sweep (git-pullable).

The Colab notebook (``metaculus/colab_capacity_sweep.ipynb``) is a thin shell:
it clones/pulls the repo and calls :func:`run_sweep` / :func:`plot_results`
from here, so iterating on the experiment is a ``git pull`` — no notebook
re-upload. Edit this file, push, `git pull` in Colab, rerun the cell.

The experiment: the 2026-warmest-year equation model (an integrator
``anomaly`` defined by a structural equation over leaves, with a threshold
target). Making ``anomaly`` a soft-constrained flow coordinate clips its tail;
the sweep asks how much flow **capacity** and, now, flow **family**
(affine RealNVP vs. rational-quadratic spline) buy that tail back — measured
against the max-entropy target ``P_maxent`` (reconvolving the fitted leaf
marginals independently) and the entropy ``I`` the clip sacrifices.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from calibrated_response.models.variable import BinaryVariable, ContinuousVariable
from calibrated_response.models.natural_response import parse_natural_syntax as _P
from calibrated_response.maxent_sampler.distribution_builder import DistributionBuilder

RECORD = 1.60
HAND_MC = 0.067    # hand Monte-Carlo of the same structural model (reference)


def build_forecast_model():
    """(variables, estimates) for the 2026-warmest-year equation model."""
    variables = [
        BinaryVariable(name="record_2026", description="2026 warmest year on record"),
        ContinuousVariable(name="anomaly_2026", description="2026 global anomaly",
                           lower_bound=1.25, upper_bound=1.85, unit="C"),
        ContinuousVariable(name="trend_2026", description="ENSO/volcano-free baseline",
                           lower_bound=1.40, upper_bound=1.66, unit="C"),
        ContinuousVariable(name="enso_oni_2026", description="2026 annual-mean ONI",
                           lower_bound=-2.0, upper_bound=2.0, unit="index"),
        ContinuousVariable(name="volcanic_2026", description="volcanic cooling in 2026",
                           lower_bound=0.0, upper_bound=0.35, unit="C"),
    ]
    exprs = [
        "E[trend_2026] = 1.53", "P(trend_2026 > 1.60) < 0.05", "P(trend_2026 < 1.46) < 0.05",
        "E[enso_oni_2026] = -0.2", "P(enso_oni_2026 > 0.5) < 0.18", "P(enso_oni_2026 < -1.1) < 0.10",
        "E[volcanic_2026] = 0.015", "P(volcanic_2026 > 0.10) < 0.06",
        "anomaly_2026 = trend_2026 + 0.075*enso_oni_2026 - volcanic_2026 ~ N(0, 0.035)",
        "record_2026 = ind(anomaly_2026 > 1.60)",
    ]
    return variables, [_P(e) for e in exprs]


def fit_and_measure(flow_type="affine", n_layers=8, hidden=64, n_dummy=0,
                    num_bins=8, steps=5000, n_samples=6000, eval_samples=200000,
                    seed=0):
    """Fit one flow and measure derived tail vs. max-ent target + entropy cost."""
    variables, ests = build_forecast_model()
    b = DistributionBuilder(variables, ests, flow_type=flow_type,
                            n_layers=n_layers, hidden=hidden, n_dummy=n_dummy,
                            num_bins=num_bins)
    npar = int(b.model.net.n_params)
    b.build(target_variable="record_2026", steps=steps, n_samples=n_samples, seed=seed)
    H = float(b.entropy(60000))
    x = b.sample_dict(eval_samples, seed=seed + 1)
    tr, on, vo, an = (x["trend_2026"], x["enso_oni_2026"],
                      x["volcanic_2026"], x["anomaly_2026"])
    r = an - (tr + 0.075 * on - vo)
    rng = np.random.default_rng(seed)
    recon = (rng.permutation(tr) + 0.075 * rng.permutation(on) - rng.permutation(vo)
             + rng.normal(0, 0.035, len(tr)))
    C = np.corrcoef(np.vstack([tr, on, vo, r]))
    I = float(-0.5 * np.log(max(np.linalg.det(C), 1e-12)))
    return dict(flow_type=flow_type, n_layers=n_layers, hidden=hidden,
                n_dummy=n_dummy, num_bins=num_bins, seed=int(seed),
                params=npar, resid_sd=float(r.std()), entropy=H,
                p_flow=float(np.mean(an > RECORD)),
                p_maxent=float(np.mean(recon > RECORD)), total_corr=I)


def default_configs():
    """Affine vs. spline across a capacity ladder, plus dummy-augmented points."""
    cfgs = []
    for nl, h in [(4, 32), (8, 64), (12, 128), (16, 256), (20, 384)]:
        cfgs.append(dict(flow_type="affine", n_layers=nl, hidden=h, n_dummy=0))
    for nl, h in [(4, 32), (8, 64), (12, 128), (16, 256)]:
        cfgs.append(dict(flow_type="spline", n_layers=nl, hidden=h, n_dummy=0, num_bins=8))
    cfgs.append(dict(flow_type="affine", n_layers=16, hidden=256, n_dummy=8))
    cfgs.append(dict(flow_type="spline", n_layers=12, hidden=128, n_dummy=8, num_bins=8))
    return cfgs


def _key(row):
    return (row["flow_type"], row["n_layers"], row["hidden"],
            row["n_dummy"], row.get("num_bins", 8), row["seed"])


def run_sweep(configs=None, seeds=(0, 1, 2), steps=5000, n_samples=6000,
              eval_samples=200000, out_path="results/capacity_sweep.jsonl"):
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
            probe = {**dict(n_dummy=0, num_bins=8), **cfg, "seed": s}
            k = (probe["flow_type"], probe["n_layers"], probe["hidden"],
                 probe["n_dummy"], probe["num_bins"], s)
            if k in done:
                continue
            row = fit_and_measure(steps=steps, n_samples=n_samples,
                                  eval_samples=eval_samples, seed=s, **cfg)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            print(f"{row['flow_type']:6} L{row['n_layers']:<2} h{row['hidden']:<3} "
                  f"d{row['n_dummy']} seed{s} | {row['params']:>7} par | "
                  f"resid={row['resid_sd']:.4f} P_flow={row['p_flow']:.4f} "
                  f"P_maxent={row['p_maxent']:.4f} I={row['total_corr']:.4f} "
                  f"({time.time() - t0:.0f}s)")
    return [json.loads(l) for l in open(out_path)]


def plot_results(rows_or_path="results/capacity_sweep.jsonl", save="results/capacity_sweep.png"):
    """Two panels: tail recovery vs. params (affine vs. spline), and I vs. params."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    rows = rows_or_path
    if isinstance(rows_or_path, str):
        rows = [json.loads(l) for l in open(rows_or_path)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for family, color in [("affine", "tab:blue"), ("spline", "tab:red")]:
        by = defaultdict(list)
        for d in rows:
            if d["flow_type"] == family:
                by[(d["n_layers"], d["hidden"], d["n_dummy"])].append(d)
        if not by:
            continue
        order = sorted(by, key=lambda k: np.mean([d["params"] for d in by[k]]))
        pr = np.array([np.mean([d["params"] for d in by[k]]) for k in order])
        pf = np.array([np.mean([d["p_flow"] for d in by[k]]) for k in order])
        ps = np.array([np.std([d["p_flow"] for d in by[k]]) for k in order])
        pm = np.array([np.mean([d["p_maxent"] for d in by[k]]) for k in order])
        Iv = np.array([np.mean([d["total_corr"] for d in by[k]]) for k in order])
        ax1.errorbar(pr, pf, yerr=ps, marker="o", lw=2, capsize=3, color=color,
                     label=f"{family}: P_flow")
        ax1.plot(pr, pm, marker="s", lw=1, ls="--", color=color, alpha=0.5,
                 label=f"{family}: P_maxent")
        ax2.plot(pr, Iv, marker="o", lw=2, color=color, label=family)
    ax1.axhline(HAND_MC, color="gray", ls=":", label=f"hand MC ({HAND_MC})")
    ax1.set_xscale("log"); ax1.set_xlabel("flow parameters")
    ax1.set_ylabel("P(anomaly > 1.60)"); ax1.set_ylim(bottom=0)
    ax1.set_title("Tail recovery: affine vs. spline"); ax1.legend(fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlabel("flow parameters")
    ax2.set_ylabel("I  [nats sacrificed]"); ax2.axhline(0, color="gray", lw=0.7)
    ax2.set_title("Entropy the clip costs (max-ent wants 0)"); ax2.legend(fontsize=8)
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, dpi=130)
    return fig
