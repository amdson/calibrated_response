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


def hybrid_ind_prod(terms, sharpness: float = 50.0, score_scale: float = 1.0):
    """Unbiased hard-indicator product via pathwise + score-function gradient.

    ``terms`` is a list of ``(site, threshold, direction)``.  Requires the
    loss to be built with ``constraint_loss(..., with_logq=True)`` so that
    ``x[:, -1]`` is the per-sample ``log q_theta`` at ``stop_gradient(x)``.

    Forward value: the EXACT hard product ``h`` (like ST -- no leakage floor).
    Gradient: pathwise through the soft product ``s`` PLUS the score-function
    correction ``E[(h - s - mean) * grad log q]`` -- the soft surrogate acts
    as a control variate, so the estimator is unbiased for ``grad E[h]`` and
    the REINFORCE variance is confined to the residual ``h - s``, nonzero
    only in the ~1/sharpness boundary band (this is what plain REINFORCE on
    a rare hard indicator lacks).  Mean baseline included (E[grad log q]=0,
    so it only reduces variance).

    ``score_scale`` dials the score term.  The pathwise part is the
    straight-through PRODUCT (each factor's gradient gated by the other
    factors' HARD values -- NOT the soft product's soft gating, so 0.0
    reproduces ``viol_mode="st"`` exactly, gradient and all).  The score
    residual is still ``h - soft_product`` (the ST product's own forward
    residual is identically 0), so at 1.0 the estimator is "ST pathwise +
    the full score correction" -- unbiased up to the hard-vs-soft gating
    difference, which is itself a boundary-band term.  Full strength
    diverged in the GPU A/B (marginal violations, NaNs): the score weight's
    magnitude is set by ``log q`` itself, which GROWS as the flow sharpens
    -- an unbounded positive-feedback channel.  Small ``score_scale`` keeps
    ST's stability while retaining a down-weighted reweighting channel that
    can grow density where transport is walled off."""
    def f(x):
        s = 1.0
        st = 1.0
        h = 1.0
        for site, thr, dirn in terms:
            d = (x[:, site] - thr) if dirn == "gt" else (thr - x[:, site])
            sj = jax.nn.sigmoid(sharpness * d)
            hj = (d > 0).astype(jnp.float32)
            s = s * sj
            st = st * (sj + jax.lax.stop_gradient(hj - sj))
            h = h * hj
        lq = x[:, -1]
        w = jax.lax.stop_gradient(h - s)
        t = score_scale * (w - jnp.mean(w)) * lq
        return st + (t - jax.lax.stop_gradient(t))
    return f


def build_constraints(n_prereq: int, p_prereq: float = P_PREREQ,
                      k_obs: float = K_OBS, k_hard: float = K_HARD,
                      viol_mode: str = "st", beta: float = 0.0):
    """Sites 0..m-1 are prerequisites, site m is the event.

    ``viol_mode="st"`` (default) scores the implications with
    straight-through indicators: the forward violation probability is EXACT
    (no soft-leakage floor at any sharpness -- see the VIOL_SHARP note), and
    gradients keep default-sharpness support.  ``"soft"`` keeps the
    sharpness-``VIOL_SHARP`` sigmoid scoring for A/B.  ``"hybrid"`` scores
    with :func:`hybrid_ind_prod`: exact forward like ST, but the gradient is
    UNBIASED (pathwise-soft + score-function residual through ``log q``);
    requires ``constraint_loss(..., with_logq=True)``.  A numeric suffix
    scales the score term, e.g. ``"hybrid0.1"`` -- the ST<->hybrid dial
    (see :func:`hybrid_ind_prod`; full strength diverged in the GPU A/B).

    ``beta`` is the prob_nll likelihood-tempering exponent, applied to ALL
    constraints (that's the point -- one global knob whose effect is
    per-constraint automatic): 0 is the plain k*KL, 1 cancels k.  Annealing
    it 1 -> 0 across fit phases is the power-posterior warmup."""
    if viol_mode.startswith("hybrid"):
        lam = float(viol_mode[6:]) if viol_mode[6:] else 1.0
        csts = []
        for i in range(n_prereq):
            csts.append(("prob_nll", soft_gt(i, 0.5), p_prereq, k_obs, beta))
            violated = hybrid_ind_prod(
                [(n_prereq, 0.5, "gt"), (i, 0.5, "lt")], score_scale=lam)
            csts.append(("prob_nll", violated, P_IMPOSSIBLE, k_hard, beta))
        return csts
    if viol_mode == "st":
        g_e = st_ind(n_prereq, 0.5, direction="gt")
        not_as = [st_ind(i, 0.5, direction="lt") for i in range(n_prereq)]
    else:
        g_e = soft_gt(n_prereq, 0.5, VIOL_SHARP)
        not_as = [soft_lt(i, 0.5, VIOL_SHARP) for i in range(n_prereq)]
    csts = []
    for i in range(n_prereq):
        csts.append(("prob_nll", soft_gt(i, 0.5), p_prereq, k_obs, beta))

        def violated(x, not_a=not_as[i]):  # e holds but prerequisite i doesn't
            return g_e(x) * not_a(x)
        csts.append(("prob_nll", violated, P_IMPOSSIBLE, k_hard, beta))
    return csts


def fit_and_measure(n_prereq=4, n_components=1, n_layers=6, hidden=64,
                    base_spread=1.0, p_prereq=P_PREREQ, steps=3000,
                    n_samples=4096, eval_samples=200_000, seed=0,
                    warmup_steps=0, warmup_k_hard=8.0, viol_mode="st",
                    beta_phases=0, beta_ramp=0.0):
    """Fit one model and measure recovery of the analytic maxent solution.

    ``warmup_steps > 0`` anneals the implications: that many steps at
    ``k_hard=warmup_k_hard`` first, then ``steps`` at full K_HARD from the
    warm-started params.  Rationale: at init e is ~uniform, so soft violation
    starts ~0.1 and the full-strength implication term (loss ~100) crushes e
    globally in the first ~100 steps; the vacuous basin is then an attractor
    (verified: longer single-phase training DECREASES P(e|C)).  A weak-k
    phase lets the corner structure form before the wall goes up.

    ``beta_phases >= 2`` uses likelihood-tempering annealing INSTEAD: the
    prob_nll beta exponent staircases linspace(1, 0, beta_phases) across
    equal warm-started phases splitting ``steps`` (last phase beta=0 = the
    true objective).  One global knob, per-constraint-automatic ramp; see
    the prob_nll comment in model.py.

    ``beta_ramp`` in (0, 1) is the PER-STEP version (the phases->infinity
    limit, one jit compile): beta declines linearly 1 -> 0 over the first
    ``beta_ramp`` fraction of ``steps``, then holds at 0 (the true
    objective) for the tail -- the hold matters because any beta > 0
    under-enforces at equilibrium.  lr pairs with it as a step schedule
    (2e-3 during the ramp, 1e-3 in the hold, mirroring the staircase's
    phase-1/rest split).  The three annealing schemes are mutually
    exclusive (asserted)."""
    assert sum(map(bool, (warmup_steps, beta_phases, beta_ramp))) <= 1, \
        "pick one annealing scheme: warmup_steps, beta_phases, or beta_ramp"
    sites = ([ContinuousVar(f"a{i}", 0.0, 1.0, 16) for i in range(n_prereq)]
             + [ContinuousVar("e", 0.0, 1.0, 16)])
    m = FlowSamplerModel(sites, n_layers=n_layers, hidden=hidden,
                         n_components=n_components, base_spread=base_spread)
    npar = int(m.net.n_params)
    wlq = viol_mode.startswith("hybrid")  # score channel needs the log-q column
    loss = m.constraint_loss(
        build_constraints(n_prereq, p_prereq, viol_mode=viol_mode),
        entropy_reg=1.0, n_samples=n_samples, with_logq=wlq)
    t0 = time.time()
    if beta_ramp:
        import optax
        ramp = max(1, int(float(beta_ramp) * steps))
        sched = lambda step: jnp.maximum(0.0, 1.0 - step / ramp)
        loss_r = m.constraint_loss(
            build_constraints(n_prereq, p_prereq, viol_mode=viol_mode),
            entropy_reg=1.0, n_samples=n_samples, with_logq=wlq,
            beta_schedule=sched)
        lr_sched = optax.join_schedules(
            [optax.constant_schedule(2e-3), optax.constant_schedule(1e-3)],
            [ramp])
        params, hist = m.optimize(loss_r, seed=seed, steps=steps,
                                  lr=lr_sched, grad_clip=5.0)
    elif beta_phases:
        betas = np.linspace(1.0, 0.0, int(beta_phases))
        per = max(1, steps // len(betas))
        params, hist = None, None
        for j, b in enumerate(betas):
            loss_b = m.constraint_loss(
                build_constraints(n_prereq, p_prereq, viol_mode=viol_mode,
                                  beta=float(b)),
                entropy_reg=1.0, n_samples=n_samples, with_logq=wlq)
            params, hist = m.optimize(loss_b, seed=seed + j, steps=per,
                                      lr=2e-3 if j == 0 else 1e-3,
                                      grad_clip=5.0, init=params)
    elif warmup_steps:
        weak = m.constraint_loss(
            build_constraints(n_prereq, p_prereq, k_hard=warmup_k_hard,
                              viol_mode=viol_mode),
            entropy_reg=1.0, n_samples=n_samples, with_logq=wlq)
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
        beta_phases=int(beta_phases), beta_ramp=float(beta_ramp),
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
        # (hybrid arms removed: the score channel was unstable at every
        #  tested score_scale, 0.03-1.0 -- see the estimator A/B results
        #  and the solver_additions.md postmortem.  ST + warmup won.)
        for K in (16, 64):                       # squashed-GMM: no coupling layers
            cfgs.append(dict(n_prereq=m, n_components=K, n_layers=0, hidden=64))
    return cfgs


def estimator_ab_configs(ms=(4, 6)):
    """Focused estimator comparison: soft vs ST vs hybrid violation scoring,
    each with and without the weak-k warmup, K=1 affine 6x64.  The question
    this grid answers (which the K-axis sweep can't isolate): does the
    unbiased pathwise+score gradient beat the biased ST / sharp-soft ones,
    and does it interact with the init-slam warmup?  m=2 is omitted by
    default (essentially solved by every arm)."""
    cfgs = []
    for m in ms:
        for mode in ("soft", "st", "hybrid"):
            for wu in (0, 1000):
                cfgs.append(dict(n_prereq=m, n_components=1, n_layers=6,
                                 hidden=64, warmup_steps=wu, viol_mode=mode))
    return cfgs


def beta_anneal_configs(ms=(4, 6), phases=(2, 4, 8)):
    """Likelihood-tempering (beta) annealing vs the k-warmup baseline, ST
    features throughout.  beta staircases linspace(1, 0, phases) over equal
    warm-started phases within the same total step budget -- so phases=2 is
    the cheapest possible schedule (half at beta=1, half at the true
    objective) and phases=8 approximates a smooth ramp.  Compare against
    the ``st wu1000`` rows: same budget, hand-tuned k-warmup."""
    return [dict(n_prereq=m, n_components=1, n_layers=6, hidden=64,
                 beta_phases=p, viol_mode="st")
            for m in ms for p in phases]


def beta_ramp_configs(ms=(4, 6), fracs=(0.5, 0.7, 0.85)):
    """Per-step linear beta ramp (the smooth limit of beta_anneal_configs;
    one jit compile per fit).  ``fracs`` is the fraction of the step budget
    spent ramping 1 -> 0; the rest holds at the true objective.  Compare
    against the b8 staircase and st wu1000 rows at the same total steps."""
    return [dict(n_prereq=m, n_components=1, n_layers=6, hidden=64,
                 beta_ramp=f, viol_mode="st")
            for m in ms for f in fracs]


def tamed_hybrid_configs(ms=(4, 6), scales=(0.3, 0.1, 0.03)):
    """The ST<->hybrid dial: score_scale=0 IS straight-through, 1.0 diverged
    in the GPU A/B (unbounded log-q feedback).  These arms probe whether a
    down-weighted score channel keeps ST's stability while adding enough
    reweighting to beat it.  All with warmup (the known-good pairing);
    compare against the ``st`` wu1000 rows from the A/B run."""
    return [dict(n_prereq=m, n_components=1, n_layers=6, hidden=64,
                 warmup_steps=1000, viol_mode=f"hybrid{lam}")
            for m in ms for lam in scales]


def summarize_ab(rows_or_path="results/estimator_ab.jsonl"):
    """Mean +/- sd over seeds per (m, viol_mode, warmup) arm, printed as a
    table.  Headline: p_e_given_c (want 0.5); also P(e|~C) (want ~0, the
    implication check), P(C)/maxent and max prerequisite corr vs its true
    (nonzero!) target."""
    if isinstance(rows_or_path, str):
        rows = [json.loads(l) for l in open(rows_or_path)]
    else:
        rows = list(rows_or_path)
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["n_prereq"], r.get("viol_mode", "soft"),
            r.get("warmup_steps", 0), r.get("beta_phases", 0),
            r.get("beta_ramp", 0.0))].append(r)
    hdr = (f"{'m':>2} {'mode':<10} {'anneal':>7} {'n':>2} | "
           f"{'P(e|C)':>13} {'P(e|~C)':>9} {'P(C)ratio':>13} "
           f"{'corr (true)':>13}")
    print(hdr)
    print("-" * len(hdr))
    for (m, mode, wu, bp, br) in sorted(by):
        rs = by[(m, mode, wu, bp, br)]
        anneal = (f"wu{wu}" if wu else f"beta{bp}" if bp
                  else f"ramp{br}" if br else "-")
        stat = lambda key: (np.mean([r[key] for r in rs]),
                            np.std([r[key] for r in rs]))
        pec, pec_sd = stat("p_e_given_c")
        pen, _ = stat("p_e_given_notc")
        pcr, pcr_sd = stat("p_c_ratio")
        cor, _ = stat("max_corr")
        print(f"{m:>2} {mode:<10} {anneal:>7} {len(rs):>2} | "
              f"{pec:6.3f} +-{pec_sd:5.3f} {pen:9.4f} "
              f"{pcr:6.3f} +-{pcr_sd:5.3f} {cor:6.3f} "
              f"({rs[0]['corr_true']:.3f})")
    return by


def _key(row):
    return (row["n_prereq"], row["n_layers"], row["hidden"],
            row["n_components"], row.get("warmup_steps", 0),
            row.get("viol_mode", "soft"), row.get("beta_phases", 0),
            row.get("beta_ramp", 0.0), row["seed"])


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
            bp = f" b{row['beta_phases']}" if row.get("beta_phases") else ""
            br = f" br{row['beta_ramp']}" if row.get("beta_ramp") else ""
            print(f"m={row['n_prereq']} L{row['n_layers']:<2} "
                  f"K{row['n_components']:<3} {row['viol_mode']:<6}{wu}{bp}{br} seed{s} | "
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
