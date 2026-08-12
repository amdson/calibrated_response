"""Payload for the peer-reliability recovery benchmark (git-pullable).

The problem: a population of ``pop`` people, each with a latent reliability
``r_i ~ Beta(8, 2)`` (strongly weighted toward reliable).  Each person rates
``n_evals`` random others; the score is the target's true reliability plus
noise whose scale depends on the *rater's* reliability::

    s_ij = clip(r_j + N(0, sigma(r_i)), 0, 1),   sigma(r) = SIG_HI - (SIG_HI - SIG_LO) * r

so unreliable raters are also unreliable *as raters* — the classic
self-referential setting roughly solved by eigenvector methods (EigenTrust
et al.).  Ground truth is known by construction, so scoring needs no
reference posterior:

* point recovery — RMSE of the fitted posterior mean vs true ``r`` (vs the
  eigenvector-style iterative baseline and the plain mean of received evals)
* calibration    — coverage of the fitted marginals' central 80% intervals
  (want 0.80 across replications; the baselines have no intervals at all)

This is deliberately a test of the CURRENT DistributionBuilder API — no
solver changes.  Two encodings, both through the production front door:

* ``mode="fixed"``  (misspecified): every eval becomes
  ``ExpectationEstimate(r_j ~= s_ij, sd=SD_AVG)`` with one global sd — rater
  identity is discarded, so the solver cannot infer reliability and should
  land near the plain-mean baseline.  Measures how gracefully ~pop*n_evals
  mutually-inconsistent fixed-strength constraints are averaged.
* ``mode="eqn"``    (heteroscedastic, expressed in the existing equation
  language): the variance-stabilised residual is homoscedastic by
  construction::

      r_j = r_j - (r_j - s_ij) * SIG0 / sigma(r_i)   ~  N(0, SIG0)

  i.e. residual = (r_j - s_ij) * SIG0 / sigma(r_i).  When the solver thinks
  rater i is unreliable (small ``SIG0/sigma``), the residual — and hence the
  eqn_dist moment penalty — shrinks, exactly the likelihood's reweighting.
  This is the arm that can beat the baselines.  NOTE eqn_dist matches the
  residual's mean and variance across the *joint* sample, which is a maxent
  moment relaxation of the true per-observation likelihood — part of what is
  being tested.

Both modes also receive the population prior as per-person weak estimates
``E[r_i] = PRIOR_MEAN (sd = PRIOR_SD)`` — the "people are generally
reliable" elicitation.  The eigenvector baseline gets the same prior as a
pseudo-observation, so the comparison is fair.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from calibrated_response.models.variable import ContinuousVariable
from calibrated_response.models.query import (EquationEstimate,
                                              ExpectationEstimate)
from calibrated_response.maxent_sampler.distribution_builder import (
    DistributionBuilder)

# ---- generative model -------------------------------------------------------
PRIOR_A, PRIOR_B = 8.0, 2.0                    # Beta prior on r_i
PRIOR_MEAN = PRIOR_A / (PRIOR_A + PRIOR_B)                       # 0.8
PRIOR_SD = float(np.sqrt(PRIOR_A * PRIOR_B /
                         ((PRIOR_A + PRIOR_B) ** 2
                          * (PRIOR_A + PRIOR_B + 1.0))))         # ~0.121
SIG_LO, SIG_HI = 0.05, 0.40    # eval noise sd at r=1 / r=0
SIG0 = 0.15                    # scale of the variance-stabilised residual
# average eval noise under the prior — the "fixed" mode's (misspecified) sd
SD_AVG = SIG_HI - (SIG_HI - SIG_LO) * PRIOR_MEAN                 # 0.12


def sigma_of_r(r):
    """True eval-noise sd as a function of the rater's reliability."""
    return SIG_HI - (SIG_HI - SIG_LO) * r


def flip_prob(r, flip: float):
    """P(this eval is inverted) as a function of the rater's reliability.

    Quadratic in unreliability: typical prior-mass raters (r ~ 0.8) flip
    ~4% of their evals at flip=1, while a dud at r=0.3 flips ~half — the
    hard regime where "target is bad" and "rater is a flipper" are
    competing explanations (a bimodal likelihood the eigenvector baseline
    cannot express: it can down-weight an inconsistent rater but never
    *invert* one)."""
    return flip * (1.0 - r) ** 2


def _connected(pop: int, edges) -> bool:
    """Is the undirected union of the eval graph connected?"""
    adj = [[] for _ in range(pop)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    seen, stack = {0}, [0]
    while stack:
        for k in adj[stack.pop()]:
            if k not in seen:
                seen.add(k)
                stack.append(k)
    return len(seen) == pop


BAD_A, BAD_B = 2.0, 6.0        # dud-reviewer reliability prior (mean 0.25)
SIG_SYM = 0.12                 # constant eval noise in the symmetric variant


def generate(pop: int = 25, n_evals: int = 6, seed: int = 0,
             flip: float = 0.0, n_bad: int = 0, sym: bool = False):
    """Draw reliabilities, a connected random n_evals-out graph, and scores.

    ``flip`` scales per-eval inversion: with probability
    ``flip_prob(r_i, flip)`` the rater scores ``1 - r_j`` instead of
    ``r_j`` (before adding noise).  flip=0 recovers the plain
    heteroscedastic model.

    ``n_bad`` guarantees a dud subpopulation: persons ``0..n_bad-1`` get
    reliabilities from Beta(2, 6) (mean 0.25) instead of the reliable
    prior — under flip they are near-inverters, so the hard case exists on
    every seed instead of riding the Beta(8, 2) tail.  The solver's priors
    still say E[r]=0.8 for everyone (it must *discover* the duds); the
    fixed indices make inspection easy.

    ``sym=True`` is the EXACT-MIRROR variant (a pure representation test):
    flip probability ``p(r) = 1 - r`` (so ``p(1-r) = 1 - p(r)``), constant
    noise ``SIG_SYM``, and a symmetric population (50/50 mixture of
    Beta(8,2) honest and Beta(2,8) anti raters).  The likelihood is then
    exactly invariant under the global mirror ``r -> 1 - r``: the true
    posterior has two equal-mass modes, and a good sampler must show BOTH
    (two clusters in the pairwise plots).  ``flip``/``n_bad`` are ignored.
    Returns ``(r, evals)`` with ``evals`` a list of ``(i, j, s_ij)``."""
    rng = np.random.default_rng(seed)
    if sym:
        hon = rng.beta(PRIOR_A, PRIOR_B, size=pop)
        anti = rng.beta(PRIOR_B, PRIOR_A, size=pop)
        r = np.where(rng.uniform(size=pop) < 0.5, hon, anti)
    else:
        r = rng.beta(PRIOR_A, PRIOR_B, size=pop)
        if n_bad:
            r[:n_bad] = rng.beta(BAD_A, BAD_B, size=n_bad)
    for _ in range(100):
        edges = []
        for i in range(pop):
            others = np.delete(np.arange(pop), i)
            for j in rng.choice(others, size=n_evals, replace=False):
                edges.append((i, int(j)))
        if _connected(pop, edges):
            break
    else:
        raise RuntimeError("could not sample a connected eval graph")
    evals = []
    for i, j in edges:
        # NB: only touch the rng when flip/sym is on, so flip=0 draws the
        # exact same eval stream as the original generator (rows comparable)
        base = r[j]
        if sym:
            if rng.uniform() < 1.0 - r[i]:
                base = 1.0 - r[j]
            sd = SIG_SYM
        else:
            if flip and rng.uniform() < flip_prob(r[i], flip):
                base = 1.0 - r[j]
            sd = sigma_of_r(r[i])
        evals.append((i, j, float(np.clip(
            base + rng.normal(0.0, sd), 0.001, 0.999))))
    return r, evals


# ---- baselines (no solver) --------------------------------------------------

def baseline_mean(pop: int, evals) -> np.ndarray:
    """Prior-shrunk mean of received evals (no rater model)."""
    num = np.full(pop, PRIOR_MEAN / PRIOR_SD ** 2)
    den = np.full(pop, 1.0 / PRIOR_SD ** 2)
    w = 1.0 / SD_AVG ** 2
    for _, j, s in evals:
        num[j] += w * s
        den[j] += w
    return num / den


def baseline_eig(pop: int, evals, n_iter: int = 100) -> np.ndarray:
    """Eigenvector-style iterative reweighting (EigenTrust flavour).

    Alternates: consensus scores from rater-weighted received evals; rater
    weights from inverse mean-squared deviation of the rater's evals from the
    consensus.  The prior enters as a pseudo-observation, matching the
    information given to the solver.  With few evals per rater the raw MSE is
    a noisy noise estimate, so it is shrunk toward the population average
    (``n_reg`` pseudo-evals at ``SD_AVG``) — without this the baseline
    overfits its weights and loses to the plain shrunk mean."""
    r_hat = baseline_mean(pop, evals)
    by_rater = [[] for _ in range(pop)]
    for i, j, s in evals:
        by_rater[i].append((j, s))
    n_reg = 4.0
    for _ in range(n_iter):
        w = np.empty(pop)
        for i in range(pop):
            sq = [(s - r_hat[j]) ** 2 for j, s in by_rater[i]]
            d = (np.sum(sq) + n_reg * SD_AVG ** 2) / (len(sq) + n_reg)
            w[i] = 1.0 / d
        num = np.full(pop, PRIOR_MEAN / PRIOR_SD ** 2)
        den = np.full(pop, 1.0 / PRIOR_SD ** 2)
        for i, j, s in evals:
            num[j] += w[i] * s
            den[j] += w[i]
        new = num / den
        if np.max(np.abs(new - r_hat)) < 1e-9:
            r_hat = new
            break
        r_hat = new
    return r_hat


# ---- solver encodings (current builder API only) ----------------------------

def _name(i: int) -> str:
    return f"rel_{i:02d}"


def build_variables(pop: int):
    return [ContinuousVariable(name=_name(i),
                               description=f"reliability of person {i}",
                               lower_bound=0.0, upper_bound=1.0)
            for i in range(pop)]


def build_estimates(pop: int, evals, mode: str = "eqn", flip: float = 0.0,
                    sym: bool = False):
    """``flip`` here is what the *solver* is told about the flip mechanism
    (the true generator value for a correctly-specified eqn arm).  The
    two-component mixture likelihood is moment-matched: the eqn residual
    uses the flip-adjusted mean ``r_j + p(r_i)(1-2 r_j)`` and total sd
    ``(sigma(r_i)^2 + p(1-p)(1-2 r_j)^2)^0.5`` — all inside the existing
    equation grammar.  The fixed arm stays flip-blind (misspecified).

    ``sym=True``: the exact-mirror model, ``p = 1 - r_i`` and constant
    ``SIG_SYM``; the prior estimates become the SYMMETRIC ``E[r]=0.5``
    (weak) so nothing in the constraint set breaks the mirror — the
    correct posterior is the 50/50 two-mode joint by construction."""
    if sym:
        ests = [ExpectationEstimate(id=f"prior_{i}", variable=_name(i),
                                    expected_value=0.5, sd=0.25)
                for i in range(pop)]
    else:
        ests = [ExpectationEstimate(id=f"prior_{i}", variable=_name(i),
                                    expected_value=PRIOR_MEAN, sd=PRIOR_SD)
                for i in range(pop)]
    for n, (i, j, s) in enumerate(evals):
        if mode == "fixed":
            ests.append(ExpectationEstimate(
                id=f"s{n}_{i}_{j}", variable=_name(j),
                expected_value=s, sd=0.35 if sym else SD_AVG))
        elif mode == "eqn":
            ri, rj = _name(i), _name(j)
            if sym:
                # p = 1 - r_i exactly; p(1-p) = r_i(1-r_i)
                mean = f"({rj} + (1 - {ri}) * (1 - 2 * {rj}))"
                sd_tot = (f"({SIG_SYM} ** 2 "
                          f"+ {ri} * (1 - {ri}) * (1 - 2 * {rj}) ** 2) ** 0.5")
            else:
                sig_i = f"({SIG_HI} - {SIG_HI - SIG_LO} * {ri})"
                if flip:
                    p = f"({flip} * (1 - {ri}) ** 2)"
                    mean = f"({rj} + {p} * (1 - 2 * {rj}))"
                    sd_tot = (f"({sig_i} ** 2 "
                              f"+ {p} * (1 - {p}) * (1 - 2 * {rj}) ** 2) ** 0.5")
                else:
                    mean, sd_tot = rj, sig_i
            # residual = (mean - s) * SIG0 / sd_tot  ~  N(0, SIG0)
            rhs = f"{rj} - ({mean} - {s:.4f}) * {SIG0} / {sd_tot}"
            ests.append(EquationEstimate(
                id=f"s{n}_{i}_{j}", lhs=rj, rhs=rhs, noise_sd=SIG0))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return ests


# ---- fit + measure ----------------------------------------------------------

def fit_and_measure(pop: int = 25, n_evals: int = 6, mode: str = "eqn",
                    steps: int = 3000, n_samples: int = 2048,
                    eval_samples: int = 20000, seed: int = 0,
                    lr: float = 2e-3, flip: float = 0.0, n_bad: int = 0,
                    sym: bool = False, return_fit: bool = False,
                    **builder_kw):
    """Run one replication; returns the metrics row.

    With ``return_fit=True`` returns ``(row, fit)`` where ``fit`` is a dict
    holding everything needed to inspect the run by hand: the fitted
    ``builder`` (query ``builder.sample_dict(n)``, ``builder.constraint_report()``),
    the ground truth ``r``, the raw ``evals``, the readout ``samples``, and
    both baseline estimates."""
    r, evals = generate(pop, n_evals, seed, flip=flip, n_bad=n_bad, sym=sym)
    b_mean = baseline_mean(pop, evals)
    b_eig = baseline_eig(pop, evals)

    t0 = time.time()
    # each equation encodes ONE eval, so its synthetic-likelihood strength
    # is one effective observation (the builder default eqn_conf=10 is
    # tuned for elicited structural identities, not per-measurement use)
    builder_kw.setdefault("eqn_conf", 1.0)
    builder = DistributionBuilder(build_variables(pop),
                                  build_estimates(pop, evals, mode,
                                                  flip=flip, sym=sym),
                                  **builder_kw)
    assert not builder.skipped, builder.skipped
    builder.fit(steps=steps, lr=lr, n_samples=n_samples, seed=seed)
    s = builder.sample_dict(eval_samples, seed=seed + 1)
    secs = time.time() - t0

    x = np.stack([s[_name(i)] for i in range(pop)], axis=1)  # (N, pop)
    mean = x.mean(axis=0)
    q10, q90 = np.quantile(x, [0.10, 0.90], axis=0)

    def rmse(a):
        return float(np.sqrt(np.mean((a - r) ** 2)))

    # per-sample hypothesis alignment: a = cos(x - 1/2, r - 1/2).  sign(a)
    # says which global mode the sample lives in, |a| whether it commits to
    # a hypothesis at all.  In the sym variant the truth is 50/50 twin
    # modes: want balance ~ 0.5 with commit HIGH (two tight clusters);
    # commit high + balance ~ 1.0 = mode collapse; commit ~ 0 = blur
    # (samples split the difference instead of choosing).  In the normal
    # variants balance ~ 1.0 is correct.
    u, v = x - 0.5, r - 0.5
    a = (u @ v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v) + 1e-12)
    row = dict(
        pop=pop, n_evals=n_evals, mode=mode, steps=steps,
        n_samples=n_samples, seed=seed, flip=flip, n_bad=n_bad,
        sym=bool(sym), K=int(builder_kw.get("n_components", 1)),
        balance=float(np.mean(a > 0)), commit=float(np.mean(np.abs(a))),
        rmse_fit=rmse(mean), rmse_mean=rmse(b_mean), rmse_eig=rmse(b_eig),
        cover80=float(np.mean((q10 <= r) & (r <= q90))),
        width80=float(np.mean(q90 - q10)),
        secs=round(secs, 1),
    )
    if return_fit:
        return row, dict(builder=builder, r=r, evals=evals, samples=s,
                         mean=mean, q10=q10, q90=q90,
                         b_mean=b_mean, b_eig=b_eig)
    return row


# ---- sweep runner (mirrors prereq_experiment) -------------------------------

def default_configs(pop: int = 25, n_evals: int = 6, steps: int = 3000,
                    seeds=range(5)):
    return [dict(pop=pop, n_evals=n_evals, mode=mode, steps=steps, seed=s)
            for mode in ("fixed", "eqn") for s in seeds]


def quick_configs():
    return [dict(pop=10, n_evals=4, mode=mode, steps=800, n_samples=1024,
                 seed=s)
            for mode in ("fixed", "eqn") for s in range(2)]


def flip_configs(pop: int = 25, n_evals: int = 6, steps: int = 3000,
                 flip: float = 1.0, n_bad: int = 3, Ks=(1, 8, 32),
                 seeds=range(5)):
    """The hard variant: unreliable raters invert reviews before noising,
    with a guaranteed dud subpopulation (persons 0..n_bad-1).  ``Ks``
    sweeps the mixture-base size on the eqn arm — K > 1 is the mechanism
    for *keeping both hypotheses* (reliable vs inverter) in one joint:
    components can reweight across basins where a K=1 flow would have to
    transport mass through a desert.  Success criterion beyond rmse: a
    dud's marginal coming out bimodal in the pairwise plot.  The fixed arm
    stays K=1 (flip-blind; K would not help it)."""
    cfgs = [dict(pop=pop, n_evals=n_evals, mode="fixed", steps=steps,
                 flip=flip, n_bad=n_bad, seed=s) for s in seeds]
    cfgs += [dict(pop=pop, n_evals=n_evals, mode="eqn", steps=steps,
                  flip=flip, n_bad=n_bad, n_components=K, seed=s)
             for K in Ks for s in seeds]
    return cfgs


def sym_configs(pop: int = 15, n_evals: int = 6, steps: int = 3000,
                Ks=(1, 8, 32), seeds=range(3)):
    """The exact-mirror representation test: the true posterior is two
    equal-mass twin modes by construction (see ``generate(sym=True)``).
    Judge on ``balance`` (~0.5) x ``commit`` (high) and the two-cluster
    pairwise plots, NOT rmse_fit — the posterior mean is 0.5 everywhere at
    the ideal answer, so rmse_fit ~ sd(r) is expected and correct.
    eqn-only: the fixed arm has no mirror (its likelihood is unimodal)."""
    return [dict(pop=pop, n_evals=n_evals, mode="eqn", steps=steps,
                 sym=True, n_components=K, seed=s)
            for K in Ks for s in seeds]


def _key(row):
    # rows record the mixture size as "K"; configs pass it as "n_components"
    k = row.get("K", row.get("n_components", 1))
    return (row["pop"], row["n_evals"], row["mode"], row["steps"],
            row.get("flip", 0.0), row.get("n_bad", 0),
            int(bool(row.get("sym", False))), int(k), row["seed"])


def run_sweep(configs, out_path: str = "results/reliability.jsonl"):
    """Run all configs (resumable via the jsonl); returns ``(rows, fits)``.

    ``fits`` maps ``_key(row) -> `` the ``return_fit`` inspection dict of
    every run executed *this call* (resumed-over rows have no live sampler),
    so the notebook can pull out a fitted builder and poke at it:

        rows, fits = run_sweep(CONFIGS)
        fit = fits[(25, 6, "eqn", 3000, 0)]
        fit["builder"].sample_dict(10000), fit["r"], fit["b_eig"], ...
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows, done = [], set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                row = json.loads(line)
                rows.append(row)
                done.add(_key(row))
    fits = {}
    for cfg in configs:
        if _key({**cfg, "steps": cfg.get("steps", 3000)}) in done:
            continue
        row, fit = fit_and_measure(**cfg, return_fit=True)
        rows.append(row)
        fits[_key(row)] = fit
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        flip_tag = f" flip={row['flip']}" if row.get("flip") else ""
        if row.get("n_bad"):
            flip_tag += f" bad={row['n_bad']}"
        if row.get("sym"):
            flip_tag += " sym"
        if row.get("K", 1) != 1:
            flip_tag += f" K={row['K']}"
        sym_tag = (f"  bal={row['balance']:.2f} commit={row['commit']:.2f}"
                   if row.get("sym") else "")
        print(f"pop={row['pop']} m={row['n_evals']} {row['mode']:>5}"
              f"{flip_tag} seed={row['seed']}  rmse fit={row['rmse_fit']:.4f} "
              f"mean={row['rmse_mean']:.4f} eig={row['rmse_eig']:.4f}  "
              f"cover80={row['cover80']:.2f} width={row['width80']:.3f}"
              f"{sym_tag} ({row['secs']:.0f}s)")
    return rows, fits


def summarize(rows_or_path="results/reliability.jsonl"):
    if isinstance(rows_or_path, str):
        with open(rows_or_path) as fh:
            rows = [json.loads(line) for line in fh]
    else:
        rows = list(rows_or_path)
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["pop"], row["n_evals"], row["mode"],
                           row["steps"], row.get("flip", 0.0),
                           row.get("n_bad", 0),
                           int(bool(row.get("sym", False))),
                           row.get("K", 1)), []).append(row)
    print(f"{'pop':>4} {'m':>3} {'mode':>6} {'steps':>6} {'flip':>5} "
          f"{'bad':>4} {'sym':>3} {'K':>3} {'n':>3}  "
          f"{'rmse_fit':>15} {'rmse_mean':>10} {'rmse_eig':>10}  "
          f"{'cover80':>8} {'width80':>8} {'balance':>8} {'commit':>7}")
    for key in sorted(groups):
        g = groups[key]

        def ms(field):
            v = [row.get(field) for row in g]
            v = [x for x in v if x is not None]
            if not v:
                return float("nan"), float("nan")
            return float(np.mean(v)), float(np.std(v))

        f_m, f_s = ms("rmse_fit")
        print(f"{key[0]:>4} {key[1]:>3} {key[2]:>6} {key[3]:>6} {key[4]:>5} "
              f"{key[5]:>4} {key[6]:>3} {key[7]:>3} {len(g):>3}  "
              f"{f_m:>8.4f}±{f_s:<6.4f} "
              f"{ms('rmse_mean')[0]:>10.4f} {ms('rmse_eig')[0]:>10.4f}  "
              f"{ms('cover80')[0]:>8.2f} {ms('width80')[0]:>8.3f} "
              f"{ms('balance')[0]:>8.2f} {ms('commit')[0]:>7.2f}")


if __name__ == "__main__":
    run_sweep(quick_configs())
    summarize()
