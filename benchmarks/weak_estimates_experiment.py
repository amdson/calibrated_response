"""Payload for the weak-estimates fusion benchmark (git-pullable).

The world is a FIXED bivariate normal (X, Y) — mean, sds, and correlation
known to the scorer but never shown to the solver.  The solver instead
receives dozens of WEAK empirical estimates, each computed from just
``N_PER_EST = 5`` iid draws of the true distribution: empirical ``E[X]``,
``E[X*Y]``, ``E[X^2]``, ``P(X > 2)``, ``P(X > Y)``, ...  Every estimate is
weighted honestly ("oracle weighting"): its ``sd`` is the true sampling sd
of that statistic at n=5, so a Bayes-optimal fuser's posterior KL to the
truth should fall roughly like 1/m in the number of estimates m.

The claim under test: **KL(fit ‖ true) decreases (near-)monotonically as
estimates accumulate** — i.e. the ``expect_nll`` machinery genuinely FUSES
weak evidence (each estimate's variance-channel adds curvature) instead of
averaging it.  Repeated draws of the same functional must concentrate
sqrt(n)-style; disjoint functionals must jointly pin the correlation.

Encodings, all through the production ``DistributionBuilder`` front door:

* moments of a single variable — ``ExpectationEstimate`` (routes to
  ``expect_nll``; ``sd`` = true sampling sd of the 5-point mean).
* composite functionals (``X*Y``, ``X^2``, ``X+Y``, ...) — an auxiliary
  variable pinned by a *deterministic* ``EquationEstimate`` link
  (``xy = x * y``), then an ``ExpectationEstimate`` on the aux var.  This
  doubles as a load test of deterministic Fermi links.
* tail events — ``ProbabilityEstimate`` with an inequality proposition;
  ``sd`` is a LOG-ODDS width (delta method: ``1/sqrt(n p (1-p))`` at the
  true p).  The 5-point frequency is Jeffreys-smoothed
  (``(k + 0.5)/(n + 1)``) so 0/5 and 5/5 stay finite.

Scoring (no reference posterior needed — the truth is closed-form):

* ``kl_gauss`` — moment-match the fitted (x, y) samples to a Gaussian,
  closed-form KL to the true Gaussian.  Exact if the fit is Gaussian.
* ``kl_knn``   — nonparametric ``KL(q ‖ p) = -H(q) - E_q[log p]`` with
  H(q) from the Kozachenko–Leonenko 1-NN estimator on the fitted 2-D
  samples (catches non-Gaussian pathology kl_gauss would forgive).
* ``kl_uniform_ref()`` — KL of the no-information maxent answer (uniform
  over the box), the m=0 anchor every curve should start near.

Each seed draws ONE pool of ``pool_size`` estimates and the m-ladder takes
prefixes of it, so "m=16" is literally "m=8 plus eight more" — the
monotonicity claim is about *adding* evidence, not resampling it.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gammaln

from calibrated_response.models.query import (EquationEstimate,
                                              ExpectationEstimate,
                                              InequalityProposition,
                                              ProbabilityEstimate)
from calibrated_response.models.variable import ContinuousVariable
from calibrated_response.maxent_sampler.distribution_builder import (
    DistributionBuilder)

# ---- the true world ---------------------------------------------------------
MU = np.array([0.5, -0.5])
SDS = np.array([1.0, 1.5])
RHO = 0.6
COV = np.array([[SDS[0] ** 2, RHO * SDS[0] * SDS[1]],
                [RHO * SDS[0] * SDS[1], SDS[1] ** 2]])
N_PER_EST = 5                  # iid draws behind each weak estimate


def sample_true(n: int, rng) -> np.ndarray:
    return rng.multivariate_normal(MU, COV, size=n)


# transforms define both the aux variables and the estimate functionals
_TRANSFORMS = {
    "x":   lambda p: p[:, 0],
    "y":   lambda p: p[:, 1],
    "xx":  lambda p: p[:, 0] * p[:, 0],
    "yy":  lambda p: p[:, 1] * p[:, 1],
    "xy":  lambda p: p[:, 0] * p[:, 1],
    "xpy": lambda p: p[:, 0] + p[:, 1],
    "xmy": lambda p: p[:, 0] - p[:, 1],
}
# deterministic identities pinning each aux variable to (x, y)
_LINKS = {"xx": "x * x", "yy": "y * y", "xy": "x * y",
          "xpy": "x + y", "xmy": "x - y"}

# oracle moments per transform from one big FIXED MC pool (used only for
# honest estimate weights and true event probabilities — never as targets)
_POOL = sample_true(400_000, np.random.default_rng(20260812))
ORACLE = {v: (float(f(_POOL).mean()), float(f(_POOL).std()))
          for v, f in _TRANSFORMS.items()}

# the functional menu the pool samples from (with replacement) — repeats of
# the same functional are the sqrt(n)-concentration test, distinct ones the
# fusion test.  P(x > 2) is deliberately a ~7% tail event.
MENU = ([("E", v, None) for v in _TRANSFORMS] +
        [("P", "x", t) for t in (-0.5, 0.5, 1.5, 2.0)] +
        [("P", "y", t) for t in (-2.0, -0.5, 1.0)] +
        [("P", "xy", 0.0), ("P", "xpy", 0.0),
         ("P", "xmy", 0.0),            # == P(X > Y)
         ("P", "xmy", 1.0),
         ("P", "xx", 1.0), ("P", "yy", 2.25)])


def true_p(v: str, t: float) -> float:
    return float(np.mean(_TRANSFORMS[v](_POOL) > t))


# ---- variables + estimates --------------------------------------------------

def _interval(v: str) -> tuple[float, float]:
    """Domain by interval arithmetic over the (x, y) box, so a deterministic
    link can never be forced outside its variable's domain."""
    lox, hix = MU[0] - 5 * SDS[0], MU[0] + 5 * SDS[0]
    loy, hiy = MU[1] - 5 * SDS[1], MU[1] + 5 * SDS[1]
    if v == "x":
        return lox, hix
    if v == "y":
        return loy, hiy
    cx, cy = np.array([lox, hix]), np.array([loy, hiy])
    if v == "xx":
        return 0.0, float(max(cx ** 2))
    if v == "yy":
        return 0.0, float(max(cy ** 2))
    if v == "xy":
        prods = np.outer(cx, cy).ravel()
        return float(prods.min()), float(prods.max())
    if v == "xpy":
        return float(lox + loy), float(hix + hiy)
    if v == "xmy":
        return float(lox - hiy), float(hix - loy)
    raise ValueError(v)


def build_variables(encode: str = "aux"):
    """``encode="aux"``: (x, y) plus one variable per composite functional,
    pinned by deterministic links.  ``encode="expr"``: just (x, y) — the
    composite functionals become expression subjects (``E[x * y]``) that
    constrain the 2-D joint directly, no links to leak through."""
    names = _TRANSFORMS if encode == "aux" else ("x", "y")
    out = []
    for v in names:
        lo, hi = _interval(v)
        desc = {"x": "first coordinate", "y": "second coordinate"}.get(
            v, f"auxiliary: {_LINKS.get(v, v)}")
        out.append(ContinuousVariable(name=v, description=desc,
                                      lower_bound=lo, upper_bound=hi))
    return out


def build_links():
    return [EquationEstimate(id=f"def_{v}", lhs=v, rhs=expr)
            for v, expr in _LINKS.items()]


def draw_pool(pool_size: int, seed: int, encode: str = "aux"):
    """One pool of weak estimates: each picks a random menu functional and
    a FRESH 5-point sample of the true distribution.

    ``encode`` picks the subject form only — the rng stream, functionals,
    values, and sds are identical, so aux-vs-expr is a controlled A/B:
    ``"aux"`` targets the linked auxiliary variable (``E[xy]``), ``"expr"``
    the expression itself (``E[x * y]``) via the expression-quantity path."""
    rng = np.random.default_rng(seed)
    ests = []
    for n in range(pool_size):
        kind, v, t = MENU[int(rng.integers(len(MENU)))]
        z = _TRANSFORMS[v](sample_true(N_PER_EST, rng))
        subject = v if encode == "aux" else _LINKS.get(v, v)
        if kind == "E":
            _, sd_v = ORACLE[v]
            ests.append(ExpectationEstimate(
                id=f"e{n:03d}_E_{v}", variable=subject,
                expected_value=float(z.mean()),
                sd=sd_v / np.sqrt(N_PER_EST)))
        else:
            p = true_p(v, t)
            phat = (float(np.sum(z > t)) + 0.5) / (N_PER_EST + 1.0)
            ests.append(ProbabilityEstimate(
                id=f"e{n:03d}_P_{v}_gt_{t:g}",
                proposition=InequalityProposition(
                    variable=subject, threshold=float(t),
                    is_lower_bound=True),
                probability=phat,
                sd=1.0 / np.sqrt(N_PER_EST * p * (1.0 - p))))
    return ests


# ---- scoring ----------------------------------------------------------------

def _true_logpdf(pts: np.ndarray) -> np.ndarray:
    d = pts - MU
    prec = np.linalg.inv(COV)
    return (-0.5 * np.einsum("ni,ij,nj->n", d, prec, d)
            - np.log(2.0 * np.pi) - 0.5 * np.log(np.linalg.det(COV)))


def kl_gauss(pts: np.ndarray) -> float:
    """Closed-form KL(N(moments of pts) ‖ true N)."""
    mq, cq = pts.mean(axis=0), np.cov(pts.T)
    prec = np.linalg.inv(COV)
    d = mq - MU
    return float(0.5 * (np.trace(prec @ cq) + d @ prec @ d - 2.0
                        + np.log(np.linalg.det(COV) / np.linalg.det(cq))))


def kl_knn(pts: np.ndarray, cap: int = 20_000) -> float:
    """KL(q ‖ true) = -H(q) - E_q[log p]; H via Kozachenko–Leonenko 1-NN."""
    pts = pts[:cap]
    n, d = pts.shape
    r = cKDTree(pts).query(pts, k=2)[0][:, 1]
    r = np.maximum(r, 1e-12)
    h = (d * float(np.mean(np.log(r))) + np.log(n - 1)
         + d / 2.0 * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
         + np.euler_gamma)
    return float(-h - np.mean(_true_logpdf(pts)))


def kl_uniform_ref(n: int = 200_000) -> float:
    """The m=0 anchor: KL(uniform over the (x, y) box ‖ true)."""
    rng = np.random.default_rng(7)
    (lox, hix), (loy, hiy) = _interval("x"), _interval("y")
    u = np.stack([rng.uniform(lox, hix, n), rng.uniform(loy, hiy, n)], axis=1)
    return float(-np.log((hix - lox) * (hiy - loy)) - np.mean(_true_logpdf(u)))


# ---- fit + measure ----------------------------------------------------------

def fit_and_measure(m: int, pool_size: int = 64, steps: int = 3000,
                    n_samples: int = 2048, eval_samples: int = 30_000,
                    seed: int = 0, lr: float = 2e-3, encode: str = "aux",
                    return_fit: bool = False, **builder_kw):
    """Fit on the first ``m`` estimates of seed's pool; score KL to truth.

    ``m = 0`` fits the deterministic links alone (``encode="aux"``) or the
    bare unconstrained box (``encode="expr"``) — the maxent no-information
    answer, which should land near ``kl_uniform_ref()`` on (x, y)."""
    assert m <= pool_size, (m, pool_size)
    pool = draw_pool(pool_size, seed, encode=encode)
    used = pool[:m]

    t0 = time.time()
    links = build_links() if encode == "aux" else []
    builder = DistributionBuilder(build_variables(encode), links + used,
                                  **builder_kw)
    assert not builder.skipped, builder.skipped
    builder.fit(steps=steps, lr=lr, n_samples=n_samples, seed=seed)
    s = builder.sample_dict(eval_samples, seed=seed + 1)
    secs = time.time() - t0

    pts = np.stack([s["x"], s["y"]], axis=1)
    mq, cq = pts.mean(axis=0), np.cov(pts.T)
    # deterministic-link fidelity: RMS(aux - f(x, y)) in units of sd(f).
    # expr mode has no links, hence nothing to leak — record None.
    link_err = (float(np.mean(
        [np.sqrt(np.mean((s[v] - _TRANSFORMS[v](pts)) ** 2)) / ORACLE[v][1]
         for v in _LINKS])) if encode == "aux" else None)

    row = dict(
        m=m, pool_size=pool_size, steps=steps, n_samples=n_samples,
        seed=seed, encode=encode,
        K=int(builder_kw.get("n_components", 1)),
        kl_gauss=kl_gauss(pts), kl_knn=kl_knn(pts),
        err_mean_x=float(mq[0] - MU[0]), err_mean_y=float(mq[1] - MU[1]),
        err_sd_x=float(np.sqrt(cq[0, 0]) - SDS[0]),
        err_sd_y=float(np.sqrt(cq[1, 1]) - SDS[1]),
        err_corr=float(cq[0, 1] / np.sqrt(cq[0, 0] * cq[1, 1]) - RHO),
        link_err=link_err,
        n_expect=sum(1 for e in used if isinstance(e, ExpectationEstimate)),
        n_prob=sum(1 for e in used if isinstance(e, ProbabilityEstimate)),
        secs=round(secs, 1),
    )
    if return_fit:
        return row, dict(builder=builder, pool=pool, used=used,
                         samples=s, pts=pts)
    return row


# ---- sweep runner (mirrors reliability_experiment) ---------------------------

def full_configs(ms=(0, 2, 4, 8, 16, 32, 64), pool_size: int = 64,
                 steps: int = 3000, seeds=range(3), encode: str = "aux"):
    """The m-ladder per seed: prefixes of one pool, so each rung strictly
    ADDS evidence to the previous one."""
    return [dict(m=m, pool_size=pool_size, steps=steps, seed=s,
                 encode=encode)
            for s in seeds for m in ms]


def expr_configs(**kw):
    """The A/B arm: identical pools, but composite functionals constrain the
    2-D joint DIRECTLY as expressions (``E[x * y]``) — no aux variables, no
    deterministic links to leak through.  Compare against the aux rows on
    kl_* and err_corr; the aux arm's link_err is the suspected culprit."""
    return full_configs(encode="expr", **kw)


def quick_configs():
    return [dict(m=m, pool_size=16, steps=800, n_samples=1024, seed=0,
                 encode=enc)
            for enc in ("aux", "expr") for m in (0, 4, 16)]


def _key(row):
    k = row.get("K", row.get("n_components", 1))
    return (row["pool_size"], row["m"], row["steps"], int(k), row["seed"],
            row.get("encode", "aux"))


def run_sweep(configs, out_path: str = "results/weak_estimates.jsonl"):
    """Run all configs (resumable via the jsonl); returns ``(rows, fits)``.

    ``fits`` maps ``_key(row) ->`` the ``return_fit`` inspection dict of every
    run executed *this call* (resumed-over rows have no live sampler)."""
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
        k_tag = f" K={row['K']}" if row.get("K", 1) != 1 else ""
        le = row.get("link_err")
        link_tag = f" link={le:.3f}" if le is not None else ""
        print(f"m={row['m']:>3} {row.get('encode', 'aux'):>4}{k_tag} "
              f"seed={row['seed']}  "
              f"kl_gauss={row['kl_gauss']:.4f} kl_knn={row['kl_knn']:.4f}  "
              f"corr_err={row['err_corr']:+.3f}{link_tag} "
              f"({row['secs']:.0f}s)")
    return rows, fits


def summarize(rows_or_path="results/weak_estimates.jsonl"):
    if isinstance(rows_or_path, str):
        with open(rows_or_path) as fh:
            rows = [json.loads(line) for line in fh]
    else:
        rows = list(rows_or_path)
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["pool_size"], row["steps"],
                           row.get("K", 1), row.get("encode", "aux"),
                           row["m"]), []).append(row)

    print(f"uniform-box reference: KL = {kl_uniform_ref():.4f}\n")
    print(f"{'pool':>5} {'steps':>6} {'K':>3} {'enc':>4} {'m':>3} {'n':>3}  "
          f"{'kl_gauss':>16} {'kl_knn':>16}  "
          f"{'err_corr':>9} {'link_err':>9}")
    for key in sorted(groups):
        g = groups[key]

        def ms(field):
            v = [row.get(field) for row in g]
            v = [x for x in v if x is not None]
            if not v:
                return float("nan"), float("nan")
            return float(np.mean(v)), float(np.std(v))

        kg, kgs = ms("kl_gauss")
        kk, kks = ms("kl_knn")
        print(f"{key[0]:>5} {key[1]:>6} {key[2]:>3} {key[3]:>4} {key[4]:>3} "
              f"{len(g):>3}  "
              f"{kg:>9.4f}±{kgs:<6.4f} {kk:>9.4f}±{kks:<6.4f}  "
              f"{ms('err_corr')[0]:>+9.3f} {ms('link_err')[0]:>9.3f}")

    # per-seed monotonicity of the m-ladder (the headline claim)
    by_seed: dict = {}
    for row in rows:
        by_seed.setdefault((row["pool_size"], row["steps"],
                            row.get("K", 1), row.get("encode", "aux"),
                            row["seed"]), []).append(row)
    for metric in ("kl_gauss", "kl_knn"):
        drops = tot = 0
        for g in by_seed.values():
            g = sorted(g, key=lambda r: r["m"])
            for a, b in zip(g, g[1:]):
                tot += 1
                drops += b[metric] < a[metric]
        if tot:
            print(f"\n{metric}: {drops}/{tot} ladder steps decreased "
                  f"({drops / tot:.0%} monotone)")


if __name__ == "__main__":
    run_sweep(quick_configs())
    summarize()
