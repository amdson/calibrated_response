"""Engine adapters: constraint bag in, fitted joint distribution out.

Every engine implements

    fit(enc, bag, seed) -> fitted

and every fitted model implements

    log_prob_rows(Xi)          (N,) log p of integer-binned rows
    marginal(var)              (d,) bin mass
    pair_marginal(var_a, var_b)  (d_a, d_b) bin mass

so metrics.py never sees engine internals. Add an engine by appending to
ENGINES at the bottom.
"""

from __future__ import annotations

import numpy as np

from .constraints import (CondExpectConstraint, CondProbConstraint,
                          Interval, MarginalConstraint, ProbConstraint)
from .encoding import TableEncoder


# ======================================================================
# Independent null: product of the (noisy) marginal constraints.
# Any engine that can't beat this is not using the correlation constraints.
# ======================================================================

class IndependentEngine:
    name = "independent"

    def fit(self, enc: TableEncoder, bag, seed: int = 0):
        margs = {c.var: np.asarray(c.probs) for c in bag
                 if isinstance(c, MarginalConstraint)}
        # uniform for any variable the bag doesn't constrain
        table = [margs.get(v, np.full(d, 1.0 / d))
                 for v, d in zip(enc.names, enc.dims)]
        return _IndependentFit(enc, table)


class _IndependentFit:
    def __init__(self, enc, table):
        self.enc, self.table = enc, [np.clip(t, 1e-12, None) for t in table]

    def log_prob_rows(self, Xi):
        return sum(np.log(t[Xi[:, i]]) for i, t in enumerate(self.table))

    def marginal(self, var):
        return self.table[self.enc.site[var]]

    def pair_marginal(self, a, b):
        return np.outer(self.marginal(a), self.marginal(b))


# ======================================================================
# Tensor chain
# ======================================================================

class TensorChainEngine:
    """Born (or nonneg) TensorChain fit to the bag by constraint SSE + regs.

    ``regularizers`` is passed straight to
    :func:`calibrated_response.tn.losses.combined_loss` — entries are
    ``(name_or_fn, weight)``.
    """

    def __init__(self, bond_dim: int = 8, kind: str = "born",
                 regularizers=(("entropy", 1e-3),), marginal_weight: float = 1.0,
                 backend: str = "adam", steps: int = 2000, lr: float = 2e-2,
                 init: str = "uniform"):
        self.name = f"tn_{kind}_r{bond_dim}"
        self.bond_dim, self.kind = bond_dim, kind
        self.regularizers = list(regularizers)
        self.marginal_weight = marginal_weight
        self.backend, self.steps, self.lr, self.init = backend, steps, lr, init

    # ---- bag -> native constraint tuples -----------------------------
    def _mask(self, enc, iv: Interval):
        m = np.zeros(enc.dims[enc.site[iv.var]], dtype=np.float32)
        m[iv.lo:iv.hi] = 1.0
        return m

    def _event(self, enc, ivs):
        return {enc.site[iv.var]: self._mask(enc, iv) for iv in ivs}

    def _convert(self, enc, bag):
        csts = []
        for c in bag:
            if isinstance(c, MarginalConstraint):
                csts.append(("kl", enc.site[c.var], np.asarray(c.probs),
                             self.marginal_weight))
            elif isinstance(c, ProbConstraint):
                csts.append(("prob", self._event(enc, c.events), c.target))
            elif isinstance(c, CondProbConstraint):
                csts.append(("cond", self._event(enc, c.event),
                             self._event(enc, c.given), c.target))
            elif isinstance(c, CondExpectConstraint):
                csts.append(("cond_expect", enc.site[c.var],
                             self._event(enc, c.given), c.target))
            else:
                raise TypeError(f"unknown constraint {type(c).__name__}")
        return csts

    def fit(self, enc: TableEncoder, bag, seed: int = 0):
        from calibrated_response.tn.chain import TensorChain
        from calibrated_response.tn.losses import combined_loss

        model = TensorChain(enc.tn_vars(), bond_dim=self.bond_dim, kind=self.kind)
        loss = combined_loss(model, self._convert(enc, bag), self.regularizers)
        kw = dict(steps=self.steps, lr=self.lr) if self.backend == "adam" else {}
        params, history = model.optimize(
            loss, backend=self.backend, seed=seed,
            init=model.init_params(seed=seed, init=self.init), **kw)
        return _TensorChainFit(enc, model, params, history)


class _TensorChainFit:
    def __init__(self, enc, model, params, history):
        self.enc, self.model, self.params, self.history = enc, model, params, history

    def log_prob_rows(self, Xi):
        return np.asarray(self.model.log_prob_idx(self.params, Xi))

    def marginal(self, var):
        return np.asarray(self.model.joint_marginal(
            self.params, (self.enc.site[var],)))

    def pair_marginal(self, a, b):
        i, j = self.enc.site[a], self.enc.site[b]
        m = np.asarray(self.model.joint_marginal(self.params, (i, j)))
        return m if i < j else m.T           # joint_marginal orders by site index


# ======================================================================
# Flow sampler (invertible RealNVP — exact entropy, exact density)
# ======================================================================

class FlowEngine:
    """FlowSamplerModel fit to the bag by soft-constrained maxent on the
    encoded ``[0,1]^D`` box (``entropy_reg=1.0`` = the true maxent objective;
    the uniform box is the zero-entropy reference, so unconstrained structure
    defaults to flat/independent).

    Bag -> constraint grammar:

    - ``MarginalConstraint`` — ``marginal_mode="cdf"`` (default): pin the
      survival function ``P(x > k/d)`` at every interior bin edge with a soft
      tail indicator; ``"mmd"``: per-site MMD against samples drawn from the
      (noisy) histogram.
    - ``ProbConstraint`` -> ``("expect", conjunction of soft interval
      indicators, target)``.
    - ``CondProbConstraint`` -> ``("cond_expect", event, given, target)``.
    - ``CondExpectConstraint`` -> ``("cond_expect", moment(site), given,
      target)`` — bag targets are already in encoded [0,1] units.

    Weights are Gaussian belief widths ``w = 1/(2 sd^2)``, matching the
    LLM-facing ``maxent_sampler.DistributionBuilder``.

    Scoring: ``marginal`` / ``pair_marginal`` histogram one big cached sample;
    ``log_prob_rows`` uses the **exact** flow density (the flow is invertible)
    MC-averaged over each test row's bin box, times the bin volume.
    """

    def __init__(self, n_layers: int = 8, hidden: int = 64, s_max: float = 3.0,
                 steps: int = 3000, lr: float = 1e-3, n_samples: int = 2048,
                 entropy_reg: float = 1.0,
                 sharpness: float = 50.0, marginal_sharpness: float = 100.0,
                 prob_sd: float = 0.05, expect_sd: float = 0.05,
                 marginal_mode: str = "cdf", mmd_weight: float = 100.0,
                 mmd_ref_n: int = 512,
                 n_mc_score: int = 400_000, nll_mc: int = 8):
        self.name = f"flow_l{n_layers}"
        self.n_layers, self.hidden, self.s_max = n_layers, hidden, s_max
        self.steps, self.lr, self.n_samples = steps, lr, n_samples
        self.entropy_reg = entropy_reg
        self.sharpness, self.marginal_sharpness = sharpness, marginal_sharpness
        self.prob_sd, self.expect_sd = prob_sd, expect_sd
        self.marginal_mode = marginal_mode
        self.mmd_weight, self.mmd_ref_n = mmd_weight, mmd_ref_n
        self.n_mc_score, self.nll_mc = n_mc_score, nll_mc

    # ---- bag -> native constraint tuples -----------------------------
    def _indicator(self, enc, iv: Interval, sharp: float):
        from calibrated_response.maxent_sampler import (soft_between, soft_gt,
                                                        soft_lt)
        site, d = enc.site[iv.var], enc.dims[enc.site[iv.var]]
        lo, hi = iv.lo / d, iv.hi / d
        if iv.lo <= 0:
            return soft_lt(site, hi, sharp)
        if iv.hi >= d:
            return soft_gt(site, lo, sharp)
        return soft_between(site, lo, hi, sharp)

    def _event(self, enc, ivs):
        fs = [self._indicator(enc, iv, self.sharpness) for iv in ivs]

        def conj(x, fs=fs):
            out = fs[0](x)
            for f in fs[1:]:
                out = out * f(x)
            return out
        return conj

    def _convert(self, enc, bag, seed: int):
        from calibrated_response.maxent_sampler import moment, soft_gt
        w_p = 1.0 / (2.0 * self.prob_sd ** 2)
        w_e = 1.0 / (2.0 * self.expect_sd ** 2)
        rng = np.random.default_rng(seed + 71)
        csts = []
        for c in bag:
            if isinstance(c, MarginalConstraint):
                site, d = enc.site[c.var], enc.dims[enc.site[c.var]]
                p = np.asarray(c.probs, np.float64)
                p = p / p.sum()
                if self.marginal_mode == "mmd":
                    b = rng.choice(d, size=self.mmd_ref_n, p=p)
                    ref = (b + rng.random(self.mmd_ref_n)) / d
                    csts.append(("mmd", site, ref, self.mmd_weight))
                else:                       # "cdf": survival at interior edges
                    tail = 1.0 - np.cumsum(p)
                    for k in range(d - 1):
                        csts.append(("expect",
                                     soft_gt(site, (k + 1) / d,
                                             self.marginal_sharpness),
                                     float(tail[k]), w_p))
            elif isinstance(c, ProbConstraint):
                csts.append(("expect", self._event(enc, c.events),
                             c.target, w_p))
            elif isinstance(c, CondProbConstraint):
                csts.append(("cond_expect", self._event(enc, c.event),
                             self._event(enc, c.given), c.target, w_p))
            elif isinstance(c, CondExpectConstraint):
                csts.append(("cond_expect", moment(enc.site[c.var]),
                             self._event(enc, c.given), c.target, w_e))
            else:
                raise TypeError(f"unknown constraint {type(c).__name__}")
        return csts

    def fit(self, enc: TableEncoder, bag, seed: int = 0):
        from calibrated_response.maxent_sampler import FlowSamplerModel
        model = FlowSamplerModel(enc.tn_vars(), n_layers=self.n_layers,
                                 hidden=self.hidden, s_max=self.s_max)
        loss = model.constraint_loss(self._convert(enc, bag, seed),
                                     entropy_reg=self.entropy_reg,
                                     n_samples=self.n_samples)
        params, history = model.optimize(loss, backend="adam", seed=seed,
                                         steps=self.steps, lr=self.lr)
        return _FlowFit(enc, model, params, history,
                        n_mc=self.n_mc_score, nll_mc=self.nll_mc, seed=seed)


class _FlowFit:
    def __init__(self, enc, model, params, history, n_mc, nll_mc, seed):
        self.enc, self.model = enc, model
        self.params, self.history = params, history
        self.nll_mc, self._seed = nll_mc, seed
        dims = np.asarray(enc.dims)
        X = model.sample(params, n_mc, seed=seed + 1)         # (N, D) in [0,1]
        self.Xi = np.minimum((X * dims).astype(np.int64), dims - 1)

    def _logp(self, pts):
        return self.model.log_prob(self.params, pts)

    def log_prob_rows(self, Xi):
        """log bin mass per row: exact density MC-averaged over ``nll_mc``
        uniform points inside the row's bin box, times the bin volume."""
        dims = np.asarray(self.enc.dims, np.float64)
        rng = np.random.default_rng(self._seed + 7)
        K, (N, D) = self.nll_mc, Xi.shape
        pts = (Xi[None] + rng.random((K, N, D))) / dims       # (K, N, D)
        lp = np.asarray(self._logp(pts.reshape(K * N, D)))
        lp = lp.reshape(K, N)
        m = lp.max(axis=0)
        log_mean_density = m + np.log(np.mean(np.exp(lp - m), axis=0))
        return log_mean_density - np.log(dims).sum()

    def marginal(self, var):
        i = self.enc.site[var]
        c = np.bincount(self.Xi[:, i], minlength=self.enc.dims[i]).astype(float)
        return c / c.sum()

    def pair_marginal(self, a, b):
        i, j = self.enc.site[a], self.enc.site[b]
        h = np.zeros((self.enc.dims[i], self.enc.dims[j]))
        np.add.at(h, (self.Xi[:, i], self.Xi[:, j]), 1.0)
        return h / h.sum()


# ======================================================================
# Gaussian baseline (single joint N(mu, Sigma), closed-form fit losses)
# ======================================================================

class GaussianEngine:
    """Single joint ``N(mu, Sigma)`` on the encoded ``[0,1]^D`` box
    (:class:`calibrated_response.pc.GaussianModel`), fit by the pc module's
    closed-form CDF / bivariate-CDF losses — **no Monte Carlo in the fit**.
    Every bag constraint is a <=2-variable box event, which the model scores
    exactly (Phi / bivariate Phi); marginal histograms are pinned via their
    CDF at every interior bin edge (the Gaussian LSQ-fits them with mu, sigma).

    The strong-linear baseline: one global correlation matrix, exact queries,
    seconds to fit. Structurally unable to represent multimodality or
    nonlinear/heteroscedastic dependence, and its density leaks outside the
    box (that leak is charged to it in ``heldout_nll``). There is no entropy
    term — directions the bag leaves free stay near initialization, and
    unconstrained correlations stay ~0 (the same independent default maxent
    gives).

    Scoring mirrors ``_FlowFit``: marginals/pairs histogram one big (clipped)
    sample; ``heldout_nll`` MC-averages the exact density over each row's bin.
    """

    def __init__(self, steps: int = 2000, lr: float = 0.03,
                 prob_sd: float = 0.05, expect_sd: float = 0.05,
                 n_mc_score: int = 400_000, nll_mc: int = 8):
        self.name = "gaussian"
        self.steps, self.lr = steps, lr
        self.prob_sd, self.expect_sd = prob_sd, expect_sd
        self.n_mc_score, self.nll_mc = n_mc_score, nll_mc

    # ---- bag -> vectorized bound arrays --------------------------------
    # Open sides use a finite sentinel, NOT inf: an inf bound survives the
    # forward pass (clip), but its VJP evaluates 0 * inf = NaN through the
    # shared sd — the same reason gaussian_baseline.py uses _LARGE sentinels.
    _OPEN = 100.0                       # encoded domain is [0, 1]

    @classmethod
    def _bounds(cls, enc, iv: Interval):
        """(site, lo, hi) in encoded units; open sides as -+_OPEN."""
        d = enc.dims[enc.site[iv.var]]
        return (enc.site[iv.var],
                -cls._OPEN if iv.lo <= 0 else iv.lo / d,
                cls._OPEN if iv.hi >= d else iv.hi / d)

    @staticmethod
    def _ev(enc, ivs):
        """GaussianModel event dict (only used for cond-expectation terms)."""
        ev = {}
        for iv in ivs:
            d = enc.dims[enc.site[iv.var]]
            ev[iv.var] = ("interval",
                          None if iv.lo <= 0 else iv.lo / d,
                          None if iv.hi >= d else iv.hi / d)
        return ev

    def fit(self, enc: TableEncoder, bag, seed: int = 0):
        """One vectorized loss: all marginal-CDF edges in a single Phi
        expression, all 2-variable boxes through one vmapped bivariate CDF.
        (Building each constraint as its own ``losses.match_prob`` term makes
        XLA unroll ~100 quadrature/covariance subgraphs — a >10 min compile
        for what is otherwise a seconds-long fit.)"""
        import jax
        import jax.numpy as jnp
        from calibrated_response.pc import (GaussianModel, TrainConfig,
                                            VarSpec, train)
        from calibrated_response.pc.gaussian_baseline import _Phi, _bvn_std

        model = GaussianModel([VarSpec(s.name, "gaussian", 0.0, 1.0)
                               for s in enc.specs])
        w_p = 1.0 / (2.0 * self.prob_sd ** 2)
        w_e = 1.0 / (2.0 * self.expect_sd ** 2)

        # marginal CDF pins: P(x_site < edge) = target
        m_site, m_edge, m_tgt = [], [], []
        # 2-var boxes (i, ai, bi, j, aj, bj): unconditional first, then the
        # numerators of the conditionals (whose 1-var denominators are on j)
        box, box_tgt, n_uncond = [], [], 0
        ce_terms = []
        for c in bag:
            if isinstance(c, MarginalConstraint):
                d = enc.dims[enc.site[c.var]]
                p = np.asarray(c.probs, np.float64)
                cdf = np.cumsum(p / p.sum())
                for k in range(d - 1):
                    m_site.append(enc.site[c.var])
                    m_edge.append((k + 1) / d)
                    m_tgt.append(float(cdf[k]))
            elif isinstance(c, ProbConstraint):
                (i, ai, bi), (j, aj, bj) = (self._bounds(enc, iv)
                                            for iv in c.events)
                box.append((i, ai, bi, j, aj, bj))
                box_tgt.append(c.target)
            elif isinstance(c, CondExpectConstraint):
                ce_terms.append((c.var, self._ev(enc, c.given), c.target))
        n_uncond = len(box)
        cond_tgt = []
        for c in bag:
            if isinstance(c, CondProbConstraint):
                (i, ai, bi), = (self._bounds(enc, iv) for iv in c.event)
                (j, aj, bj), = (self._bounds(enc, iv) for iv in c.given)
                box.append((i, ai, bi, j, aj, bj))
                cond_tgt.append(c.target)

        MS = jnp.asarray(m_site, jnp.int32)
        ME = jnp.asarray(m_edge, jnp.float32)
        MT = jnp.asarray(m_tgt, jnp.float32)
        B = np.asarray(box, np.float32).reshape(-1, 6)
        BI, BJ = jnp.asarray(B[:, 0], jnp.int32), jnp.asarray(B[:, 3], jnp.int32)
        BAI, BBI = jnp.asarray(B[:, 1]), jnp.asarray(B[:, 2])
        BAJ, BBJ = jnp.asarray(B[:, 4]), jnp.asarray(B[:, 5])
        PT = jnp.asarray(box_tgt, jnp.float32)
        CT = jnp.asarray(cond_tgt, jnp.float32)
        bvn = jax.vmap(_bvn_std)

        def loss_fn(p):
            mu = p["mean"]
            L = GaussianModel._L(p)
            cov = L @ L.T
            sd = jnp.sqrt(jnp.diagonal(cov))
            std = lambda b, s: jnp.clip((b - mu[s]) / sd[s], -8.0, 8.0)
            tot = w_p * jnp.sum((_Phi(std(ME, MS)) - MT) ** 2)
            if len(B):
                rho = cov[BI, BJ] / (sd[BI] * sd[BJ])
                xa, xb = std(BAI, BI), std(BBI, BI)
                ya, yb = std(BAJ, BJ), std(BBJ, BJ)
                p2 = (bvn(xb, yb, rho) - bvn(xa, yb, rho)
                      - bvn(xb, ya, rho) + bvn(xa, ya, rho))
                tot = tot + w_p * jnp.sum((p2[:n_uncond] - PT) ** 2)
                den = _Phi(yb[n_uncond:]) - _Phi(ya[n_uncond:])
                tot = tot + w_p * jnp.sum(
                    (p2[n_uncond:] / (den + 1e-6) - CT) ** 2)
            for a, given, target in ce_terms:
                tot = tot + w_e * (model.expectation(p, a, event=given)
                                   - target) ** 2
            return tot

        params, history = train(loss_fn, model.init_params(seed=seed + 1),
                                TrainConfig(steps=self.steps, lr=self.lr))
        return _GaussianFit(enc, model, params, history,
                            n_mc=self.n_mc_score, nll_mc=self.nll_mc, seed=seed)


class _GaussianFit(_FlowFit):
    def _logp(self, pts):
        return self.model.marginal_log_prob(self.params, pts)


# ======================================================================
# Gaussian copula: exact histogram marginals + Gaussian dependence
# ======================================================================

class GaussianCopulaEngine:
    """Gaussian *copula* on the encoded grid: each variable keeps its (noisy)
    bag marginal **exactly** — piecewise-uniform over the bins — and all
    dependence lives in a single correlation matrix ``R`` in latent z-space.

    This is the fair "a Gaussian is probably strong" baseline for this
    benchmark: a plain joint Gaussian is misspecified on quantile-encoded
    data (uniform marginals, bounded box), which costs it more nats than its
    correlation structure earns back. The copula removes exactly that
    handicap while keeping the linear-dependence hypothesis.

    Fitting is fully closed-form and tiny: because marginals are fixed, every
    bag constraint reduces to 2-variable z-space boxes with **constant**
    bounds (bin edge -> z via the marginal CDF), so only ``R`` is learned:

    - ``ProbConstraint``      -> (box2(R) - target)^2
    - ``CondProbConstraint``  -> (box2(R)/P(given) - target)^2, ``P(given)``
      a known constant (it is a marginal probability).
    - ``CondExpectConstraint``-> E[x_a | given] = sum_k center_k *
      box2(bin_k, given; R) / P(given) — one batched sum of bin boxes.

    Unconstrained correlations stay at their ~0 init (independent default).
    Scoring mirrors the other engines: samples for marginals/pairs, exact
    copula density MC-averaged over each row's bin for ``heldout_nll``.
    """

    _OPEN = GaussianEngine._OPEN

    def __init__(self, steps: int = 1500, lr: float = 0.05,
                 prob_sd: float = 0.05, expect_sd: float = 0.05,
                 n_mc_score: int = 400_000, nll_mc: int = 8):
        self.name = "gaussian_copula"
        self.steps, self.lr = steps, lr
        self.prob_sd, self.expect_sd = prob_sd, expect_sd
        self.n_mc_score, self.nll_mc = n_mc_score, nll_mc

    def fit(self, enc: TableEncoder, bag, seed: int = 0):
        import jax
        import jax.numpy as jnp
        from scipy.special import ndtri
        from calibrated_response.pc import TrainConfig, train
        from calibrated_response.pc.gaussian_baseline import _bvn_std

        # ---- fixed per-variable marginals -> z-space bin-edge tables -----
        probs = {v: np.full(d, 1.0 / d) for v, d in zip(enc.names, enc.dims)}
        for c in bag:
            if isinstance(c, MarginalConstraint):
                p = np.clip(np.asarray(c.probs, np.float64), 1e-6, None)
                probs[c.var] = p / p.sum()
        z_edges = {}
        for v in enc.names:
            cdf = np.concatenate([[0.0], np.cumsum(probs[v])])
            cdf[-1] = 1.0
            z = ndtri(np.clip(cdf, 1e-9, 1.0 - 1e-9))
            z[0], z[-1] = -self._OPEN, self._OPEN
            z_edges[v] = z

        def zb(iv: Interval):
            """(site, z_lo, z_hi) — constant z-space bounds of an interval."""
            return (enc.site[iv.var], float(z_edges[iv.var][iv.lo]),
                    float(z_edges[iv.var][iv.hi]))

        def phi(t):
            from scipy.special import ndtr
            return float(ndtr(np.clip(t, -self._OPEN, self._OPEN)))

        # ---- gather one batch of 2-var z-boxes ---------------------------
        # rows: (i, zai, zbi, j, zaj, zbj); groups tagged by (kind, ...)
        rows, prob_tgt, cond = [], [], []
        ce_seg, ce_coeff, ce_tgt = [], [], []
        for c in bag:
            if isinstance(c, ProbConstraint):
                (i, ai, bi), (j, aj, bj) = (zb(iv) for iv in c.events)
                rows.append((i, ai, bi, j, aj, bj))
                prob_tgt.append(c.target)
        n_prob = len(rows)
        for c in bag:
            if isinstance(c, CondProbConstraint):
                (i, ai, bi), = (zb(iv) for iv in c.event)
                (j, aj, bj), = (zb(iv) for iv in c.given)
                rows.append((i, ai, bi, j, aj, bj))
                cond.append((c.target, phi(bj) - phi(aj)))
        n_cond = len(rows) - n_prob
        n_ce = 0
        for c in bag:
            if isinstance(c, CondExpectConstraint):
                a = enc.site[c.var]
                d = enc.dims[a]
                (j, aj, bj), = (zb(iv) for iv in c.given)
                den = phi(bj) - phi(aj)
                ez = z_edges[c.var]
                for k in range(d):
                    rows.append((a, float(ez[k]), float(ez[k + 1]), j, aj, bj))
                    ce_seg.append(n_ce)
                    ce_coeff.append(((k + 0.5) / d) / max(den, 1e-6))
                ce_tgt.append(c.target)
                n_ce += 1

        n = len(enc.names)
        if not rows:                        # marginals only: R = I
            return _CopulaFit(enc, _Copula(np.eye(n),
                                           [z_edges, probs, enc.names]),
                              None, [0.0],
                              n_mc=self.n_mc_score, nll_mc=self.nll_mc,
                              seed=seed)

        B = np.asarray(rows, np.float32)
        BI, BJ = jnp.asarray(B[:, 0], jnp.int32), jnp.asarray(B[:, 3], jnp.int32)
        XA = jnp.clip(jnp.asarray(B[:, 1]), -8.0, 8.0)
        XB = jnp.clip(jnp.asarray(B[:, 2]), -8.0, 8.0)
        YA = jnp.clip(jnp.asarray(B[:, 4]), -8.0, 8.0)
        YB = jnp.clip(jnp.asarray(B[:, 5]), -8.0, 8.0)
        PT = jnp.asarray(prob_tgt, jnp.float32)
        CT = jnp.asarray([t for t, _ in cond], jnp.float32)
        CD = jnp.asarray([d_ for _, d_ in cond], jnp.float32)
        SEG = jnp.asarray(ce_seg, jnp.int32)
        CO = jnp.asarray(ce_coeff, jnp.float32)
        ET = jnp.asarray(ce_tgt, jnp.float32)
        w_p = 1.0 / (2.0 * self.prob_sd ** 2)
        w_e = 1.0 / (2.0 * self.expect_sd ** 2)
        bvn = jax.vmap(_bvn_std)

        def corr(p):
            Lr = jnp.tril(p["L_raw"])
            Lu = Lr / jnp.sqrt(jnp.sum(Lr * Lr, axis=1, keepdims=True) + 1e-12)
            return Lu @ Lu.T                # unit diagonal by construction

        def loss_fn(p):
            rho = corr(p)[BI, BJ]
            p2 = (bvn(XB, YB, rho) - bvn(XA, YB, rho)
                  - bvn(XB, YA, rho) + bvn(XA, YA, rho))
            tot = w_p * jnp.sum((p2[:n_prob] - PT) ** 2)
            tot = tot + w_p * jnp.sum(
                (p2[n_prob:n_prob + n_cond] / (CD + 1e-6) - CT) ** 2)
            if n_ce:
                e = jax.ops.segment_sum(CO * p2[n_prob + n_cond:], SEG,
                                        num_segments=n_ce)
                tot = tot + w_e * jnp.sum((e - ET) ** 2)
            return tot

        rng = np.random.default_rng(seed + 1)
        p0 = {"L_raw": jnp.asarray(
            np.eye(n) + 1e-2 * rng.standard_normal((n, n)), jnp.float32)}
        params, history = train(loss_fn, p0,
                                TrainConfig(steps=self.steps, lr=self.lr))
        R = np.asarray(corr(params), np.float64)
        return _CopulaFit(enc, _Copula(R, [z_edges, probs, enc.names]),
                          params, history,
                          n_mc=self.n_mc_score, nll_mc=self.nll_mc, seed=seed)


class _Copula:
    """Frozen Gaussian copula over the encoded box: exact piecewise-uniform
    marginals, correlation ``R`` in z-space. Exposes the same ``sample`` /
    batched log-density surface the flow fit consumes."""

    def __init__(self, R, tables):
        z_edges, probs, names = tables
        self.R = np.asarray(R, np.float64)
        self.L = np.linalg.cholesky(self.R + 1e-9 * np.eye(len(self.R)))
        self.cdf = [np.concatenate([[0.0], np.cumsum(probs[v])])
                    for v in names]
        for c in self.cdf:
            c[-1] = 1.0
        self.p = [np.diff(c) for c in self.cdf]

    def sample(self, params, n, seed=0):
        from scipy.special import ndtr
        rng = np.random.default_rng(seed)
        u = ndtr(rng.standard_normal((n, len(self.R))) @ self.L.T)
        cols = []
        for i, c in enumerate(self.cdf):
            d = len(c) - 1
            k = np.clip(np.searchsorted(c, u[:, i], side="right") - 1, 0, d - 1)
            frac = np.clip((u[:, i] - c[k]) / np.maximum(self.p[i][k], 1e-12),
                           0.0, 1.0)
            cols.append((k + frac) / d)
        return np.stack(cols, axis=1)

    def log_prob(self, params, x):
        """Exact copula density at encoded points ``x`` (N, D):
        ``log c(u(x); R) + sum log f_i(x_i)`` with piecewise-constant
        marginal densities ``f_i = d * p_ik``."""
        from scipy.linalg import solve_triangular
        from scipy.special import ndtri
        x = np.atleast_2d(np.asarray(x, np.float64))
        zs, logm = [], 0.0
        for i, c in enumerate(self.cdf):
            d = len(c) - 1
            k = np.clip((x[:, i] * d).astype(np.int64), 0, d - 1)
            pk = np.maximum(self.p[i][k], 1e-12)
            u = np.clip(c[k] + (x[:, i] * d - k) * pk, 1e-9, 1.0 - 1e-9)
            zs.append(ndtri(u))
            logm = logm + np.log(pk * d)
        z = np.stack(zs, axis=1)
        w = solve_triangular(self.L, z.T, lower=True).T
        logdet = 2.0 * np.log(np.diag(self.L)).sum()
        log_c = -0.5 * ((w * w).sum(1) - (z * z).sum(1)) - 0.5 * logdet
        return log_c + logm


class _CopulaFit(_FlowFit):
    pass                                   # _Copula already matches the surface


# ======================================================================
# TODO stubs — wire in the existing implementations for the bake-off.
# ======================================================================


class PCEngine:
    """Probabilistic-circuit adapter over calibrated_response.pc.Circuit with
    losses from calibrated_response.pc.losses (match_prob / match_cond_prob /
    match_marginal / match_expectation map 1:1 onto the bag)."""
    name = "pc"

    def fit(self, enc, bag, seed: int = 0):
        raise NotImplementedError


ENGINES = {
    "independent": IndependentEngine,
    "tn": TensorChainEngine,
    "flow": FlowEngine,
    "gaussian": GaussianEngine,
    "copula": GaussianCopulaEngine,
    "pc": PCEngine,
}
