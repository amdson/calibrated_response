"""Implicit-sampler density model with **sample-native** losses.

``SamplerModel`` represents ``p(x)`` implicitly as a reparametrised sampler::

    x = lower + span * sigmoid(MLP_θ(z)),   z ~ N(0, I)

and fits it by matching *sample expectations* of continuous feature functions —
no bins, no marginal tables.  Every constraint is an expectation ``E[f(x)]`` (or
a ratio of expectations for conditionals, or an MMD for whole-distribution
matching), estimated by Monte-Carlo over a batch of latents and differentiated in
``θ`` by the reparametrisation trick.

Why sample-native (vs. the tensor-network bin grammar).  A scalar expectation
``E[g(x)] ≈ (1/N) Σ g(xᵢ)`` has variance ``Var[g]/N`` *independent of how many
variables g touches* — so a 3-variable interaction constraint is as cheap and as
accurate as a 1-variable mean.  The only quantity that needs more than a few
scalars is "match this entire marginal", for which we use **MMD** (a kernel
two-sample distance to reference samples) rather than a KL over a histogram.

Constraint grammar (``f``/``cond`` are callables ``x:(N,n) -> (N,)`` on the
continuous batch; use the smooth factories below for indicators)::

    ("expect",      f, target[, weight])          #  E[f(x)]            = target
    ("cond_expect", f, cond, target[, weight])    #  E[f·cond]/E[cond]  = target
    ("expect_nll",      f, target, k[, tau[, beta]])        # t ~ N(E[f], Var[f]/k + tau²)
    ("cond_expect_nll", f, cond, target, k[, tau[, beta]])  # conditional variant
    ("prob_nll",        f, target, k[, beta])               # binomial: k·KL(t ‖ E[f])
    ("cond_prob_nll",   f, cond, target, k[, beta])         # conditional variant
    ("cov",         f, g, target[, weight])       #  Cov(f, g)          = target
    ("corr",        f, g, target[, weight])       #  Corr(f, g)         = target
    ("mmd",         sites, ref_samples[, weight]) #  MMD(x[:,sites], ref) -> 0
    ("eqn_det",     r, weight)                     #  residual r(x)      = 0
    ("eqn_dist",    r, sigma, weight)              #  residual r(x)      ~ N(0, sigma)
    ("eqn_nll",     r, sigma, k[, beta])           #  k·KL(N(0,σ²) ‖ N(μ_r, v_r)), one-sided in v

One-sided ("hinge") variants penalise only the violating side of the bound
(``direction`` is ``"le"`` for ``<= target`` or ``"ge"`` for ``>= target``)::

    ("expect_ineq",            f, target, weight, direction)
    ("cond_expect_ineq",       f, cond, target, weight, direction)
    ("logit_expect_ineq",      f, target, weight, direction)   # log-odds hinge
    ("logit_cond_expect_ineq", f, cond, target, weight, direction)
    ("corr_ineq",              f, g, target, weight, direction)
    ("eqn_ineq",               r, weight, direction)  # residual r(x) >= 0 / <= 0

``eqn_ineq`` is the per-sample member of that family: it hinges the residual
*inside* the expectation (``E[relu(-r)**2]``), so it restricts the support to
one side of a surface, where the others hinge an aggregate statistic.

Head-to-head with :mod:`calibrated_response.tn`: build both models from the same
:class:`~calibrated_response.tn.discretize.ContinuousVar` list and compare on the
*query outputs* (``expectation`` / ``prob_gt`` / ``site_marginal`` overlays), each
model expressing the constraint in its own natural form.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from calibrated_response.tn.discretize import ContinuousVar, Discretizer
from calibrated_response.maxent_sampler.fit import fit_adam_stochastic
from calibrated_response.maxent_sampler.sampler_model import (
    NeuralSampler, neg_gaussian_entropy_proxy, neg_marginal_entropy_proxy,
    neg_pairwise_entropy_proxy,
)

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Feature factories — smooth, continuous, sample-native (no bins)
# ---------------------------------------------------------------------------
#
# Each returns a callable ``f(x) -> (N,)`` over the continuous sample batch
# ``x`` of shape ``(N, n)``.  Use the soft (sigmoid) variants inside a loss so
# gradients flow; the hard variants are for evaluation queries.

def moment(site: int, order: int = 1) -> Callable:
    """``x[:, site] ** order`` — mean (order=1) or higher-moment feature."""
    return lambda x: x[:, site] ** order


def product(site_a: int, site_b: int) -> Callable:
    """``x[:, a] * x[:, b]`` — a pairwise interaction / (uncentred) covariance term."""
    return lambda x: x[:, site_a] * x[:, site_b]


def soft_gt(site: int, threshold: float, sharpness: float = 50.0) -> Callable:
    """Smooth ``P(x_site > threshold)`` surrogate: ``sigmoid(k·(x - t))``."""
    return lambda x: jax.nn.sigmoid(sharpness * (x[:, site] - threshold))


def soft_lt(site: int, threshold: float, sharpness: float = 50.0) -> Callable:
    """Smooth ``P(x_site < threshold)`` surrogate: ``sigmoid(k·(t - x))``."""
    return lambda x: jax.nn.sigmoid(sharpness * (threshold - x[:, site]))


def soft_between(site: int, lower: float, upper: float,
                 sharpness: float = 50.0) -> Callable:
    """Smooth ``P(lower < x_site < upper)`` surrogate (product of two sigmoids)."""
    return lambda x: (jax.nn.sigmoid(sharpness * (x[:, site] - lower))
                      * jax.nn.sigmoid(sharpness * (upper - x[:, site])))


def hard_gt(site: int, threshold: float) -> Callable:
    """Exact indicator ``1[x_site > threshold]`` — for evaluation, not the loss."""
    return lambda x: (x[:, site] > threshold).astype(jnp.float32)


def straight_through(soft_f: Callable, hard_f: Callable) -> Callable:
    """Forward = ``hard_f`` exactly, backward = ``soft_f``'s gradient.

    Fixes the soft-indicator *leakage floor*: a sigmoid at sharpness ``s``
    reads ~``density·ln2/s`` of spurious event mass per side, which for a
    near-impossibility target (elicited ``P < eps``) can exceed the target
    itself and flip the optimum to a vacuous joint (see solver_additions.md).
    The ST form measures the event exactly at ANY backward sharpness, so the
    gradient keeps a wide support band while the objective stays unbiased.
    In *products* of ST features each factor's gradient is gated by the other
    factors' hard values — e.g. samples inside an implication's satisfied
    region feel zero violation gradient.  Validated against sharp-soft and
    pathwise+score alternatives on the prereq benchmark (ST won)."""
    def f(x):
        s = soft_f(x)
        return s + jax.lax.stop_gradient(hard_f(x) - s)
    return f


def _logit(p):
    """Numerically-safe log-odds (clips into [1e-4, 1 - 1e-4])."""
    p = jnp.clip(p, 1e-4, 1.0 - 1e-4)
    return jnp.log(p) - jnp.log1p(-p)


def _hinge_pen(resid, direction: str):
    """One-sided squared penalty on a signed residual ``resid = fitted - target``.

    ``direction="le"`` encodes ``fitted <= target``: only *exceeding* the bound
    (``resid > 0``) is penalised, so the constraint is a soft ceiling and the
    entropy term is free to push the quantity anywhere below it.  ``"ge"`` is the
    mirror soft floor (``fitted >= target``, penalising ``resid < 0``).  ``"eq"``
    would be the ordinary two-sided ``resid**2`` — callers use the plain kinds for
    that, so it is not accepted here.
    """
    if direction == "le":
        return jax.nn.relu(resid) ** 2
    if direction == "ge":
        return jax.nn.relu(-resid) ** 2
    raise ValueError(f"inequality direction must be 'le' or 'ge', got {direction!r}")


class SamplerModel:
    """Implicit-sampler density model fit by sample-native expectation losses.

    Parameters
    ----------
    vars : sequence of ContinuousVar
        Variable specs (shared with the tensor-network models).  Only the
        ``lower``/``upper`` domains are used for fitting; ``n_bins`` is used only
        by :meth:`site_marginal` for histogram overlays against ``tn``.
    latent_dim : int
        Dimensionality of the Gaussian latent ``z`` (the sampler's capacity knob,
        analogous to the tensor network's ``bond_dim``).
    hidden_sizes : int or list of int
        MLP hidden widths of the sampler network.
    """

    def __init__(self, vars: Sequence[ContinuousVar], latent_dim: int = 8,
                 hidden_sizes=64):
        self.disc = Discretizer(vars)
        self.n = self.disc.n_sites
        self.dims = self.disc.dims
        self.latent_dim = int(latent_dim)

        self.net = NeuralSampler(self.n, self.latent_dim, hidden_sizes)
        self.lower = jnp.asarray(self.disc.lower, jnp.float32)
        self.span = jnp.asarray(self.disc.upper - self.disc.lower, jnp.float32)

        # Per-``onoff``-constraint prior break probabilities, filled in by the most
        # recent constraint_loss() call so init_params() can size the credence gates.
        self._gate_pbroken: list = []

    # ---- parameters -------------------------------------------------
    def init_params(self, seed: int = 0):
        """Initialise sampler parameters.

        Returns ``{"theta": (n_params,)}``, plus ``"gates": (n_onoff,)`` credence
        logits when the last :meth:`constraint_loss` had ``onoff`` constraints
        (each gate initialised to its prior ``logit(1 - p_broken)``).  In the usual
        ``optimize(constraint_loss(...))`` call the loss is built first, so the gate
        count is known by the time ``optimize`` initialises params.
        """
        key = jax.random.PRNGKey(seed)
        params = {"theta": self.net.pack_params(self.net.init_params(key))}
        if self._gate_pbroken:
            logits = [float(np.log((1.0 - pb) / pb)) for pb in self._gate_pbroken]
            params["gates"] = jnp.asarray(logits, jnp.float32)
        return params

    def credences(self, params):
        """Posterior ``P(constraint active)`` for each ``onoff`` gate, in order.

        A low value means the constraint was *convicted* (explained away as broken)
        during the fit — the sampler's analogue of ``joint_marginal(c_site)[1]``."""
        if "gates" not in params:
            return np.array([])
        return np.asarray(jax.nn.sigmoid(params["gates"]))

    # ---- sampling ---------------------------------------------------
    def _sample_x(self, params, z):
        """Latents ``z`` (N, latent_dim) -> states ``x`` (N, n) in variable domains."""
        u = jax.vmap(self.net.sample_fn_flat, in_axes=(None, 0))(params["theta"], z)
        return self.lower + self.span * u

    def _draw_z(self, n_samples: int, seed: int):
        return jax.random.normal(jax.random.PRNGKey(seed), (n_samples, self.latent_dim))

    def sample(self, params, n_samples: int, seed: int = 0):
        """Draw ``n_samples`` continuous samples, shape ``(n_samples, n)`` (numpy)."""
        return np.asarray(self._sample_x(params, self._draw_z(n_samples, seed)))

    # ==================================================================
    # MMD (kernel two-sample) — the bin-free "match this marginal" loss
    # ==================================================================
    @staticmethod
    def _sqdist(A, B):
        aa = jnp.sum(A * A, axis=1)[:, None]
        bb = jnp.sum(B * B, axis=1)[None, :]
        return jnp.clip(aa + bb - 2.0 * (A @ B.T), a_min=0.0)

    @staticmethod
    def _median_bandwidths(ref: np.ndarray) -> list:
        """Median-heuristic RBF bandwidths (multi-scale) from reference samples."""
        R = np.atleast_2d(np.asarray(ref, np.float64))
        m = min(len(R), 1000)                       # subsample for the heuristic
        R = R[np.linspace(0, len(R) - 1, m).astype(int)]
        d2 = np.maximum(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1), 0.0)
        med = np.sqrt(np.median(d2[d2 > 0]) / 2.0) if np.any(d2 > 0) else 1.0
        return [float(med) * s for s in (0.5, 1.0, 2.0)]

    def _mmd2(self, X, Y, bandwidths):
        """Biased RBF-kernel MMD² between sample sets ``X`` and reference ``Y``."""
        def kern(d2):
            return sum(jnp.exp(-d2 / (2.0 * h * h)) for h in bandwidths) / len(bandwidths)
        Kxx = kern(self._sqdist(X, X)).mean()
        Kxy = kern(self._sqdist(X, Y)).mean()
        Kyy = kern(self._sqdist(Y, Y)).mean()
        return Kxx - 2.0 * Kxy + Kyy

    # ==================================================================
    # constraint loss
    # ==================================================================
    def _prepare(self, cst):
        """Turn one constraint tuple into a scoring closure ``score(x) -> scalar``."""
        kind = cst[0]
        if kind == "expect":
            f, tg = cst[1], cst[2]
            w = cst[3] if len(cst) > 3 else 1.0
            return lambda x: w * (jnp.mean(f(x)) - tg) ** 2
        if kind == "expect_ineq":
            # One-sided E[f(x)] <= / >= target (see _hinge_pen).
            f, tg, w, direction = cst[1], cst[2], cst[3], cst[4]
            return lambda x: w * _hinge_pen(jnp.mean(f(x)) - tg, direction)
        if kind == "cond_expect":
            f, cond, tg = cst[1], cst[2], cst[3]
            w = cst[4] if len(cst) > 4 else 1.0
            def score(x):
                c = cond(x)
                num = jnp.mean(f(x) * c)
                den = jnp.mean(c) + _EPS
                return w * (num / den - tg) ** 2
            return score
        if kind == "cond_expect_ineq":
            f, cond, tg, w, direction = cst[1], cst[2], cst[3], cst[4], cst[5]
            def score(x):
                c = cond(x)
                num = jnp.mean(f(x) * c)
                den = jnp.mean(c) + _EPS
                return w * _hinge_pen(num / den - tg, direction)
            return score
        if kind == "logit_expect":
            # Probability belief penalised in LOG-ODDS: an absolute-scale
            # penalty gives a tail target (p = 0.02, sd = 0.05) several-x
            # odds of nearly-free slack, which the entropy term spends
            # inflating rare events toward 0.5. Log-odds slack is
            # multiplicative, uniform at every probability level.
            f, tg = cst[1], cst[2]
            w = cst[3] if len(cst) > 3 else 1.0
            lt = _logit(tg)
            return lambda x: w * (_logit(jnp.mean(f(x))) - lt) ** 2
        if kind == "logit_expect_ineq":
            # One-sided probability bound penalised in log-odds. "le" keeps the
            # hinge in the same monotone space as the equality logit penalty, so
            # the ceiling stays uniform in odds across the scale.
            f, tg, w, direction = cst[1], cst[2], cst[3], cst[4]
            lt = _logit(tg)
            return lambda x: w * _hinge_pen(_logit(jnp.mean(f(x))) - lt, direction)
        if kind == "logit_cond_expect":
            f, cond, tg = cst[1], cst[2], cst[3]
            w = cst[4] if len(cst) > 4 else 1.0
            lt = _logit(tg)
            def score(x):
                c = cond(x)
                num = jnp.mean(f(x) * c)
                den = jnp.mean(c) + _EPS
                return w * (_logit(num / den) - lt) ** 2
            return score
        if kind == "logit_cond_expect_ineq":
            f, cond, tg, w, direction = cst[1], cst[2], cst[3], cst[4], cst[5]
            lt = _logit(tg)
            def score(x):
                c = cond(x)
                num = jnp.mean(f(x) * c)
                den = jnp.mean(c) + _EPS
                return w * _hinge_pen(_logit(num / den) - lt, direction)
            return score
        if kind == "expect_nll":
            # Synthetic-likelihood mean constraint (Wood 2010): the target is
            # modelled as the mean of k draws from the fitted distribution,
            #   target ~ N(E[f], Var[f]/k + tau²),
            # and this scores its negative log-likelihood.  Unlike "expect",
            # the model's own Var[f] carries gradient: agreeing targets buy a
            # ½log-var shrink, while targets the mean cannot reach are cheapest
            # to explain by inflating Var[f] — conflicting estimates become
            # width instead of a silently-averaged tight fit.  ``k`` is the
            # unitless strength ("effective observation count"); ``tau`` floors
            # the implied noise sd, bounding the 1/var mean-gradient and capping
            # how far a single conflict can inflate the fit; ``beta`` is the
            # β-NLL exponent (Seitzer et al. 2022) — the NLL is scaled by
            # sg(Var/k + tau²)^β, so β=1 restores a constant-weight mean
            # gradient (no variance death spiral) while every β keeps the
            # width-seeking gradient path through Var.
            f, tg, k = cst[1], cst[2], float(cst[3])
            tau = float(cst[4]) if len(cst) > 4 else 0.0
            beta = float(cst[5]) if len(cst) > 5 else 0.5
            def score(x):
                v = f(x)
                m, var = jnp.mean(v), jnp.var(v)
                s2 = var / k + tau * tau + _EPS
                # debias: E[(m - tg)²] = (mu - tg)² + Var/N over the batch
                r2 = (m - tg) ** 2 - var / v.shape[0]
                nll = 0.5 * (r2 / s2 + jnp.log(s2))
                return jax.lax.stop_gradient(s2) ** beta * nll
            return score
        if kind == "cond_expect_nll":
            # Conditional synthetic likelihood: tg ~ N(mu_cond, var_cond/k
            # + tau²).  The likelihood depends only on the *conditional*
            # moments — "worth k observations of the slice" — so P(cond)
            # never scales the strength.  Every pc factor below is Monte
            # Carlo estimator bookkeeping and carries live gradients: the
            # −var/n_eff debias then cancels (in expectation) the spurious
            # ∇[var/(N·pc)] hidden in ∇(m − tg)², same as expect_nll.  Mass
            # may shift in/out of the condition, but only via the membership
            # gradients of the slice moments, priced against entropy.
            f, cond, tg, k = cst[1], cst[2], cst[3], float(cst[4])
            tau = float(cst[5]) if len(cst) > 5 else 0.0
            beta = float(cst[6]) if len(cst) > 6 else 0.5
            def score(x):
                v, c = f(x), cond(x)
                N = v.shape[0]
                pc = jnp.mean(c)
                denom = jnp.maximum(pc, 1.0 / N)
                m = jnp.mean(v * c) / denom
                var = jnp.mean((v - m) ** 2 * c) / denom
                s2 = var / k + tau * tau + _EPS
                n_eff = jnp.maximum(N * pc, 1.0)
                r2 = (m - tg) ** 2 - var / n_eff
                nll = 0.5 * (r2 / s2 + jnp.log(s2))
                return jax.lax.stop_gradient(s2) ** beta * nll
            return score
        if kind == "prob_nll":
            # Binomial pseudo-count constraint: the target is the empirical
            # rate of k Bernoulli draws, scored as k·KL(target ‖ E[f]) (the
            # binomial NLL minus its value at p = target, so a matched
            # constraint scores 0).  Exact — no CLT, no variance denominator:
            # curvature k/(p(1-p)) is bounded by the clip, and conflicting
            # targets resolve toward the max-variance p between them, so the
            # Gaussian kinds' beta/tau machinery is unnecessary here.
            # Optional ``beta``: likelihood tempering on the same footing as
            # expect_nll/eqn_nll -- the score is scaled by sg(p(1-p)/k)^beta,
            # the variance of the k-sample binomial mean.  beta=0 (default)
            # is the plain k*KL; beta=1 cancels k entirely (unit-strength
            # residual, incapable of dominating the entropy term).  Annealing
            # beta 1 -> 0 during fitting is an AUTOMATIC per-constraint
            # warmup: each constraint's strength ramps over a range set by
            # its own k and variance (power-posterior / homotopy
            # continuation), so stiff near-impossibility constraints get a
            # long ramp and mild marginals barely move -- no per-term
            # schedule to hand-tune.
            # The score also accepts a traced runtime override ``beta_t``
            # (marked via ``score.takes_beta``): constraint_loss passes it
            # when built with a per-step ``beta_schedule``, superseding the
            # baked value.  ``s ** 0.0 == 1.0`` so a schedule ending at 0
            # lands exactly on the plain k*KL.
            f, tg, k = cst[1], cst[2], float(cst[3])
            beta = float(cst[4]) if len(cst) > 4 else 0.0
            t = float(np.clip(tg, 1e-4, 1.0 - 1e-4))
            def score(x, beta_t=None):
                b = beta if beta_t is None else beta_t
                p = jnp.clip(jnp.mean(f(x)), 1e-4, 1.0 - 1e-4)
                kl = k * (t * (np.log(t) - jnp.log(p))
                          + (1.0 - t) * (np.log1p(-t) - jnp.log1p(-p)))
                if beta_t is None and not beta:
                    return kl
                return jax.lax.stop_gradient(p * (1.0 - p) / k) ** b * kl
            score.takes_beta = True
            return score
        if kind == "cond_prob_nll":
            # Conditional binomial pseudo-counts; k_eff = k·E[cond] as above.
            # beta / traced beta_t temper exactly as in prob_nll.
            f, cond, tg, k = cst[1], cst[2], cst[3], float(cst[4])
            beta = float(cst[5]) if len(cst) > 5 else 0.0
            t = float(np.clip(tg, 1e-4, 1.0 - 1e-4))
            def score(x, beta_t=None):
                b = beta if beta_t is None else beta_t
                c = cond(x)
                pc_g = jax.lax.stop_gradient(jnp.mean(c))
                p = jnp.clip(jnp.mean(f(x) * c) / (jnp.mean(c) + _EPS),
                             1e-4, 1.0 - 1e-4)
                k_eff = jnp.maximum(k * pc_g, 1e-3)
                kl = k_eff * (t * (np.log(t) - jnp.log(p))
                              + (1.0 - t) * (np.log1p(-t) - jnp.log1p(-p)))
                if beta_t is None and not beta:
                    return kl
                return (jax.lax.stop_gradient(p * (1.0 - p) / k_eff) ** b
                        * kl)
            score.takes_beta = True
            return score
        if kind == "cov":
            # Centred second moment: targets the *dependence* directly, invariant
            # to mean shifts — unlike ("expect", product(i,j), t), which a maxent
            # fit will happily satisfy by moving means (cheaper in entropy).
            f, g, tg = cst[1], cst[2], cst[3]
            w = cst[4] if len(cst) > 4 else 1.0
            def score(x):
                a, b = f(x), g(x)
                n = a.shape[0]
                c = (jnp.mean(a * b) - jnp.mean(a) * jnp.mean(b)) * (n / (n - 1.0))
                return w * (c - tg) ** 2
            return score
        if kind == "corr":
            # Scale-free version of "cov": target the Pearson correlation.
            f, g, tg = cst[1], cst[2], cst[3]
            w = cst[4] if len(cst) > 4 else 1.0
            def score(x):
                a, b = f(x), g(x)
                c = jnp.mean(a * b) - jnp.mean(a) * jnp.mean(b)
                r = c / jnp.sqrt((jnp.var(a) + _EPS) * (jnp.var(b) + _EPS))
                return w * (r - tg) ** 2
            return score
        if kind == "corr_ineq":
            # One-sided Corr(f, g) <= / >= target.
            f, g, tg, w, direction = cst[1], cst[2], cst[3], cst[4], cst[5]
            def score(x):
                a, b = f(x), g(x)
                c = jnp.mean(a * b) - jnp.mean(a) * jnp.mean(b)
                r = c / jnp.sqrt((jnp.var(a) + _EPS) * (jnp.var(b) + _EPS))
                return w * _hinge_pen(r - tg, direction)
            return score
        if kind == "eqn_det":
            # Deterministic identity lhs = rhs: drive the residual to zero.
            # r(x) is a per-sample residual closure (see maxent_sampler.equations).
            g, w = cst[1], cst[2]
            return lambda x: w * jnp.mean(g(x) ** 2)
        if kind == "eqn_ineq":
            # Inequality lhs > rhs ("ge") / lhs < rhs ("le"): a one-sided
            # constraint on the same residual. Unlike eqn_det/eqn_dist this
            # restricts the *support* rather than matching moments, so maxent
            # spreads over the whole allowed region instead of hugging a line;
            # 1/sqrt(2w) is the width over which the boundary softens.
            g, w, direction = cst[1], cst[2], cst[3]
            return lambda x: w * jnp.mean(_hinge_pen(g(x), direction))
        if kind == "eqn_dist":
            # Noisy identity lhs = rhs + N(0, sigma): match the residual's first
            # two moments (mean 0, variance sigma**2). Maxent completes the pair
            # to a Gaussian residual ~independent of the rhs variables. Both
            # moments are low-variance sample means -> good signal at N~few-k.
            g, sigma, w = cst[1], cst[2], cst[3]
            s2 = sigma * sigma
            def score(x):
                r = g(x)
                return w * ((jnp.mean(r) / sigma) ** 2
                            + (jnp.mean(r * r) / s2 - 1.0) ** 2)
            return score
        if kind == "eqn_nll":
            # Synthetic-likelihood equation belief: "an informant saw the
            # identity hold on k data points with residual scatter <= sigma".
            # Their report is the sufficient statistics of k Gaussian residual
            # draws (mean 0, variance sigma²), whose NLL under the model's
            # actual residual moments (mu_r, v_r) is
            #   k · KL( N(0, sigma²) ‖ N(mu_r, v_r) ).
            # The variance side is one-sided: believing an equation means
            # "residual *at most* sigma", so a residual tighter than sigma is
            # free — implemented as v_eff = max(v_r, sigma²), which reduces the
            # KL to the plain mean term mu_r²/(2 sigma²) below the floor and
            # doubles as the tau floor (bounded 1/v gradients).  Above the
            # floor the k·log(v_eff/sigma²) term prices the escape: conflicts
            # with other constraints resolve by loosening the equation instead
            # of silently distorting marginals.  beta as in expect_nll; like
            # prob_nll it also accepts a traced runtime override ``beta_t``
            # (marked via ``score.takes_beta``) so a per-step beta_schedule
            # anneals equation stiffness on the same footing as the
            # pseudo-count kinds.
            g, sigma, k = cst[1], float(cst[2]), float(cst[3])
            beta = float(cst[4]) if len(cst) > 4 else 0.5
            s2 = sigma * sigma
            def score(x, beta_t=None):
                b = beta if beta_t is None else beta_t
                r = g(x)
                mu, vr = jnp.mean(r), jnp.var(r)
                v_eff = jnp.maximum(vr, s2)
                # debias: E[mu_hat²] = mu² + v_r/N over the batch
                m2 = mu ** 2 - vr / r.shape[0]
                kl = 0.5 * (s2 / v_eff + m2 / v_eff - 1.0
                            + jnp.log(v_eff / s2))
                # β-scale by the k-sample mean's variance v_eff/k (the analog
                # of expect_nll's s², floored at σ²/k) so the two kinds price
                # their escapes on the same footing.
                return jax.lax.stop_gradient(v_eff / k) ** b * k * kl
            score.takes_beta = True
            return score
        if kind == "mmd":
            sites, ref = cst[1], cst[2]
            w = cst[3] if len(cst) > 3 else 1.0
            sites = (sites,) if isinstance(sites, int) else tuple(sites)
            Y = jnp.asarray(np.atleast_2d(np.asarray(ref, np.float32)), jnp.float32)
            if Y.shape[1] != len(sites):                 # (M,) ref for a single site
                Y = Y.reshape(-1, len(sites))
            bandwidths = self._median_bandwidths(np.asarray(Y))
            cols = jnp.asarray(sites)
            return lambda x: w * self._mmd2(x[:, cols], Y, bandwidths)
        raise ValueError(f"unknown constraint kind {kind!r}")

    def _make_onoff(self, gate_idx, f, given, target, value_sd, p_broken,
                    space: str = "abs", direction: str = "eq"):
        """Scoring closure ``(x, gates) -> scalar`` for a robust on/off constraint.

        The sample-native analogue of :func:`calibrated_response.tn.losses.onoff_expectation`::

            KL( Bernoulli(q) || Bernoulli(1 - p_broken) )  +  w · ( q · (mu - target) )²

        where ``q = sigmoid(gates[gate_idx])`` is a learnable credence, ``mu`` is the
        (conditional) expectation of ``f`` estimated from the sample batch, and
        ``w = 1/(2 value_sd²)`` encodes the Gaussian belief width.  The gate scales
        the *residual* (division-free), so convicting the constraint (``q -> 0``)
        smoothly switches its pull off at a KL price of ``-log p_broken`` nats.

        ``space="logit"`` computes the residual in log-odds (for probability
        beliefs; ``value_sd`` is then a log-odds width) — see ``logit_expect``.

        ``direction`` (``"eq"`` default, else ``"le"``/``"ge"``) makes the gated
        residual one-sided: the credence only pulls when the soft bound is
        violated, so a satisfied inequality leaves the gate at its prior and
        never distorts the joint.
        """
        w = 1.0 / (2.0 * value_sd * value_sd)
        pi = 1.0 - p_broken
        logit_target = _logit(target) if space == "logit" else None

        def score(x, gates):
            q = jnp.clip(jax.nn.sigmoid(gates[gate_idx]), 1e-6, 1.0 - 1e-6)
            if given is None:
                mu = jnp.mean(f(x))
            else:
                c = given(x)
                mu = jnp.sum(f(x) * c) / (jnp.sum(c) + _EPS)
            resid = (_logit(mu) - logit_target) if space == "logit" \
                else (mu - target)
            if direction == "le":
                resid = jax.nn.relu(resid)
            elif direction == "ge":
                resid = jax.nn.relu(-resid)
            kl = q * jnp.log(q / pi) + (1.0 - q) * jnp.log((1.0 - q) / (1.0 - pi))
            return kl + w * (q * resid) ** 2
        return score

    def _entropy_pairs(self, max_pairs: int = 128, seed: int = 0):
        """Site pairs to score in the pairwise entropy proxy: all ``D(D-1)/2``
        when that is small, otherwise a seeded random subsample."""
        all_pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        if len(all_pairs) <= max_pairs:
            return all_pairs
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        return [all_pairs[k] for k in idx]

    def constraint_loss(self, constraints, entropy_reg: float = 0.0,
                        pair_entropy_reg: float = 0.0,
                        weight_reg: float = 0.0, n_samples: int = 4096,
                        seed: int = 0):
        """Squared-error expectation loss — stochastic ``loss(params, key)``.

        The latent batch is redrawn from ``key`` on every call, and
        :meth:`optimize` splits a fresh key per Adam step.  (Never optimise
        thousands of steps against one fixed batch: a flexible sampler overfits
        the realised points — minibatch statistics stay great while the true
        expectations drift, catastrophically so for a flow's entropy term.)
        For conditional constraints check :meth:`conditioning_report` (the
        effective sample size is ``N·E[cond]``, so a rare condition needs a
        bigger batch).

        Constraint kinds (see :meth:`_prepare` and :meth:`_make_onoff`)::

            ("expect",            f, target[, weight])
            ("cond_expect",       f, cond, target[, weight])
            ("expect_nll",        f, target, k[, tau[, beta]])        # synthetic likelihood
            ("cond_expect_nll",   f, cond, target, k[, tau[, beta]])
            ("prob_nll",          f, target, k)                       # binomial pseudo-counts
            ("cond_prob_nll",     f, cond, target, k)
            ("logit_expect",      f, target[, weight])        # log-odds residual
            ("logit_cond_expect", f, cond, target[, weight])  # log-odds residual
            ("cov",               f, g, target[, weight])
            ("corr",              f, g, target[, weight])
            ("mmd",               sites, ref_samples[, weight])
            ("eqn_det",           r, weight)                 # residual -> 0
            ("eqn_dist",          r, sigma, weight)          # residual ~ N(0, sigma)
            ("onoff",             f, given, target, value_sd[, p_broken[, space[, direction]]])

        plus the one-sided hinge kinds (``direction`` ``"le"``/``"ge"``), which
        penalise only a violated bound — see :meth:`_prepare`::

            ("expect_ineq",            f, target, weight, direction)
            ("cond_expect_ineq",       f, cond, target, weight, direction)
            ("logit_expect_ineq",      f, target, weight, direction)
            ("logit_cond_expect_ineq", f, cond, target, weight, direction)
            ("corr_ineq",              f, g, target, weight, direction)
            ("eqn_ineq",               r, weight, direction)  # r >= 0 / r <= 0

        The ``onoff`` kind is the sample-native robust constraint: a belief that
        ``E[f | given] = target`` with width ``value_sd``, protected by a learnable
        Bernoulli credence the optimiser can lower (convicting the constraint) at a
        KL cost of ``-log p_broken``.  ``given`` may be ``None`` for an
        unconditional robust constraint.  Each ``onoff`` adds one gate to ``params``
        (see :meth:`init_params` / :meth:`credences`).

        Parameters
        ----------
        constraints : list of tuples
            See the grammar above.
        entropy_reg : float
            Weight on the soft-histogram marginal entropy proxy; positive
            *encourages* spread, with the per-site maximiser being the uniform
            marginal (the optional stand-in for the tensor network's structural
            max-entropy).
        pair_entropy_reg : float
            Weight on the 2-D pair-marginal entropy proxy.  The 1-D proxy alone
            can be satisfied by a degenerate joint (all mass on a curve with
            uniform 1-D marginals); this term penalises that, pushing pairs
            toward independence where the constraints allow it.
        weight_reg : float
            L2 penalty on the sampler parameters ``θ``.
        n_samples : int
            Monte-Carlo latent batch size per step.
        seed : int
            Seeds only the pairwise-entropy pair subsample (the latent batch
            comes from the per-step key).
        """
        scorers = []            # plain  score(x) -> scalar
        gate_scorers = []       # gated  score(x, gates) -> scalar
        gate_pbroken = []
        for c in constraints:
            if c[0] == "onoff":
                f, given, target, value_sd = c[1], c[2], c[3], c[4]
                p_broken = c[5] if len(c) > 5 else 0.05
                space = c[6] if len(c) > 6 else "abs"
                direction = c[7] if len(c) > 7 else "eq"
                gi = len(gate_pbroken)
                gate_pbroken.append(p_broken)
                gate_scorers.append(
                    self._make_onoff(gi, f, given, target, value_sd, p_broken,
                                     space, direction))
            else:
                scorers.append(self._prepare(c))
        self._gate_pbroken = gate_pbroken      # so init_params sizes the gates
        ent_pairs = self._entropy_pairs(seed=seed) if pair_entropy_reg else None

        def body(params, z):
            x = self._sample_x(params, z)          # (N, n), differentiable in θ
            tot = 0.0
            for score in scorers:
                tot = tot + score(x)
            if gate_scorers:
                gates = params["gates"]
                for score in gate_scorers:
                    tot = tot + score(x, gates)
            if entropy_reg:
                tot = tot + entropy_reg * neg_marginal_entropy_proxy(
                    x, self.lower, self.span)
            if pair_entropy_reg:
                tot = tot + pair_entropy_reg * neg_pairwise_entropy_proxy(
                    x, self.lower, self.span, ent_pairs)
            if weight_reg:
                tot = tot + weight_reg * jnp.mean(params["theta"] ** 2)
            return tot

        return self._wrap_loss(body, n_samples)

    def _wrap_loss(self, body, n_samples):
        """Close a ``body(params, z)`` over per-call key-drawn latents:
        the stochastic ``loss(params, key)`` that :meth:`optimize` drives."""
        def loss(params, key):
            z = jax.random.normal(key, (n_samples, self.latent_dim))
            return body(params, z)
        return loss

    # ---- optimisation ------------------------------------------------
    def optimize(self, loss_fn, backend: str = "adam", seed: int = 0,
                 init=None, **kw):
        """Minimise a stochastic ``loss_fn(params, key)`` with Adam, splitting a
        fresh latent-batch key every step.  ``backend`` is kept for signature
        parity with ``TensorTree.optimize`` but only ``"adam"`` is supported —
        deterministic backends (L-BFGS) would require a fixed batch, which the
        sampler losses deliberately do not offer.

        ``**kw`` goes to :func:`~calibrated_response.maxent_sampler.fit.fit_adam_stochastic`:
        ``steps``, ``lr`` (float or optax schedule), ``grad_clip``,
        ``log_every``, the Adam hyper-parameters ``b1``/``b2``/``eps``/
        ``eps_root``/``weight_decay``, or ``optimizer=`` for a hand-built optax
        chain.  ``b2`` and ``eps`` are the ones worth reaching for when a faint
        gradient signal is competing with the per-step Monte-Carlo noise."""
        if backend != "adam":
            raise ValueError(f"backend {backend!r} not supported: sampler losses "
                             "are stochastic (fresh z per step), adam only")
        params = self.init_params(seed=seed) if init is None else init
        return fit_adam_stochastic(loss_fn, params, seed=seed, **kw)

    # ==================================================================
    # query / evaluation  (Monte-Carlo estimates on the actual sampler)
    # ==================================================================
    def expectation(self, params, f: Callable, n_samples: int = 20000, seed: int = 0):
        """Monte-Carlo ``E[f(x)]`` for a callable feature ``f: (N,n) -> (N,)``."""
        x = self._sample_x(params, self._draw_z(n_samples, seed))
        return float(jnp.mean(f(x)))

    def mean(self, params, site: int, n_samples: int = 20000, seed: int = 0):
        """``E[x_site]``."""
        return self.expectation(params, moment(site, 1), n_samples, seed)

    def prob_gt(self, params, site: int, threshold: float,
                n_samples: int = 20000, seed: int = 0):
        """``P(x_site > threshold)`` from continuous samples (exact indicator)."""
        x = self.sample(params, n_samples, seed=seed)
        return float(np.mean(x[:, site] > threshold))

    def event_prob(self, params, event: Callable, n_samples: int = 20000, seed: int = 0):
        """``P(event)`` for a callable event ``event: (N,n) -> (N,)`` in {0,1}."""
        x = self._sample_x(params, self._draw_z(n_samples, seed))
        return float(jnp.mean(event(x)))

    def cond_expectation(self, params, f: Callable, cond: Callable,
                         n_samples: int = 20000, seed: int = 0):
        """``E[f | cond] = E[f·cond]/E[cond]``."""
        x = self._sample_x(params, self._draw_z(n_samples, seed))
        c = cond(x)
        return float(jnp.sum(f(x) * c) / (jnp.sum(c) + _EPS))

    def cond_prob(self, params, event: Callable, given: Callable,
                  n_samples: int = 20000, seed: int = 0):
        """``P(event | given)``."""
        return self.cond_expectation(params, event, given, n_samples, seed)

    def site_marginal(self, params, site: int, n_samples: int = 20000, seed: int = 0):
        """Empirical per-bin mass ``p(X_site = k)`` (uses the var's bins) for overlay."""
        x = self.sample(params, n_samples, seed=seed)
        idx = self.disc.to_index(x)[:, site]
        return np.bincount(idx, minlength=self.dims[site]) / len(idx)

    # ---- conditioning diagnostics ----------------------------------
    def effN(self, params, cond: Callable, n_samples: int = 20000, seed: int = 0):
        """Kish effective sample size ``(Σc)²/Σc²`` for a conditioning function.

        The trustworthy-sample budget behind any ``cond_expect`` / ``cond``
        constraint: if this is small (say < a few hundred) that constraint is
        under-sampled and needs a larger ``n_samples`` in ``constraint_loss``.
        """
        x = self._sample_x(params, self._draw_z(n_samples, seed))
        c = np.asarray(cond(x), np.float64)
        s1, s2 = c.sum(), (c * c).sum()
        return float(s1 * s1 / (s2 + _EPS))

    def conditioning_report(self, params, constraints, n_samples: int = 20000,
                            seed: int = 0):
        """For each conditional constraint, report ``(index, E[cond], effN)``.

        ``effN`` here is scaled to the *fit* batch size, not the eval size, so it
        reflects the actual budget the loss had per conditional constraint.
        """
        x = self._sample_x(params, self._draw_z(n_samples, seed))
        out = []
        for i, cst in enumerate(constraints):
            if cst[0] == "cond_expect":
                cond = cst[2]
            elif cst[0] == "cond":
                cond = cst[2]
            else:
                continue
            c = np.asarray(cond(x), np.float64)
            s1, s2 = c.sum(), (c * c).sum()
            eff_frac = (s1 * s1 / (s2 + _EPS)) / len(c)      # fraction of a batch
            out.append((i, float(c.mean()), eff_frac))
        return out
