"""Pure multivariate-Gaussian baseline with the circuit query API (sanity check).

A deliberately dumb stand-in for :class:`~calibrated_response.pc.circuit.Circuit`:
the whole model is a single joint Gaussian ``N(mu, Sigma)`` — *literally a mean
vector and a covariance matrix* — exposing the same query surface
(``expectation`` / ``prob`` / ``cond_prob`` / ``marginal_log_prob`` /
``linear_moment``) so the same ``losses`` terms and ``train`` loop drive it. Use
it to see how much of a constraint set is already explained by a plain Gaussian
before crediting the circuit's mixture structure.

Scope (intentionally narrow — see the toy notebook for the contrast):

* **Gaussian variables only.** A pure Gaussian has no native notion of a
  categorical, so a categorical ``VarSpec`` raises at construction rather than
  being silently relaxed.
* **Everything is exact, or it raises.** Marginals and unconditional moments are
  exact for any subset. Threshold/interval *probabilities* are exact for one
  variable (Gaussian CDF) and two variables (bivariate-normal CDF via Plackett's
  integral, fully differentiable); three or more jointly-constrained variables
  have no closed form, so :meth:`prob` raises there instead of approximating.
  Conditional moments are exact when conditioning on a single variable's interval
  (truncated-Gaussian moments); conditioning on two or more raises.

What a single Gaussian cannot do — and why this is *only* a baseline: it is
unimodal and its dependence is purely linear, so it cannot carve the threshold
troughs or represent the law-of-total-expectation bimodality the mixture circuit
captures (notebook problems 3-5). It will fit marginal means/variances and
pairwise correlations and nothing finer.

Regularizers: ``losses.isotropy_regularizer``, ``uniform_coverage_regularizer``
and ``gaussian_crossentropy_ref`` work unchanged (they only touch the query API).
``dirichlet_regularizer`` and ``leaf_entropy_regularizer`` are circuit-weight
priors and do **not** apply — pass ``w_dir=0, w_ent=0``.

Parametrization: ``mean`` is free; ``Sigma = L Lᵀ`` with ``L`` lower-triangular
and a strictly-positive ``exp``-parametrized diagonal, so the covariance is PSD
by construction and every query is differentiable in both.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc

from .circuit import VarSpec

# Large finite sentinel for open interval sides. Using a finite value (rather
# than inf) keeps the standardized argument finite, so autodiff through the CDF
# w.r.t. (mu, sigma) yields a clean 0 instead of nan at an open boundary.
_LARGE = 1e8
_BIG = 1e7                       # "is this side effectively open?" threshold
_INV_SQRT2 = 0.7071067811865476
_LOG2PI = float(np.log(2.0 * np.pi))

# Gauss-Legendre nodes/weights on [-1, 1] for Plackett's bivariate-normal integral.
_GLx, _GLw = np.polynomial.legendre.leggauss(24)
_GLx = jnp.asarray(_GLx, jnp.float32)
_GLw = jnp.asarray(_GLw, jnp.float32)


# ---------------------------------------------------------------------------
# standard-normal primitives (differentiable, inf-safe via finite sentinels)
# ---------------------------------------------------------------------------

def _Phi(z):
    """Standard-normal CDF."""
    return 0.5 * erfc(-z * _INV_SQRT2)


def _phi(z):
    """Standard-normal pdf."""
    return jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)


def _bvn_std(x, y, rho):
    """Standardized bivariate-normal CDF ``P(X<=x, Y<=y)`` at correlation ``rho``.

    Plackett's identity: ``Phi2(x,y;rho) = Phi(x)Phi(y) + ∫_0^rho phi2(x,y;t) dt``,
    with ``d Phi2 / d rho = phi2`` (Price's theorem). The 1-D integral over the
    correlation is done by Gauss-Legendre quadrature — smooth and differentiable
    in ``x, y, rho``. With the ``_LARGE`` sentinels an "open" side drives the
    integrand to 0 and ``Phi`` to 0/1, so the box reduction needs no inf cases.
    """
    rho = jnp.clip(rho, -0.9999, 0.9999)
    t = 0.5 * rho * (_GLx + 1.0)              # map [-1,1] -> [0,rho]
    w = 0.5 * rho * _GLw
    omt2 = 1.0 - t * t
    integrand = jnp.exp(-(x * x - 2.0 * t * x * y + y * y) / (2.0 * omt2)) / (
        2.0 * jnp.pi * jnp.sqrt(omt2))
    return _Phi(x) * _Phi(y) + jnp.sum(w * integrand)


def _mvn_logpdf(x, mu, cov):
    """log N(x; mu, cov) for ``x`` shape ``(B, d)``, returns ``(B,)``."""
    d = mu.shape[0]
    L = jnp.linalg.cholesky(cov)
    diff = x - mu                                          # (B,d)
    z = jax.scipy.linalg.solve_triangular(L, diff.T, lower=True).T   # (B,d)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
    return -0.5 * (jnp.sum(z * z, axis=-1) + d * _LOG2PI + logdet)


class GaussianModel:
    """A single joint Gaussian exposing the :class:`Circuit` query API."""

    is_gaussian_baseline = True

    def __init__(self, var_specs, seed=0):
        self.var_specs = list(var_specs)
        bad = [v.name for v in self.var_specs if v.kind != "gaussian"]
        if bad:
            raise ValueError(
                f"GaussianModel is Gaussian-only; non-gaussian VarSpec(s): {bad}. "
                "Use Circuit for categorical variables.")
        self.n_vars = len(self.var_specs)
        self.M = 0                                  # no categoricals (API parity)
        self.lower = np.array([v.lower for v in self.var_specs], dtype=np.float32)
        self.upper = np.array([v.upper for v in self.var_specs], dtype=np.float32)
        self._name_to_idx = {v.name: i for i, v in enumerate(self.var_specs)}

    # ------------------------------------------------------------------
    # parameters: mean (n,) and a Cholesky factor L (n,n), Sigma = L Lᵀ
    # ------------------------------------------------------------------

    def init_params(self, seed=0):
        rng = np.random.default_rng(seed)
        n = self.n_vars
        mid = 0.5 * (self.lower + self.upper)
        sigma = np.maximum(self.upper - self.lower, 1.0) / 4.0     # spec init scale
        L_raw = rng.normal(0, 1e-3, size=(n, n)).astype(np.float32)  # break symmetry
        L_raw[np.diag_indices(n)] = np.log(sigma)                    # diag = log sd
        return {"mean": jnp.asarray(mid, jnp.float32), "L_raw": jnp.asarray(L_raw)}

    @staticmethod
    def _L(params):
        L_raw = params["L_raw"]
        n = L_raw.shape[0]
        return jnp.tril(L_raw, -1) + jnp.diag(jnp.exp(jnp.diagonal(L_raw)))

    def _mean(self, params):
        return params["mean"]

    def _cov(self, params):
        L = self._L(params)
        return L @ L.T

    # ------------------------------------------------------------------
    # event plumbing
    # ------------------------------------------------------------------

    def _coeff_vector(self, a):
        """Accept an ``(n,)`` array, a ``{name: coeff}`` dict, or a single name."""
        if a is None:
            return jnp.zeros((self.n_vars,))
        if isinstance(a, str):
            a = {a: 1.0}
        if isinstance(a, dict):
            vec = np.zeros(self.n_vars, dtype=np.float32)
            for k, val in a.items():
                vec[self._name_to_idx[k]] = val
            return jnp.asarray(vec)
        return jnp.asarray(a, jnp.float32)

    def _event_to_AB(self, event):
        """Turn an ``event`` dict into per-variable interval bounds ``(A, B)``.

        Returned as **numpy** (the event is static Python data), so the set of
        constrained variables can be inspected even when called under ``jit``.
        Open sides use the ``_LARGE`` sentinel. Only Gaussian ops are accepted:
        ``("interval", lo, hi)``, ``(">", t)``, ``("<", t)``.
        """
        A = np.full((self.n_vars,), -_LARGE, dtype=np.float32)
        B = np.full((self.n_vars,), _LARGE, dtype=np.float32)
        for name, spec in (event or {}).items():
            i = self._name_to_idx[name]
            op = spec[0]
            if op == "interval":
                lo, hi = spec[1], spec[2]
                A[i] = -_LARGE if lo is None else lo
                B[i] = _LARGE if hi is None else hi
            elif op == ">":
                A[i] = spec[1]
            elif op == "<":
                B[i] = spec[1]
            elif op in ("=", "in"):
                raise ValueError(
                    f"event op {op!r} on {name!r} is categorical; GaussianModel "
                    "is Gaussian-only.")
            else:
                raise ValueError(f"unknown event op {op!r}")
        return A, B

    @staticmethod
    def _constrained(A, B):
        """Indices with a real (non-sentinel) bound on either side (static)."""
        return [i for i in range(len(A)) if A[i] > -_BIG or B[i] < _BIG]

    def encode_event(self, event):
        """``(lo, hi, cmask)`` in the circuit's format (cmask empty: no categoricals)."""
        A, B = self._event_to_AB(event)
        lo = np.where(A <= -_BIG, -np.inf, A)
        hi = np.where(B >= _BIG, np.inf, B)
        return jnp.asarray(lo), jnp.asarray(hi), jnp.zeros((self.n_vars, 0), jnp.float32)

    # ------------------------------------------------------------------
    # box probability (exact for k<=2, raises for k>=3)
    # ------------------------------------------------------------------

    def _box_prob(self, params, A, B, idxs):
        mu, cov = self._mean(params), self._cov(params)
        k = len(idxs)
        if k == 0:
            return jnp.asarray(1.0)
        if k == 1:
            i = idxs[0]
            s = jnp.sqrt(cov[i, i])
            return jnp.clip(_Phi((B[i] - mu[i]) / s) - _Phi((A[i] - mu[i]) / s), 0.0, 1.0)
        if k == 2:
            i, j = idxs
            si, sj = jnp.sqrt(cov[i, i]), jnp.sqrt(cov[j, j])
            rho = cov[i, j] / (si * sj)
            xa, xb = (A[i] - mu[i]) / si, (B[i] - mu[i]) / si
            ya, yb = (A[j] - mu[j]) / sj, (B[j] - mu[j]) / sj
            P = (_bvn_std(xb, yb, rho) - _bvn_std(xa, yb, rho)
                 - _bvn_std(xb, ya, rho) + _bvn_std(xa, ya, rho))
            return jnp.clip(P, 0.0, 1.0)
        raise NotImplementedError(
            f"prob() over {k} jointly-constrained variables has no closed form "
            "for a Gaussian; only k<=2 is exact. (Constrained vars: "
            f"{[self.var_specs[i].name for i in idxs]}.)")

    # ------------------------------------------------------------------
    # conditional linear moments (exact for <=1 constrained variable)
    # ------------------------------------------------------------------

    def _cond_lin_moment(self, params, a, A, B, idxs):
        """``(mean, var)`` of ``a·X`` given ``X in box``. Exact for ``len(idxs)<=1``."""
        mu, cov = self._mean(params), self._cov(params)
        if len(idxs) == 0:
            mean = a @ mu
            var = a @ cov @ a
            return mean, jnp.clip(var, 0.0)
        if len(idxs) == 1:
            i = idxs[0]
            s = jnp.sqrt(cov[i, i])
            al, be = (A[i] - mu[i]) / s, (B[i] - mu[i]) / s
            pa, pb = _phi(al), _phi(be)
            p = jnp.clip(_Phi(be) - _Phi(al), 1e-12)
            lam = (pa - pb) / p                       # truncated standardized mean
            eta = (al * pa - be * pb) / p             # al*pa -> 0 at sentinel (huge*0)
            tvar = jnp.clip(1.0 + eta - lam * lam, 0.0, 1.0)
            ai = cov[:, i]                            # Sigma[:, i]
            mean = a @ mu + (a @ ai) * (lam / s)      # = a·mu + (a·Σ_i / s)·s·lam / s
            var = a @ cov @ a - (1.0 - tvar) * (a @ ai) ** 2 / cov[i, i]
            return mean, jnp.clip(var, 0.0)
        raise NotImplementedError(
            f"conditional moment given {len(idxs)} variables is not closed-form "
            "for a Gaussian; only single-variable conditioning is exact. "
            f"(Conditioning vars: {[self.var_specs[i].name for i in idxs]}.)")

    # ------------------------------------------------------------------
    # public queries (same names/signatures as Circuit)
    # ------------------------------------------------------------------

    def prob(self, params, event):
        """P(event) — exact for events touching <=2 variables."""
        A, B = self._event_to_AB(event)
        return self._box_prob(params, A, B, self._constrained(A, B))

    def cond_prob(self, params, event, given):
        """P(event AND given) / P(given)."""
        joint = {**(given or {}), **(event or {})}
        return self.prob(params, joint) / self.prob(params, given)

    def linear_moment(self, params, a, event=None):
        """Exact ``(mean, var)`` of ``a·X``, conditioned on ``event`` if given."""
        av = self._coeff_vector(a)
        A, B = self._event_to_AB(event)
        return self._cond_lin_moment(params, av, A, B, self._constrained(A, B))

    def expectation(self, params, a, event=None):
        """E[a·X | event] (unconditional if ``event`` is None)."""
        return self.linear_moment(params, a, event)[0]

    def _moment_pass(self, params, a, lo, hi, cmask):
        """Unconditional raw moments ``(m0, m1, m2) = (1, E[a·X], E[(a·X)^2])``.

        Mirrors ``Circuit._moment_pass`` *only for the open box* — which is all
        ``losses.isotropy_regularizer`` ever asks for (it passes ``±inf`` bounds).
        The ``lo/hi/cmask`` arguments are accepted for signature parity but the
        baseline does not read them (they are traced under ``jit``); for genuinely
        restricted moments use :meth:`prob` / :meth:`linear_moment` with an event.
        """
        mu, cov = self._mean(params), self._cov(params)
        mean = a @ mu
        return jnp.asarray(1.0), mean, a @ cov @ a + mean ** 2

    # ---- marginal densities (exact Gaussian sub-density) ----

    def _subset_idxs(self, subset):
        return (list(range(self.n_vars)) if subset is None
                else [self._name_to_idx[s] for s in subset])

    def marginal_log_prob(self, params, x_batch, subset=None):
        """Batched marginal log-density ``log p(X_S = x_S)`` (exact Gaussian)."""
        idxs = self._subset_idxs(subset)
        xb = jnp.asarray(x_batch, jnp.float32)
        if xb.ndim == 1:
            xb = xb[:, None] if len(idxs) == 1 else xb[None, :]
        if len(idxs) == 0:
            return jnp.zeros((xb.shape[0],))
        ii = np.asarray(idxs)
        mu = self._mean(params)[ii]
        cov = self._cov(params)[ii[:, None], ii[None, :]]
        return _mvn_logpdf(xb, mu, cov)

    def marginal_prob(self, params, x_batch, subset=None):
        """``exp`` of :meth:`marginal_log_prob`."""
        return jnp.exp(self.marginal_log_prob(params, x_batch, subset))

    def log_marginal(self, params, x, observed):
        """log p(X_S = x_S) with ``observed`` a bool ``(n,)`` mask."""
        x = jnp.asarray(x, jnp.float32)
        idxs = [i for i in range(self.n_vars) if bool(np.asarray(observed)[i])]
        if not idxs:
            return jnp.asarray(0.0)                  # logZ = 0
        return self.marginal_log_prob(params, x[jnp.asarray(idxs)][None, :],
                                      subset=[self.var_specs[i].name for i in idxs])[0]

    def log_prob(self, params, x):
        """log p(X = x). ``x`` an ``(n,)`` vector."""
        return self.log_marginal(params, x, np.ones(self.n_vars, bool))

    # ------------------------------------------------------------------
    # sampling (exact: mu + z Lᵀ), clamped to declared bounds like Circuit
    # ------------------------------------------------------------------

    def sample(self, params, n_samples, seed=0):
        rng = np.random.default_rng(seed)
        mu = np.asarray(self._mean(params))
        L = np.asarray(self._L(params))
        z = rng.normal(size=(n_samples, self.n_vars))
        X = mu[None, :] + z @ L.T
        return np.clip(X, self.lower[None, :], self.upper[None, :])
