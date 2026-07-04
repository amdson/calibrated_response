"""Tensorized smooth & decomposable probabilistic circuit in JAX (spec §2, §5).

The :class:`Circuit` owns the *frozen* region-graph structure and exposes a small
set of **queryable functions**, each a pure function of a ``params`` pytree:

* :meth:`log_prob` / :meth:`log_marginal`     — likelihood pass (§5.1–5.2)
* :meth:`prob`                                — probability of an event (subset marginal)
* :meth:`expectation` / :meth:`linear_moment` — exact moments of ``a·X`` (§5.3), optionally
                                                conditioned on an event

Loss functions are written by the caller on top of these queries; the circuit
itself knows nothing about constraints, the codebase, or training.

Design note — one pass to rule them all
---------------------------------------
Every probability/moment query is an instance of computing

    E[ (a·X) ^ d  *  prod_v 1{X_v in A_v} ]   for d = 0, 1, 2,

propagated bottom-up by carrying the triple ``(m0, m1, m2)`` per region
component (``m0`` = restricted mass, ``m1`` = restricted E[a·X], ``m2`` =
restricted E[(a·X)^2]). Product nodes combine independent sub-scopes; sum nodes
mix *raw* moments linearly (spec §5.3). At the root: ``prob = m0``,
``E[a·X | A] = m1/m0``, ``Var(a·X | A) = m2/m0 - (m1/m0)^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from . import leaves
from .region_graph import build_region_graph


@dataclass
class VarSpec:
    """Declarative description of one variable. Self-contained (no repo deps)."""

    name: str
    kind: str = "gaussian"                  # "gaussian" | "categorical"
    lower: float = 0.0                      # gaussian: domain bounds (init + sampling clamp)
    upper: float = 1.0
    values: Optional[Sequence[float]] = None  # categorical: domain values, e.g. (0.0, 1.0)


class Circuit:
    """A static tensorized SPN over a fixed set of variables."""

    def __init__(self, var_specs: Sequence[VarSpec], C=16, K=4, R=8, seed=0):
        self.var_specs = list(var_specs)
        self.n_vars = len(var_specs)
        self.C, self.K, self.R = C, K, R
        self.rg = build_region_graph(self.n_vars, C, K, R, seed=seed)

        # ---- split leaf regions by variable type ----
        leaf_var = self.rg.leaf_var
        leaf_rid = self.rg.leaf_region_id
        kinds = np.array([self.var_specs[v].kind for v in leaf_var])

        self.g_local = np.where(kinds == "gaussian")[0]      # leaf-local indices
        self.c_local = np.where(kinds == "categorical")[0]
        self.g_region = leaf_rid[self.g_local]
        self.g_var = leaf_var[self.g_local]
        self.c_region = leaf_rid[self.c_local]
        self.c_var = leaf_var[self.c_local]

        # categorical domain: single padded width M with a per-leaf validity mask
        cat_vars = [v for v in var_specs if v.kind == "categorical"]
        self.M = max((len(v.values) for v in cat_vars), default=0)
        if self.M:
            cvals = np.zeros((len(self.c_var), self.M), dtype=np.float32)
            cvalid = np.zeros((len(self.c_var), self.M), dtype=np.float32)
            for i, v in enumerate(self.c_var):
                dom = list(self.var_specs[v].values)
                cvals[i, : len(dom)] = dom
                cvalid[i, : len(dom)] = 1.0
            self.c_values = jnp.asarray(cvals)
            self.c_valid = jnp.asarray(cvalid)

        self.num_regions = self.rg.num_regions

        # bounds vectors (for gaussian variables; categorical entries unused)
        self.lower = np.array([v.lower for v in var_specs], dtype=np.float32)
        self.upper = np.array([v.upper for v in var_specs], dtype=np.float32)

        self._build_jit()

    # ------------------------------------------------------------------
    # parameter initialisation
    # ------------------------------------------------------------------

    def init_params(self, seed=0):
        rng = np.random.default_rng(seed)
        p = {}

        Lg = len(self.g_var)
        if Lg:
            lo = self.lower[self.g_var][:, None]
            hi = self.upper[self.g_var][:, None]
            span = (hi - lo)
            # spread component means across the domain, break symmetry over K
            mu = lo + span * rng.uniform(0.1, 0.9, size=(Lg, self.K))
            logvar = np.log((span / 4.0) ** 2) + rng.normal(0, 0.1, size=(Lg, self.K))
            p["g_mu"] = jnp.asarray(mu, jnp.float32)
            p["g_logvar"] = jnp.asarray(logvar, jnp.float32)
            p["g_mix"] = jnp.asarray(rng.normal(0, 0.1, size=(Lg, self.C, self.K)), jnp.float32)

        Lc = len(self.c_var)
        if Lc:
            p["c_logits"] = jnp.asarray(rng.normal(0, 0.1, size=(Lc, self.K, self.M)), jnp.float32)
            p["c_mix"] = jnp.asarray(rng.normal(0, 0.1, size=(Lc, self.C, self.K)), jnp.float32)

        layer_W = []
        for layer in self.rg.layers:
            P = layer["parent"].shape[0]
            layer_W.append(jnp.asarray(rng.normal(0, 0.1, size=(P, self.C, self.C, self.C)), jnp.float32))
        p["layer_W"] = layer_W

        p["root_W"] = jnp.asarray(rng.normal(0, 0.1, size=(self.R * self.C,)), jnp.float32)
        return p

    # ------------------------------------------------------------------
    # leaf helpers
    # ------------------------------------------------------------------

    def _gauss_leaf_logvals(self, params, x, observed):
        """(Lg, C) log mixture value of each gaussian leaf region."""
        mu, lv, mix = params["g_mu"], params["g_logvar"], params["g_mix"]   # (Lg,K),(Lg,K),(Lg,C,K)
        xv = x[self.g_var][:, None]                                          # (Lg,1)
        obs = observed[self.g_var][:, None]                                 # (Lg,1)
        logf = jnp.where(obs, leaves.gaussian_log_pdf(mu, lv, xv), 0.0)      # (Lg,K)
        log_mix = jax.nn.log_softmax(mix, axis=-1)                           # (Lg,C,K)
        return logsumexp(log_mix + logf[:, None, :], axis=-1)                # (Lg,C)

    def _cat_leaf_logvals(self, params, x, observed):
        logit, mix = params["c_logits"], params["c_mix"]                     # (Lc,K,M),(Lc,C,K)
        logit = logit + jnp.log(self.c_valid[:, None, :] + 1e-30)            # mask invalid cats
        xv = x[self.c_var][:, None, None]                                    # (Lc,1,1) -> broadcasts over (K,M)
        logp_at = leaves.categorical_log_pdf_at(logit, self.c_values[:, None, :], xv)  # (Lc,K)
        obs = observed[self.c_var][:, None]
        logf = jnp.where(obs, logp_at, 0.0)
        log_mix = jax.nn.log_softmax(mix, axis=-1)                           # (Lc,C,K)
        return logsumexp(log_mix + logf[:, None, :], axis=-1)                # (Lc,C)

    def _gauss_leaf_moments(self, params, a, lo, hi):
        mu, lv, mix = params["g_mu"], params["g_logvar"], params["g_mix"]
        lo_v = lo[self.g_var][:, None]
        hi_v = hi[self.g_var][:, None]
        e0, t1, t2 = leaves.gaussian_restricted_moments(mu, lv, lo_v, hi_v)   # (Lg,K)
        w = jax.nn.softmax(mix, axis=-1)                                      # (Lg,C,K)
        m0 = jnp.sum(w * e0[:, None, :], axis=-1)                             # (Lg,C)
        r1 = jnp.sum(w * t1[:, None, :], axis=-1)
        r2 = jnp.sum(w * t2[:, None, :], axis=-1)
        av = a[self.g_var][:, None]                                           # (Lg,1)
        return m0, av * r1, av ** 2 * r2

    def _cat_leaf_moments(self, params, a, cmask):
        logit, mix = params["c_logits"], params["c_mix"]
        logit = logit + jnp.log(self.c_valid[:, None, :] + 1e-30)
        mask = cmask[self.c_var][:, None, :] * self.c_valid[:, None, :]       # (Lc,1,M)
        e0, t1, t2 = leaves.categorical_restricted_moments(
            logit, self.c_values[:, None, :], mask)                          # (Lc,K)
        w = jax.nn.softmax(mix, axis=-1)
        m0 = jnp.sum(w * e0[:, None, :], axis=-1)
        r1 = jnp.sum(w * t1[:, None, :], axis=-1)
        r2 = jnp.sum(w * t2[:, None, :], axis=-1)
        av = a[self.c_var][:, None]
        return m0, av * r1, av ** 2 * r2

    # ------------------------------------------------------------------
    # passes
    # ------------------------------------------------------------------

    def _layer_logW(self, w):
        P = w.shape[0]
        return jax.nn.log_softmax(w.reshape(P, self.C, self.C * self.C), axis=-1).reshape(
            P, self.C, self.C, self.C)

    def _layer_W(self, w):
        P = w.shape[0]
        return jax.nn.softmax(w.reshape(P, self.C, self.C * self.C), axis=-1).reshape(
            P, self.C, self.C, self.C)

    def _log_pass(self, params, x, observed):
        vals = jnp.zeros((self.num_regions, self.C))
        if len(self.g_var):
            vals = vals.at[self.g_region].set(self._gauss_leaf_logvals(params, x, observed))
        if len(self.c_var):
            vals = vals.at[self.c_region].set(self._cat_leaf_logvals(params, x, observed))

        for li, layer in enumerate(self.rg.layers):
            logW = self._layer_logW(params["layer_W"][li])                   # (P,C,C,C)
            uL = vals[layer["left"]]                                         # (P,C)
            uR = vals[layer["right"]]                                        # (P,C)
            comb = uL[:, :, None] + uR[:, None, :]                           # (P,Ci,Cj)
            out = logsumexp(logW + comb[:, None, :, :], axis=(2, 3))         # (P,C)
            vals = vals.at[layer["parent"]].set(out)

        roots = vals[self.rg.rep_root_id].reshape(-1)                        # (R*C,)
        log_w = jax.nn.log_softmax(params["root_W"])
        return logsumexp(log_w + roots)

    def _moment_pass(self, params, a, lo, hi, cmask):
        b0 = jnp.zeros((self.num_regions, self.C))
        b1 = jnp.zeros((self.num_regions, self.C))
        b2 = jnp.zeros((self.num_regions, self.C))
        if len(self.g_var):
            m0, m1, m2 = self._gauss_leaf_moments(params, a, lo, hi)
            b0 = b0.at[self.g_region].set(m0)
            b1 = b1.at[self.g_region].set(m1)
            b2 = b2.at[self.g_region].set(m2)
        if len(self.c_var):
            m0, m1, m2 = self._cat_leaf_moments(params, a, cmask)
            b0 = b0.at[self.c_region].set(m0)
            b1 = b1.at[self.c_region].set(m1)
            b2 = b2.at[self.c_region].set(m2)

        for li, layer in enumerate(self.rg.layers):
            W = self._layer_W(params["layer_W"][li])                         # (P,C,C,C)
            l0, l1, l2 = b0[layer["left"]], b1[layer["left"]], b2[layer["left"]]
            r0, r1, r2 = b0[layer["right"]], b1[layer["right"]], b2[layer["right"]]
            # product of independent sub-scopes, per (i,j)
            p0 = l0[:, :, None] * r0[:, None, :]
            p1 = l1[:, :, None] * r0[:, None, :] + l0[:, :, None] * r1[:, None, :]
            p2 = (l2[:, :, None] * r0[:, None, :]
                  + 2.0 * l1[:, :, None] * r1[:, None, :]
                  + l0[:, :, None] * r2[:, None, :])
            # sum node: mix raw moments
            o0 = jnp.einsum("pcij,pij->pc", W, p0)
            o1 = jnp.einsum("pcij,pij->pc", W, p1)
            o2 = jnp.einsum("pcij,pij->pc", W, p2)
            b0 = b0.at[layer["parent"]].set(o0)
            b1 = b1.at[layer["parent"]].set(o1)
            b2 = b2.at[layer["parent"]].set(o2)

        w = jax.nn.softmax(params["root_W"])
        f0 = b0[self.rg.rep_root_id].reshape(-1)
        f1 = b1[self.rg.rep_root_id].reshape(-1)
        f2 = b2[self.rg.rep_root_id].reshape(-1)
        return jnp.sum(w * f0), jnp.sum(w * f1), jnp.sum(w * f2)

    def _build_jit(self):
        self._jit_logpass = jax.jit(self._log_pass)
        self._jit_moment = jax.jit(self._moment_pass)
        # batched over a stack of evidence points (same observed mask)
        self._jit_logpass_batched = jax.jit(
            jax.vmap(self._log_pass, in_axes=(None, 0, None)))

    # ------------------------------------------------------------------
    # event plumbing
    # ------------------------------------------------------------------

    def encode_event(self, event):
        """Turn an ``event`` dict into ``(lo, hi, cmask)`` arrays.

        ``event`` maps a variable *name* to one of:

        * ``("interval", lo, hi)`` — gaussian var in ``(lo, hi)`` (use ``None``/inf for open side)
        * ``(">", t)`` / ``("<", t)`` — gaussian threshold shorthands
        * ``("=", value)`` — categorical equality
        * ``("in", [values...])`` — categorical set membership

        Unlisted variables are marginalised. ``event`` is static Python data, so
        this builds concrete numpy arrays (constant-folded under ``jit``).
        """
        lo = np.full((self.n_vars,), -np.inf, dtype=np.float32)
        hi = np.full((self.n_vars,), np.inf, dtype=np.float32)
        cmask = np.ones((self.n_vars, self.M), dtype=np.float32) if self.M else np.zeros((self.n_vars, 0), np.float32)
        name_to_idx = {v.name: i for i, v in enumerate(self.var_specs)}

        for name, spec in (event or {}).items():
            i = name_to_idx[name]
            op = spec[0]
            if op == "interval":
                a, b = spec[1], spec[2]
                lo[i] = -np.inf if a is None else a
                hi[i] = np.inf if b is None else b
            elif op == ">":
                lo[i] = spec[1]
            elif op == "<":
                hi[i] = spec[1]
            elif op in ("=", "in"):
                wanted = [spec[1]] if op == "=" else list(spec[1])
                dom = list(self.var_specs[i].values)
                m = np.zeros(self.M, dtype=np.float32)
                for j, val in enumerate(dom):
                    if any(abs(val - w) < 1e-9 for w in wanted):
                        m[j] = 1.0
                cmask[i] = m
            else:
                raise ValueError(f"unknown event op {op!r}")
        return jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(cmask)

    def _coeff_vector(self, a):
        """Accept an ``(n,)`` array, a ``{name: coeff}`` dict, or a single name."""
        if a is None:
            return jnp.zeros((self.n_vars,))
        if isinstance(a, str):
            a = {a: 1.0}
        if isinstance(a, dict):
            vec = np.zeros(self.n_vars, dtype=np.float32)
            name_to_idx = {v.name: i for i, v in enumerate(self.var_specs)}
            for k, val in a.items():
                vec[name_to_idx[k]] = val
            return jnp.asarray(vec)
        return jnp.asarray(a, jnp.float32)

    # ------------------------------------------------------------------
    # public queries
    # ------------------------------------------------------------------

    def log_prob(self, params, x):
        """log p(X = x). ``x`` an ``(n,)`` vector of values."""
        x = jnp.asarray(x, jnp.float32)
        return self._jit_logpass(params, x, jnp.ones((self.n_vars,), bool))

    def log_marginal(self, params, x, observed):
        """log p(X_S = x_S) marginalising the rest. ``observed`` is a bool ``(n,)`` mask."""
        x = jnp.asarray(x, jnp.float32)
        return self._jit_logpass(params, x, jnp.asarray(observed, bool))

    def _subset_inputs(self, x_batch, subset):
        """Build ``(X_full (B, n), observed (n,))`` for a marginal over ``subset``.

        ``x_batch`` holds the values of the subset variables, shape ``(B, |S|)``
        (a 1-D array is treated as ``(B,)`` for a single-variable subset).
        ``subset`` is a list of variable names; ``None`` means the full joint.
        Columns of ``x_batch`` align to the order of ``subset``.
        """
        name_to_idx = {v.name: i for i, v in enumerate(self.var_specs)}
        idxs = (list(range(self.n_vars)) if subset is None
                else [name_to_idx[s] for s in subset])

        xb = jnp.asarray(x_batch, jnp.float32)
        if xb.ndim == 1:
            xb = xb[:, None] if len(idxs) == 1 else xb[None, :]
        B = xb.shape[0]

        observed = np.zeros(self.n_vars, dtype=bool)
        observed[idxs] = True
        x_full = jnp.zeros((B, self.n_vars)).at[:, jnp.asarray(idxs)].set(xb)
        return x_full, jnp.asarray(observed)

    def marginal_log_prob(self, params, x_batch, subset=None):
        """Batched marginal log-density ``log p(X_S = x_S)`` (spec §5.2).

        Evaluates the marginal over the variables in ``subset`` at every point in
        ``x_batch`` (shape ``(B, |S|)``), integrating the rest out. Returns a
        ``(B,)`` array, differentiable in ``params`` — this is the marginal
        interface to build losses on (e.g. fit a marginal by NLL, KL, or by
        matching a target table). For categorical variables the value is a
        probability; for gaussians it is a density.
        """
        x_full, observed = self._subset_inputs(x_batch, subset)
        return self._jit_logpass_batched(params, x_full, observed)

    def marginal_prob(self, params, x_batch, subset=None):
        """``exp`` of :meth:`marginal_log_prob` — batched marginal probability/density."""
        return jnp.exp(self.marginal_log_prob(params, x_batch, subset))

    def prob(self, params, event):
        """P(event) — a subset marginal probability (spec §5.2)."""
        lo, hi, cmask = self.encode_event(event)
        a = jnp.zeros((self.n_vars,))
        m0, _, _ = self._jit_moment(params, a, lo, hi, cmask)
        return m0

    def linear_moment(self, params, a, event=None):
        """Exact ``(mean, var)`` of ``a·X`` (spec §5.3), conditioned on ``event`` if given."""
        av = self._coeff_vector(a)
        lo, hi, cmask = self.encode_event(event)
        m0, m1, m2 = self._jit_moment(params, av, lo, hi, cmask)
        mean = m1 / m0
        var = jnp.clip(m2 / m0 - mean ** 2, a_min=0.0)
        return mean, var

    def expectation(self, params, a, event=None):
        """E[a·X | event] (unconditional if ``event`` is None)."""
        return self.linear_moment(params, a, event)[0]

    def cond_prob(self, params, event, given):
        """P(event AND given) / P(given)."""
        joint = {**(given or {}), **(event or {})}
        return self.prob(params, joint) / self.prob(params, given)
