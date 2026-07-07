"""Robust constraints with credences as **sampled latents** (not gate parameters).

The default :class:`~calibrated_response.maxent_sampler.model.SamplerModel` represents
each ``onoff`` constraint's credence as a single learnable scalar gate.  This variant
instead does what the tensor-network robust losses do: it treats each credence as a
genuine **latent variable of the model**.  The sampler emits the ``n`` data variables
*plus* one extra ``(0, 1)`` output per robust constraint — a per-sample "active"
probability — and the credence is that latent's **marginal** ``q_k = E_z[c_k(z)]``,
read by sampling rather than off the parameters.

This is the sample-native mirror of "30 data variables + 107 binary constraint
latents".  Because the ``onoff`` free energy couples only the *marginals*
(``KL(Bernoulli(q_k) ‖ Bernoulli(1-p_broken))`` plus the gated marginal residual), the
latent's correlation with the data is unconstrained and the result matches the gate
version — but now you can draw the credence latents and look at their distribution.

    m = LatentCredenceModel(vars, n_credences=len(constraints), latent_dim=8)
    p, hist = m.optimize(m.constraint_loss(constraints), steps=3000)
    q = m.credences(p)                 # marginal credence per constraint (sampled)
    C = m.credence_samples(p)          # (N, n_credences) per-sample latent draws
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from calibrated_response.tn.discretize import ContinuousVar, Discretizer
from calibrated_response.maxent_sampler.model import SamplerModel
from calibrated_response.maxent_sampler.sampler_model import (
    NeuralSampler, neg_marginal_entropy_proxy, neg_pairwise_entropy_proxy,
)

_EPS = 1e-30


class LatentCredenceModel(SamplerModel):
    """Sampler whose ``onoff`` credences are sampled latent outputs.

    Parameters
    ----------
    vars : sequence of ContinuousVar
        The data variables (defines ``n`` and their domains).
    n_credences : int
        Number of robust (``onoff``) constraints you intend to fit — the sampler
        gets this many extra ``(0, 1)`` latent outputs, one credence each.
    latent_dim, hidden_sizes, out_scale :
        As in :class:`NeuralSampler`.
    """

    def __init__(self, vars: Sequence[ContinuousVar], n_credences: int,
                 latent_dim: int = 8, hidden_sizes=64, out_scale: float = 1.0):
        self.disc = Discretizer(vars)
        self.n = self.disc.n_sites
        self.dims = self.disc.dims
        self.n_cred = int(n_credences)
        self.latent_dim = int(latent_dim)
        self._gate_pbroken = []                 # unused here; keeps init_params happy

        # One network emitting data dims first, then the credence latents.
        self.net = NeuralSampler(self.n + self.n_cred, self.latent_dim,
                                 hidden_sizes, out_scale=out_scale)
        self.lower = jnp.asarray(self.disc.lower, jnp.float32)
        self.span = jnp.asarray(self.disc.upper - self.disc.lower, jnp.float32)

    def init_params(self, seed: int = 0, prior_active: float = 0.95):
        """Initialise params, biasing the credence-latent outputs to start near the
        prior ``prior_active`` (mirrors the gate model's ``logit(1-p_broken)`` init,
        so true constraints do not have to be *raised* from 0.5 while the data fits)."""
        params = super().init_params(seed=seed)
        pytree = self.net.unpack_params(params["theta"])
        logit = float(np.log(prior_active / (1.0 - prior_active)))
        b = pytree[-1]["b"].at[self.n:].set(logit)     # output biases of the latent rows
        pytree[-1]["b"] = b
        params["theta"] = self.net.pack_params(pytree)
        return params

    # ---- split the network output into (data in domain, credence latents) ----
    def _sample_full(self, params, z):
        u = jax.vmap(self.net.sample_fn_flat, in_axes=(None, 0))(params["theta"], z)
        x = self.lower + self.span * u[:, :self.n]      # (N, n) data in domains
        c = u[:, self.n:]                               # (N, n_cred) credence latents in (0,1)
        return x, c

    def _sample_x(self, params, z):
        """Data variables only (so inherited sample()/queries behave as usual)."""
        u = jax.vmap(self.net.sample_fn_flat, in_axes=(None, 0))(params["theta"], z)
        return self.lower + self.span * u[:, :self.n]

    # ---- credence read-out: by SAMPLING the latents ----
    def credence_samples(self, params, n_samples: int = 20000, seed: int = 0):
        """Per-sample credence latent draws, shape ``(n_samples, n_credences)``."""
        _, c = self._sample_full(params, self._draw_z(n_samples, seed))
        return np.asarray(c)

    def credences(self, params, n_samples: int = 20000, seed: int = 0):
        """Marginal credence ``q_k = E[c_k]`` per constraint (sampled read-out)."""
        return self.credence_samples(params, n_samples, seed).mean(axis=0)

    # ---- loss: onoff credences come from the sampled latent marginals ----
    def constraint_loss(self, constraints, entropy_reg: float = 0.0,
                        pair_entropy_reg: float = 0.0,
                        weight_reg: float = 0.0, n_samples: int = 4096,
                        seed: int = 0):
        """Same grammar as :meth:`SamplerModel.constraint_loss`, but each ``onoff``
        constraint's credence is the marginal of a sampled latent output rather than
        a gate parameter.  The k-th ``onoff`` constraint uses latent column ``k``."""
        plain = []                 # score(x) -> scalar
        gated = []                 # (latent_idx, f, given, target, w, pi)
        ci = 0
        for c in constraints:
            if c[0] == "onoff":
                f, given, target, value_sd = c[1], c[2], c[3], c[4]
                p_broken = c[5] if len(c) > 5 else 0.05
                gated.append((ci, f, given, target,
                              1.0 / (2.0 * value_sd * value_sd), 1.0 - p_broken))
                ci += 1
            else:
                plain.append(self._prepare(c))
        if ci > self.n_cred:
            raise ValueError(f"{ci} onoff constraints but only {self.n_cred} "
                             f"credence latents; construct with n_credences>={ci}")
        ent_pairs = self._entropy_pairs(seed=seed) if pair_entropy_reg else None

        def body(params, z):
            x, cl = self._sample_full(params, z)
            tot = 0.0
            for score in plain:
                tot = tot + score(x)
            for (k, f, given, target, w, pi) in gated:
                q = jnp.clip(jnp.mean(cl[:, k]), 1e-6, 1.0 - 1e-6)
                if given is None:
                    mu = jnp.mean(f(x))
                else:
                    g = given(x)
                    mu = jnp.sum(f(x) * g) / (jnp.sum(g) + _EPS)
                kl = q * jnp.log(q / pi) + (1.0 - q) * jnp.log((1.0 - q) / (1.0 - pi))
                tot = tot + kl + w * (q * (mu - target)) ** 2
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
