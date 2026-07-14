"""SamplerModel variant backed by an invertible flow with **exact** entropy.

Same constraint grammar and query API as
:class:`~calibrated_response.maxent_sampler.model.SamplerModel`, but the
sampler is a :class:`~calibrated_response.maxent_sampler.flow_sampler.FlowSampler`
and ``entropy_reg`` weights the *exact joint* differential entropy instead of a
marginal histogram proxy::

    loss = sum constraint penalties  -  entropy_reg * H(x)
    H(x) = H(z) + E[log |det J|] + sum log span        (exact, O(D)/sample)

``entropy_reg=1.0`` is the natural scale: the loss is then a true
soft-constrained maximum-entropy objective.  Degenerate joints (mass on any
lower-dimensional manifold) have ``H = -inf``, so collapse is impossible rather
than merely discouraged — and unlike the histogram proxies this guards *all*
orders of structure, not just 1-D/2-D marginals, at a cost linear in D.

    m = FlowSamplerModel(vars, n_layers=8, hidden=64)
    p, hist = m.optimize(m.constraint_loss(constraints, entropy_reg=1.0),
                         steps=3000, lr=1e-3)
    m.entropy(p)        # exact joint entropy of the fit (nats)
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from calibrated_response.tn.discretize import ContinuousVar, Discretizer
from calibrated_response.maxent_sampler.model import SamplerModel
from calibrated_response.maxent_sampler.flow_sampler import FlowSampler

_EPS = 1e-30


class FlowSamplerModel(SamplerModel):
    """Invertible-sampler model: exact joint entropy, same API as SamplerModel.

    Parameters
    ----------
    vars : sequence of ContinuousVar
        Variable specs; the flow's latent dimension is forced to ``len(vars)``
        (invertibility).
    n_layers, hidden, s_max :
        As in :class:`FlowSampler`.
    """

    def __init__(self, vars: Sequence[ContinuousVar], n_layers: int = 8,
                 hidden: int = 64, s_max: float = 3.0):
        self.disc = Discretizer(vars)
        self.n = self.disc.n_sites
        self.dims = self.disc.dims
        self.latent_dim = self.n                     # invertible: same dim

        self.net = FlowSampler(self.n, n_layers=n_layers, hidden=hidden,
                               s_max=s_max)
        self.lower = jnp.asarray(self.disc.lower, jnp.float32)
        self.span = jnp.asarray(self.disc.upper - self.disc.lower, jnp.float32)
        self._gate_pbroken = []

        # H(z) + sum log span: the constant part of H(x).
        self._h_const = (0.5 * self.n * float(np.log(2.0 * np.pi * np.e))
                         + float(jnp.sum(jnp.log(self.span))))

    # ---- sampling with log-det --------------------------------------
    def _sample_x_logdet(self, params, z):
        u, ld = jax.vmap(self.net.forward_flat, in_axes=(None, 0))(
            params["theta"], z)
        return self.lower + self.span * u, ld

    def entropy(self, params, n_samples: int = 20000, seed: int = 0):
        """Exact joint differential entropy ``H(x)`` in nats (MC over z)."""
        _, ld = self._sample_x_logdet(params, self._draw_z(n_samples, seed))
        return float(self._h_const + jnp.mean(ld))

    def log_prob(self, params, x, chunk: int = 65536):
        """Exact log density at points ``x`` (N, n) in original units (numpy).

        Change of variables through the inverse flow::

            log p(x) = log N(z(x)) - log|det J_g(z(x))| - sum log span

        The flow is a tractable density model as well as a sampler — this is
        what e.g. held-out NLL evaluation uses."""
        x = np.atleast_2d(np.asarray(x, np.float32))
        u = (jnp.asarray(x) - self.lower) / self.span
        inv = jax.vmap(self.net.inverse_flat, in_axes=(None, 0))
        out = []
        for k in range(0, len(x), chunk):
            z, ld = inv(params["theta"], u[k:k + chunk])
            log_n = (-0.5 * jnp.sum(z * z, axis=1)
                     - 0.5 * self.n * float(np.log(2.0 * np.pi)))
            out.append(np.asarray(log_n - ld))
        return np.concatenate(out) - float(jnp.sum(jnp.log(self.span)))

    # ---- loss: same grammar, exact entropy term ----------------------
    def constraint_loss(self, constraints, entropy_reg: float = 1.0,
                        weight_reg: float = 0.0, n_samples: int = 4096,
                        seed: int = 0, domain_prior: str = "uniform",
                        prior_bound_sds: float = 2.0, ref_mask=None):
        """Same constraint grammar as :meth:`SamplerModel.constraint_loss`;
        ``entropy_reg`` weights the **exact** joint entropy (default 1.0 — the
        soft-constrained maxent objective).  No histogram proxies needed.

        ``domain_prior`` selects the reference measure the entropy term is
        implicitly (or explicitly) a KL against:

        * ``"uniform"`` (default): the plain ``- entropy_reg * H(x)`` term.
          Maxent on a box == min KL(p ‖ Uniform(box)) up to a constant.
        * ``"gaussian"``: ``+ entropy_reg * KL(p ‖ q0)`` with a factorized
          reference ``q0_i = Normal(mid_i, span_i / (2 * prior_bound_sds))``
          per site — the elicited bounds are read as ±k·sd of a default
          belief, so conservative bounds widen the default instead of
          flattening it.  ``KL = -H - E_p[log q0]``; the ``-H`` term is the
          same exact entropy machinery, so anti-collapse survives.  Gradients
          under ``"uniform"`` are identical to the pre-KL objective (log q0
          would be constant); loss VALUES shift by a constant between modes.

        ``ref_mask`` (length-n, 1.0/0.0) zeroes the Gaussian log q0 on chosen
        sites — binary sites keep their Uniform(0,1) reference (log q0 = 0)
        for free.  Default: all ones.

        Stochastic ``loss(params, key)`` like the parent: fresh latents every
        step.  This matters doubly for a flow — it is exactly the kind of
        flexible model that overfits a fixed batch, warping to put the training
        ``z_i`` in high ``log|det J|`` regions while the true entropy collapses
        between them."""
        if domain_prior not in ("uniform", "gaussian"):
            raise ValueError(f"domain_prior must be 'uniform' or 'gaussian', "
                             f"got {domain_prior!r}")
        scorers = []
        gate_scorers = []
        gate_pbroken = []
        for c in constraints:
            if c[0] == "onoff":
                f, given, target, value_sd = c[1], c[2], c[3], c[4]
                p_broken = c[5] if len(c) > 5 else 0.05
                space = c[6] if len(c) > 6 else "abs"
                gi = len(gate_pbroken)
                gate_pbroken.append(p_broken)
                gate_scorers.append(
                    self._make_onoff(gi, f, given, target, value_sd, p_broken,
                                     space))
            else:
                scorers.append(self._prepare(c))
        self._gate_pbroken = gate_pbroken

        h_const = self._h_const
        if domain_prior == "gaussian":
            ref_mu = self.lower + 0.5 * self.span
            ref_sd = self.span / (2.0 * float(prior_bound_sds))
            mask = (jnp.ones(self.n, jnp.float32) if ref_mask is None
                    else jnp.asarray(ref_mask, jnp.float32))
            log_norm = -0.5 * jnp.log(2.0 * jnp.pi * ref_sd ** 2)

        def body(params, z):
            # One forward pass: constraints score the samples x, the entropy
            # term is the mean per-sample log-likelihood on that SAME batch
            # (log p(x_i) = log N(z_i) - logdet_i; the z-density term is
            # theta-independent, so h_const carries its analytic expectation).
            x, ld = self._sample_x_logdet(params, z)
            tot = 0.0
            for score in scorers:
                tot = tot + score(x)
            if gate_scorers:
                gates = params["gates"]
                for score in gate_scorers:
                    tot = tot + score(x, gates)
            if entropy_reg:
                ent = h_const + jnp.mean(ld)               # H(p), exact
                if domain_prior == "uniform":
                    tot = tot - entropy_reg * ent          # maxent
                else:
                    logq0 = jnp.mean(jnp.sum(mask * (
                        log_norm
                        - (x - ref_mu) ** 2 / (2.0 * ref_sd ** 2)), axis=1))
                    tot = tot + entropy_reg * (-ent - logq0)  # KL(p ‖ q0)
            if weight_reg:
                tot = tot + weight_reg * jnp.mean(params["theta"] ** 2)
            return tot

        return self._wrap_loss(body, n_samples)
