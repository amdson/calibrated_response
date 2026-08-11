"""SamplerModel variant backed by an invertible flow with **exact** entropy.

Same constraint grammar and query API as
:class:`~calibrated_response.maxent_sampler.model.SamplerModel`, but the
sampler is a :class:`~calibrated_response.maxent_sampler.flow_sampler.FlowSampler`
and ``entropy_reg`` weights the *exact joint* differential entropy instead of a
marginal histogram proxy::

    loss = sum constraint penalties  -  entropy_reg * H(x)
    H(x) = H(z) + E[log |det J|] + sum log span        (exact, O(D)/sample)

``entropy_reg=1.0`` is the natural scale: the loss is then a true
soft-constrained maximum-entropy objective.  ``domain_prior="gaussian"``
generalizes it to ``+ entropy_reg * KL(p ‖ q0)`` with a per-site Gaussian
reference centered in the box (bounds read as ±``prior_bound_sds``·sd of a
default belief) — same exact entropy machinery, one extra O(D)/sample term.  Degenerate joints (mass on any
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
from calibrated_response.maxent_sampler.spline_sampler import SplineFlowSampler

_EPS = 1e-30


class FlowSamplerModel(SamplerModel):
    """Invertible-sampler model: exact joint entropy, same API as SamplerModel.

    Parameters
    ----------
    vars : sequence of ContinuousVar
        Variable specs; the flow's latent dimension is forced to
        ``len(vars) + n_dummy`` (invertibility).
    n_layers, hidden, s_max :
        As in :class:`FlowSampler`.
    n_dummy : int
        Extra flow dimensions carrying no variable.  Dummies widen every
        coupling layer's conditioning input (more expressive transport for the
        same depth) while leaving the objective unchanged in expectation: they
        live on a unit box, so their ``log span`` is 0, the maxent optimum
        keeps them independent-uniform (0 net entropy contribution), and no
        constraint or Gaussian reference ever touches them.  All sampling
        readouts return only the real sites; the entropy term (and
        :meth:`entropy`) is the exact *extended* joint including dummies — the
        real-site marginal entropy alone is no longer tractable, which is also
        why :meth:`log_prob` is unavailable with ``n_dummy > 0``.
    flow_type : str
        ``"affine"`` (default) uses the RealNVP affine-coupling
        :class:`FlowSampler`; ``"spline"`` uses the more expressive
        rational-quadratic :class:`SplineFlowSampler` (``num_bins`` /
        ``tail_bound`` apply).  Both keep exact entropy, so the maxent
        machinery below is identical.
    num_bins, tail_bound :
        Spline knobs (``flow_type="spline"`` only): bins per transformed dim and
        the ``[-B, B]`` interval outside which the spline is the identity.
    n_components : int
        Base-distribution mixture size.  ``1`` (default) keeps the standard
        ``z ~ N(0, I)`` base.  ``K > 1`` replaces it with a uniform-weight
        Gaussian mixture with learnable per-component means and diagonal
        scales (params leaf ``"base"``).  Multimodal targets then come from
        component placement instead of forcing the invertible map to tear a
        unimodal base apart — the usual source of extreme Jacobians.  Entropy
        stays exact: ``H(x) = -E[log q_mix(z)] + E[log|det J|] + sum log
        span`` with the mixture density in closed form; the component index
        carries no parameters (uniform weights), so the reparameterized
        gradient is unbiased.  :meth:`log_prob` remains exact too.  With
        ``flow_type="spline"``, keep component means inside ``[-tail_bound,
        tail_bound]`` (see ``base_spread``) or they land in the identity
        tails.
    base_spread : float
        Init sd of the mixture means (``n_components > 1`` only).  Component
        log-scales start at 0.
    """

    def __init__(self, vars: Sequence[ContinuousVar], n_layers: int = 8,
                 hidden: int = 64, s_max: float = 3.0, n_dummy: int = 0,
                 flow_type: str = "affine", num_bins: int = 8,
                 tail_bound: float = 4.0, n_components: int = 1,
                 base_spread: float = 1.0):
        self.disc = Discretizer(vars)
        self.n = self.disc.n_sites
        self.dims = self.disc.dims
        self.n_dummy = int(n_dummy)
        self.n_flow = self.n + self.n_dummy
        self.latent_dim = self.n_flow                # invertible: same dim

        if flow_type == "affine":
            self.net = FlowSampler(self.n_flow, n_layers=n_layers,
                                   hidden=hidden, s_max=s_max)
        elif flow_type == "spline":
            self.net = SplineFlowSampler(self.n_flow, n_layers=n_layers,
                                         hidden=hidden, num_bins=num_bins,
                                         tail_bound=tail_bound)
        else:
            raise ValueError(f"flow_type must be 'affine' or 'spline', "
                             f"got {flow_type!r}")
        self.flow_type = flow_type
        lower = np.concatenate([self.disc.lower, np.zeros(self.n_dummy)])
        upper = np.concatenate([self.disc.upper, np.ones(self.n_dummy)])
        self.lower = jnp.asarray(lower, jnp.float32)
        self.span = jnp.asarray(upper - lower, jnp.float32)
        self._gate_pbroken = []
        self.n_components = int(n_components)
        self.base_spread = float(base_spread)

        # H(z) + sum log span: the constant part of H(x).  Dummy spans are 1,
        # so they add nothing here.  (n_components == 1 only; a mixture base
        # has no analytic H(z), so the loss uses -E[log q_mix(z)] instead.)
        self._log_span_sum = float(jnp.sum(jnp.log(self.span)))
        self._h_const = (0.5 * self.n_flow * float(np.log(2.0 * np.pi * np.e))
                         + self._log_span_sum)

    # ---- base distribution (standard normal or Gaussian mixture) ------
    def init_params(self, seed: int = 0):
        params = super().init_params(seed)
        if self.n_components > 1:
            key = jax.random.PRNGKey(seed + 0x5F3759DF)
            mu = self.base_spread * jax.random.normal(
                key, (self.n_components, self.n_flow))
            params["base"] = {
                "mu": mu.astype(jnp.float32),
                "log_sig": jnp.zeros((self.n_components, self.n_flow),
                                     jnp.float32),
            }
        return params

    def _draw_z_key(self, key, n_samples: int):
        """Raw per-sample randomness.  Standard base: ``(N, D)`` normals.
        Mixture base: ``(N, D + 1)`` — D normals plus one uniform column that
        selects the component (kept as raw randomness so ``_base_z`` can
        rebuild ``z`` differentiably from params)."""
        if self.n_components == 1:
            return jax.random.normal(key, (n_samples, self.latent_dim))
        ke, kc = jax.random.split(key)
        eps = jax.random.normal(ke, (n_samples, self.n_flow))
        u = jax.random.uniform(kc, (n_samples, 1))
        return jnp.concatenate([eps, u], axis=1)

    def _draw_z(self, n_samples: int, seed: int):
        return self._draw_z_key(jax.random.PRNGKey(seed), n_samples)

    def _base_z(self, params, z):
        """Raw randomness -> actual latent ``z`` (pathwise-differentiable in
        the mixture params).  Identity for the standard base."""
        if self.n_components == 1:
            return z
        eps, u = z[:, :self.n_flow], z[:, self.n_flow]
        c = jnp.clip((u * self.n_components).astype(jnp.int32),
                     0, self.n_components - 1)
        b = params["base"]
        return b["mu"][c] + jnp.exp(b["log_sig"])[c] * eps

    def _base_log_prob(self, params, zb):
        """Exact base log-density at latents ``zb`` (N, n_flow)."""
        log2pi = float(np.log(2.0 * np.pi))
        if self.n_components == 1:
            return (-0.5 * jnp.sum(zb * zb, axis=1)
                    - 0.5 * self.n_flow * log2pi)
        b = params["base"]
        diff = (zb[:, None, :] - b["mu"][None]) * jnp.exp(-b["log_sig"])[None]
        comp = (-0.5 * jnp.sum(diff * diff, axis=-1)
                - jnp.sum(b["log_sig"], axis=-1)[None]
                - 0.5 * self.n_flow * log2pi)               # (N, K)
        return jax.nn.logsumexp(comp, axis=1) - jnp.log(self.n_components)

    def _wrap_loss(self, body, n_samples):
        def loss(params, key):
            return body(params, self._draw_z_key(key, n_samples))
        return loss

    # ---- sampling with log-det --------------------------------------
    def _sample_x_logdet(self, params, z):
        """Full extended batch ``(N, n_flow)`` — dummy columns included (the
        loss needs the whole invertible image for the entropy term).  ``z`` is
        the *raw* randomness from ``_draw_z``; the mixture base rebuilds the
        actual latent from it first."""
        zb = self._base_z(params, z)
        u, ld = jax.vmap(self.net.forward_flat, in_axes=(None, 0))(
            params["theta"], zb)
        return self.lower + self.span * u, ld

    def _sample_x(self, params, z):
        """Query-facing samples: real sites only, shape ``(N, n)``."""
        x, _ = self._sample_x_logdet(params, z)
        return x[:, :self.n]

    def entropy(self, params, n_samples: int = 20000, seed: int = 0):
        """Exact joint differential entropy ``H(x)`` in nats (MC over z).

        With ``n_dummy > 0`` this is the entropy of the *extended* joint
        (real sites + dummies) — the quantity the loss regularizes; the
        real-site marginal entropy alone is not tractable."""
        z = self._draw_z(n_samples, seed)
        _, ld = self._sample_x_logdet(params, z)
        if self.n_components == 1:
            return float(self._h_const + jnp.mean(ld))
        zb = self._base_z(params, z)
        return float(-jnp.mean(self._base_log_prob(params, zb))
                     + self._log_span_sum + jnp.mean(ld))

    def log_prob(self, params, x, chunk: int = 65536):
        """Exact log density at points ``x`` (N, n) in original units (numpy).

        Change of variables through the inverse flow::

            log p(x) = log N(z(x)) - log|det J_g(z(x))| - sum log span

        The flow is a tractable density model as well as a sampler — this is
        what e.g. held-out NLL evaluation uses.  Unavailable with
        ``n_dummy > 0``: the real-site density is then a marginal of the
        extended joint, which the inverse pass cannot integrate out."""
        if self.n_dummy:
            raise NotImplementedError(
                "log_prob requires n_dummy=0: with dummy dimensions the "
                "real-site density is an intractable marginal")
        x = np.atleast_2d(np.asarray(x, np.float32))
        u = (jnp.asarray(x) - self.lower) / self.span
        inv = jax.vmap(self.net.inverse_flat, in_axes=(None, 0))
        out = []
        for k in range(0, len(x), chunk):
            z, ld = inv(params["theta"], u[k:k + chunk])
            log_n = self._base_log_prob(params, z)
            out.append(np.asarray(log_n - ld))
        return np.concatenate(out) - float(jnp.sum(jnp.log(self.span)))

    # ---- loss: same grammar, exact entropy term ----------------------
    def constraint_loss(self, constraints, entropy_reg: float = 1.0,
                        weight_reg: float = 0.0, n_samples: int = 4096,
                        seed: int = 0, domain_prior: str = "uniform",
                        prior_bound_sds: float = 2.0, ref_mask=None,
                        with_logq: bool = False):
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

        ``with_logq=True`` appends one extra column to the sample matrix seen
        by constraint features: the per-sample ``log q_theta(x)`` evaluated at
        ``stop_gradient(x)`` (one extra inverse pass per loss eval).  Features
        that ignore it (all the standard site-indexed ones) are unaffected;
        features can read ``x[:, -1]`` to build score-function (REINFORCE)
        gradient terms — ``grad E[f] = E[f * grad log q]`` needs the *partial*
        theta-derivative of ``log q`` at fixed samples, which the forward-path
        ``log N(z) - logdet`` does NOT give (its gradient includes the
        transport term); hence the inverse pass.  With ``n_dummy > 0`` the
        column is the extended-joint density — still the correct score weight,
        since the samples are drawn from that joint.

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
            if self.n_dummy:                 # dummies keep the uniform ref
                mask = jnp.concatenate(
                    [mask, jnp.zeros(self.n_dummy, jnp.float32)])
            log_norm = -0.5 * jnp.log(2.0 * jnp.pi * ref_sd ** 2)

        def body(params, z):
            # One forward pass: constraints score the samples x, the entropy
            # term is the mean per-sample log-likelihood on that SAME batch
            # (log p(x_i) = log N(z_i) - logdet_i; the z-density term is
            # theta-independent, so h_const carries its analytic expectation).
            x, ld = self._sample_x_logdet(params, z)
            if with_logq:
                u_bar = jax.lax.stop_gradient((x - self.lower) / self.span)
                z_bar, ld_bar = jax.vmap(
                    self.net.inverse_flat, in_axes=(None, 0))(
                        params["theta"], u_bar)
                lq = (self._base_log_prob(params, z_bar) - ld_bar
                      - self._log_span_sum)
                x_feat = jnp.concatenate([x, lq[:, None]], axis=1)
            else:
                x_feat = x
            tot = 0.0
            for score in scorers:
                tot = tot + score(x_feat)
            if gate_scorers:
                gates = params["gates"]
                for score in gate_scorers:
                    tot = tot + score(x_feat, gates)
            if entropy_reg:
                if self.n_components == 1:
                    ent = h_const + jnp.mean(ld)           # H(p), exact
                else:
                    # mixture base: -E[log q_mix(z)] replaces the analytic
                    # H(z); still exact in expectation, gradients pathwise
                    zb = self._base_z(params, z)
                    ent = (-jnp.mean(self._base_log_prob(params, zb))
                           + self._log_span_sum + jnp.mean(ld))
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
