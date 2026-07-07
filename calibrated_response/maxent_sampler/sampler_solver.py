"""Implicit-sampler MaxEnt solver via direct moment matching.

Fits a sampler ``x = g_θ(z)``, ``z ~ N(0, I)`` so that the model moments match
targets::

    J(θ) = (1/2) || E_z[f(g_θ(z))] - μ ||²   (+ optional regulariser)

Unlike the energy-model solver (``maxent_smm.MaxEntSolver``), there is no HMC and
no covariance/score-function estimator.  Because ``z`` does not depend on ``θ``,
``J`` is differentiated straight through the Monte-Carlo sample average with
``jax.grad`` (the reparametrisation trick), giving a low-variance, unbiased
gradient at a fraction of the cost.

Trade-off: a free sampler has no built-in maximum-entropy property.  Supply a
``reg_fn(theta, z, x) -> scalar`` (e.g. ``neg_gaussian_entropy_proxy``) and set
``entropy_reg_weight > 0`` to push the distribution towards higher entropy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from calibrated_response.maxent_smm.variable_spec import VariableSpec
from calibrated_response.maxent_smm.features import FeatureSpec, compile_feature_vector


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SamplerSolverConfig:
    """Configuration for the implicit-sampler MaxEnt solver."""

    # Monte-Carlo / training loop
    num_samples: int = 512          # latent draws per gradient step (batch size)
    num_iterations: int = 500
    learning_rate: float = 0.01
    l2_regularization: float = 0.0  # L2 on sampler params θ
    grad_clip: float = 5.0
    tolerance: float = 1e-4         # stop when max moment error < tolerance
    verbose: bool = False

    # Entropy / maxent regulariser.  When entropy_reg_weight != 0 and a reg_fn is
    # supplied to build(), the loss becomes  J + entropy_reg_weight * reg_fn(...).
    # With `neg_gaussian_entropy_proxy` a positive weight encourages spread.
    entropy_reg_weight: float = 0.0

    # If True, draw fresh latents every iteration (unbiased SGD-style gradient,
    # recommended).  If False, reuse one fixed latent batch for all iterations
    # (deterministic objective, but biased toward that particular sample).
    resample_each_iter: bool = True

    seed: int = 0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SamplerSolver:
    """Fits an implicit sampler to moment targets by reparametrised autodiff.

    Usage::

        solver = SamplerSolver(config)
        solver.build(var_specs, feature_specs, feature_targets,
                     sampler_fn=sampler.sample_fn_flat,
                     init_theta=theta, latent_dim=sampler.latent_dim)
        theta, info = solver.solve()
    """

    def __init__(self, config: Optional[SamplerSolverConfig] = None):
        self.config = config or SamplerSolverConfig()

        # Populated by build()
        self._feature_vector_fn: Optional[Callable] = None
        self._sampler_fn: Optional[Callable] = None
        self._loss_and_grad_fn: Optional[Callable] = None
        self._theta = None
        self._feature_targets: Optional[jnp.ndarray] = None
        self._latent_dim: int = 0
        self._reg_fn: Optional[Callable] = None
        self._opt_state: Optional[optax.OptState] = None
        self._optimizer: Optional[optax.GradientTransformation] = None
        self._n_features: int = 0
        self._var_specs: Optional[Sequence[VariableSpec]] = None
        self._n_vars: int = 0
        self._rng = None
        self._built = False

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(
        self,
        var_specs: Sequence[VariableSpec],
        feature_specs: Sequence[FeatureSpec],
        feature_targets: jnp.ndarray | np.ndarray,
        sampler_fn: Callable,
        init_theta: jnp.ndarray,
        latent_dim: int,
        reg_fn: Optional[Callable] = None,
    ) -> None:
        """Compile the JAX loss/gradient and initialise solver state.

        Parameters
        ----------
        var_specs : sequence of VariableSpec
            The variables in the joint distribution (defines n_vars).
        feature_specs : sequence of FeatureSpec
            Declarative feature definitions f_k(x).
        feature_targets : array-like, shape (K,)
            Target expectations μ_k for each feature.
        sampler_fn : callable ``(theta, z) -> x``
            Maps a flat parameter vector ``theta`` and a single latent ``z``
            (shape ``(latent_dim,)``) to a state ``x`` (shape ``(n_vars,)``).
            Must be differentiable in ``theta``.
        init_theta : jnp.ndarray, shape (n_params,)
            Initial flat sampler parameters.
        latent_dim : int
            Dimensionality of the Gaussian latent z.
        reg_fn : callable ``(theta, z, x) -> scalar``, optional
            Extra penalty added to the loss, scaled by
            ``config.entropy_reg_weight``.  ``x`` is the full sample batch
            ``(num_samples, n_vars)``.  See ``neg_gaussian_entropy_proxy``.
        """
        cfg = self.config
        self._var_specs = var_specs
        self._n_vars = len(var_specs)
        self._n_features = len(feature_specs)
        self._feature_targets = jnp.asarray(feature_targets, dtype=jnp.float32)
        self._sampler_fn = sampler_fn
        self._latent_dim = latent_dim
        self._reg_fn = reg_fn
        self._theta = init_theta

        # Feature vector function: x -> (K,)
        self._feature_vector_fn = compile_feature_vector(feature_specs)

        self._compile_loss()

        # Optax optimiser (minimises the loss directly).
        self._optimizer = optax.adam(cfg.learning_rate)
        self._opt_state = self._optimizer.init(self._theta)

        self._rng = jax.random.PRNGKey(cfg.seed)
        self._built = True

    def _compile_loss(self) -> None:
        """Build and JIT the moment-matching loss and its gradient in θ."""
        cfg = self.config
        sampler_fn = self._sampler_fn
        feature_vector_fn = self._feature_vector_fn
        reg_fn = self._reg_fn
        reg_weight = cfg.entropy_reg_weight

        def loss_fn(theta, z, targets):
            # z: (N, latent_dim) -> x: (N, n_vars)
            x = jax.vmap(sampler_fn, in_axes=(None, 0))(theta, z)
            feats = jax.vmap(feature_vector_fn)(x)          # (N, K)
            model_exp = feats.mean(axis=0)                  # (K,)
            delta = model_exp - targets
            moment_loss = 0.5 * jnp.sum(delta ** 2)

            reg = jnp.asarray(0.0)
            if reg_fn is not None and reg_weight != 0.0:
                reg = reg_fn(theta, z, x)
            loss = moment_loss + reg_weight * reg
            # aux carries diagnostics out of the value_and_grad call.
            return loss, (model_exp, moment_loss, reg)

        self._loss_and_grad_fn = jax.jit(
            jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        )

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[jnp.ndarray, dict]:
        """Run the training loop.

        Returns
        -------
        theta : jnp.ndarray, shape (n_params,)
            Learned sampler parameters.
        info : dict
            Diagnostics including ``history`` and a batch of final samples.
        """
        if not self._built:
            raise RuntimeError("Call build() before solve().")

        cfg = self.config
        targets = self._feature_targets
        theta = self._theta
        opt_state = self._opt_state
        optimizer = self._optimizer
        rng = self._rng

        history = {
            "iteration": [],
            "smm_loss": [],
            "moment_loss": [],
            "reg": [],
            "max_error": [],
            "mean_error": [],
            "mean_squared_error": [],
            "runtime_sec": [],
        }

        # Fixed latent batch when resampling is disabled.
        z_fixed = None
        if not cfg.resample_each_iter:
            rng, sub = jax.random.split(rng)
            z_fixed = jax.random.normal(sub, (cfg.num_samples, self._latent_dim))

        model_expectations = None
        it = 0
        for it in range(1, cfg.num_iterations + 1):
            t0 = time.time()

            if cfg.resample_each_iter:
                rng, sub = jax.random.split(rng)
                z = jax.random.normal(sub, (cfg.num_samples, self._latent_dim))
            else:
                z = z_fixed

            (loss, (model_expectations, moment_loss, reg)), grad = (
                self._loss_and_grad_fn(theta, z, targets)
            )

            if cfg.l2_regularization > 0.0:
                grad = grad + cfg.l2_regularization * theta
            grad = jnp.clip(grad, -cfg.grad_clip, cfg.grad_clip)

            updates, opt_state = optimizer.update(grad, opt_state, theta)
            theta = optax.apply_updates(theta, updates)

            # --- Diagnostics ---
            delta = targets - model_expectations
            err = jnp.abs(delta)
            max_err = float(err.max())
            mean_err = float(err.mean())
            mean_squared_err = float((err ** 2).mean())
            smm_loss = float(0.5 * jnp.sum(delta ** 2))

            history["iteration"].append(it)
            history["smm_loss"].append(smm_loss)
            history["moment_loss"].append(float(moment_loss))
            history["reg"].append(float(reg))
            history["max_error"].append(max_err)
            history["mean_error"].append(mean_err)
            history["mean_squared_error"].append(mean_squared_err)
            history["runtime_sec"].append(time.time() - t0)

            if cfg.verbose and it % 50 == 0:
                print(
                    f"[SamplerSolver] iter {it:4d}  "
                    f"max_err={max_err:.6f}  mean_err={mean_err:.6f}  "
                    f"mse={mean_squared_err:.6f}  reg={float(reg):.4f}"
                )

            if max_err < cfg.tolerance:
                if cfg.verbose:
                    print(f"[SamplerSolver] Converged at iteration {it}.")
                break

        # Store final state.
        self._theta = theta
        self._opt_state = opt_state
        self._rng = rng

        # Draw a fresh batch of samples from the trained model for inspection.
        rng, sub = jax.random.split(rng)
        z_final = jax.random.normal(sub, (cfg.num_samples, self._latent_dim))
        final_samples = jax.vmap(self._sampler_fn, in_axes=(None, 0))(theta, z_final)

        info = {
            "history": history,
            "samples": np.asarray(final_samples),        # (num_samples, n_vars)
            "theta": np.asarray(theta),
            "feature_targets": np.asarray(targets),
            "final_model_expectations": np.asarray(model_expectations),
            "n_iterations": it,
            "converged": bool(history["max_error"][-1] < cfg.tolerance) if it else False,
        }
        return theta, info

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def sample(self, n: int, key: Optional[jax.Array] = None) -> jnp.ndarray:
        """Draw ``n`` samples from the current sampler.  Shape ``(n, n_vars)``."""
        if not self._built:
            raise RuntimeError("Call build() before sample().")
        if key is None:
            self._rng, key = jax.random.split(self._rng)
        z = jax.random.normal(key, (n, self._latent_dim))
        return jax.vmap(self._sampler_fn, in_axes=(None, 0))(self._theta, z)
