"""Core MaxEnt-SMM solver using persistent HMC and optax optimisation.

The solver learns weights θ for an energy-based model::

    p_θ(x) = (1/Z(θ)) exp( -E_θ(x) )

by minimising the stochastic moment-matching (SMM) loss::

    J(θ) = (1/2) || E_θ[f(x)] - μ ||²

using persistent HMC chains to draw samples and estimate the gradient via the
efficient covariance estimator (no explicit covariance matrix formed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from calibrated_response.maxent_smm.variable_spec import GaussianPriorSpec, BetaPriorSpec, VariableSpec
from calibrated_response.models.variable import Variable, VariableType
from calibrated_response.maxent_smm.features import FeatureSpec, compile_feature_vector
from calibrated_response.maxent_smm.mcmc import (HMCConfig, PersistentBuffer,
                                                  build_hmc_step_fn, 
                                                  build_batch_hmc_chain_step_fn,
                                                  advance_buffer)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JAXSolverConfig:
    """Configuration for the HMC-based MaxEnt solver."""

    # Histogram / discretisation
    max_bins: int = 10

    # Training loop
    num_chains: int = 128
    num_iterations: int = 300
    mcmc_steps_per_iteration: int = 5
    learning_rate: float = 0.01
    l2_regularization: float = 1e-4
    grad_clip: float = 5.0
    tolerance: float = 1e-4
    verbose: bool = False

    mean_initialisation: bool = True  # whether to initialise theta with mean-matching guess
    continuous_prior: str = "gaussian"  # "gaussian", "beta", or "uniform" (default)

    # HMC kernel
    hmc_step_size: float = 0.01
    hmc_leapfrog_steps: int = 10
    adapt_step_size: bool = True
    target_accept_rate: float = 0.65

    # prior: str = "uniform"  # "uniform" or "gaussian"
    # prior_std: float = 0.5   # only used if prior="gaussian"
    indicator_sharpness: float = 10.0   # sharpness for sigmoid surrogates in features

    # Roughness penalty: adds roughness_gamma * theta @ R @ theta to the dual objective,
    # where R_{ij} = E[nabla f_i . nabla f_j].  Disabled when roughness_gamma == 0.
    roughness_gamma: float = 0.0
    roughness_n_samples: int = 10_000   # MC samples used to estimate R

    # Misc
    seed: int = 0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class MaxEntSolver:
    """JAX-based multivariate MaxEnt-SMM solver using persistent HMC.

    Minimises the moment-matching loss J(θ) = (1/2)||E_θ[f(x)] - μ||² via
    gradient descent, where gradients are estimated from persistent HMC chains
    using the efficient covariance estimator::

        w_i        = -dot(delta, f(x_i) - g_hat)   # scalar weight per sample
        ∇J ≈ (1/N) Σ_i  w_i · ∇_θ E(θ, x_i)       # uses energy param-gradient

    Usage::

        solver = MaxEntSolver(config)
        solver.build(var_specs, feature_specs, feature_targets)
        theta, info = solver.solve()
    """

    def __init__(self, config: Optional[JAXSolverConfig] = None):
        self.config = config or JAXSolverConfig()

        # Populated by build()
        self._feature_vector_fn: Optional[Callable] = None
        self._energy_fn: Optional[Callable] = None
        self._grad_energy_fn: Optional[Callable] = None
        self._batch_grad_theta_fn: Optional[Callable] = None
        self._theta: Optional[jnp.ndarray] = None
        self._feature_targets: Optional[jnp.ndarray] = None
        self._buffer: Optional[PersistentBuffer] = None
        self._opt_state: Optional[optax.OptState] = None
        self._optimizer: Optional[optax.GradientTransformation] = None
        self._n_features: int = 0
        self._var_specs: Optional[Sequence[VariableSpec]] = None
        self._n_vars: int = 0
        self._hmc_config: Optional[HMCConfig] = None
        self._R: Optional[jnp.ndarray] = None   # roughness matrix, populated by build()
        self._built = False

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(
        self,
        var_specs: Sequence[VariableSpec],
        feature_specs: Sequence[FeatureSpec],
        feature_targets: jnp.ndarray | np.ndarray,
        energy_fn: Callable,
        init_theta: jnp.ndarray,
    ) -> None:
        """Compile JAX programs and initialise solver state.

        Parameters
        ----------
        var_specs : sequence of Variable
            The variables in the joint distribution.
        feature_specs : sequence of FeatureSpec
            Declarative feature definitions.
        feature_targets : array-like, shape (K,)
            Target expectations for each feature.
        energy_fn : callable
            Energy function E(θ, x) → scalar.
        """
        cfg = self.config
        self._var_specs = var_specs
        self._n_vars = len(var_specs)
        self._n_features = len(feature_specs)
        self._feature_targets = jnp.asarray(feature_targets, dtype=jnp.float32)
        self._param_energy_fn = energy_fn

        # Compile feature vector function: x → (K,)
        self._feature_vector_fn = compile_feature_vector(feature_specs)

        # Batch feature function: (C, D) → (C, K)
        self._batch_feature_fn = jax.jit(jax.vmap(self._feature_vector_fn))

        # Energy and gradient as functions of (theta, x) — traced once,
        # never recompiled as theta values change across iterations.
        fv_fn = self._feature_vector_fn

        self.prior_mean = jnp.array([var.prior.mean if isinstance(var.prior, GaussianPriorSpec) else 0.0 for var in var_specs], dtype=jnp.float32)
        self.prior_std = jnp.array([var.prior.std if isinstance(var.prior, GaussianPriorSpec) else 1.0 for var in var_specs], dtype=jnp.float32)
        self.is_gaussian = jnp.array([1.0 if isinstance(var.prior, GaussianPriorSpec) else 0.0 for var in var_specs], dtype=jnp.float32)
        self.prior_alpha = jnp.array([var.prior.alpha if isinstance(var.prior, BetaPriorSpec) else 1.0 for var in var_specs], dtype=jnp.float32)
        self.prior_beta = jnp.array([var.prior.beta if isinstance(var.prior, BetaPriorSpec) else 1.0 for var in var_specs], dtype=jnp.float32)
        self.is_beta = jnp.array([1.0 if isinstance(var.prior, BetaPriorSpec) else 0.0 for var in var_specs], dtype=jnp.float32)

        # Per-variable prior mean used only for mean-matching theta initialisation.
        prior_init_means = jnp.array([
            var.prior.mean if isinstance(var.prior, GaussianPriorSpec) else
            var.prior.alpha / (var.prior.alpha + var.prior.beta) if isinstance(var.prior, BetaPriorSpec) else
            0.0
            for var in var_specs
        ], dtype=jnp.float32)

        # print(f"Prior means: {self.prior_mean}")
        # print(f"Prior stds: {self.prior_std}")
        # print(f"Is Gaussian: {self.is_gaussian}")
        # print(f"Prior alpha: {self.prior_alpha}")
        # print(f"Prior beta: {self.prior_beta}")
        # print(f"Is Beta: {self.is_beta}")

        # def _energy(theta, x):
        #     return jnp.dot(theta, fv_fn(x))
        
        # if cfg.prior == "gaussian":
        def _energy(theta, x):
            gaussian_energy = (x - self.prior_mean) ** 2 / (2 * self.prior_std ** 2) - jnp.log(self.prior_std * jnp.sqrt(2 * jnp.pi))
            beta_energy = (-(self.prior_alpha - 1) * jnp.log(x)
                           - (self.prior_beta - 1) * jnp.log(1 - x)
                           - jax.scipy.special.betaln(self.prior_alpha, self.prior_beta))
            prior_energy = jnp.sum(
                self.is_gaussian * gaussian_energy +
                self.is_beta * beta_energy
                # uniform contributes zero; no term needed
            )
            # energy = jnp.dot(theta, fv_fn(x)) + prior_energy
            energy = self._param_energy_fn(theta, x) + prior_energy
            return energy

        self._energy_fn = _energy
        self._grad_energy_fn = jax.grad(_energy, argnums=1)

        # Gradient of energy w.r.t. θ (not x), batched over chain states.
        # Used by the SMM estimator: ∇J ≈ (1/N) Σ_i w_i · ∇_θ E(θ, x_i)
        _grad_theta_fn = jax.grad(_energy, argnums=0)          # (θ, x) → (K,)
        self._batch_grad_theta_fn = jax.vmap(_grad_theta_fn, in_axes=(None, 0))         # (θ, (C,D)) → (C, K)
        
        self.compile_grad() # compile the SMM gradient estimator separately (since it doesn't depend on theta values and can be traced once)

        # Initial parameters
        # self._theta = jnp.zeros(self._n_features, dtype=jnp.float32)
        self._theta = init_theta
        
        # HMC config
        self._hmc_config = HMCConfig(
            step_size=cfg.hmc_step_size,
            num_leapfrog_steps=cfg.hmc_leapfrog_steps,
            target_accept_rate=cfg.target_accept_rate,
            adapt_step_size=cfg.adapt_step_size,
        )

        hmc_step_fn =  build_hmc_step_fn(self._energy_fn, self._grad_energy_fn, num_leapfrog=cfg.hmc_leapfrog_steps)
        # hmc_step_fn = build_batch_hmc_chain_step_fn(self._energy_fn, self._grad_energy_fn)
        self._hmc_step_fn = hmc_step_fn

        # Persistent buffer
        rng = jax.random.PRNGKey(cfg.seed)
        self._buffer = PersistentBuffer.initialize(
            n_chains=cfg.num_chains,
            var_specs=self._var_specs,
            rng_key=rng,
            step_size=cfg.hmc_step_size,
        )
        
        # Roughness matrix  R_{ij} = E[∇f_i · ∇f_j]  (estimated by MC)
        if cfg.roughness_gamma > 0.0:
            from calibrated_response.maxent_smm.estimate_r import estimate_R
            rng_R = jax.random.PRNGKey(cfg.seed + 1)
            if cfg.verbose:
                print(f"[MaxEntSolver] Estimating roughness matrix R "
                      f"({cfg.roughness_n_samples} samples)...")
            self._R = estimate_R(
                self._feature_vector_fn,
                self._n_vars,
                n_samples=cfg.roughness_n_samples,
                key=rng_R,
            )
        else:
            self._R = None

        # Optax optimiser (we *negate* gradients since optax minimises)
        self._optimizer = optax.adam(cfg.learning_rate)
        self._opt_state = self._optimizer.init(self._theta)

        self._built = True

    def compile_grad(self):
        #incomplete function to compile the gradient estimator separately if needed

        #batch estimator of gradient of MSE of model expectations vs targets
        def smm_potential(theta, states, targets):
            sample_features = jax.vmap(self._feature_vector_fn)(states) #feature_fn(states)    # (C, K)
            model_expectations = sample_features.mean(axis=0)          # (K,)
            delta = model_expectations - targets               # (p,)  E_θ[f] - μ
            centered = sample_features - model_expectations        # (C, p) centred features
            sample_energy = jax.vmap(self._energy_fn, in_axes=(None, 0))(theta, states)  # (C,)
            mean_energy = sample_energy.mean()  # scalar
            delta_energy = sample_energy - mean_energy  # (C,)
            sample_potential = - jnp.einsum('j, ij, i -> ', 
                                             delta,
                                             centered,
                                             delta_energy) / states.shape[0]  # (C,)
            return sample_potential
        smm_grad_fn = jax.grad(smm_potential, argnums=0)
        self._smm_grad_fn = jax.jit(smm_grad_fn)
        self._smm_potential_fn = smm_potential
        return self._smm_grad_fn, self._smm_potential_fn
                                           
    def solve(self) -> Tuple[jnp.ndarray, dict]:
        """Run the training loop.

        Returns
        -------
        theta : jnp.ndarray, shape (K,)
            Learned feature weights.
        info : dict
            Diagnostics including ``history``, ``chain_states``, etc.
        """
        if not self._built:
            raise RuntimeError("Call build() before solve().")

        cfg = self.config
        targets = self._feature_targets
        theta = self._theta
        buffer = self._buffer
        opt_state = self._opt_state
        optimizer = self._optimizer
        hmc_cfg = self._hmc_config

        history = {
            "iteration": [],
            "smm_loss": [],
            "max_error": [],
            "mean_error": [],
            "mean_squared_error": [],
            "accept_rate": [],
            "step_size": [],
            "runtime_sec": [],
        }



        for it in range(1, cfg.num_iterations + 1):
            t0 = time.time()

            buffer = advance_buffer(
                buffer=buffer,
                theta=theta,
                n_steps=cfg.mcmc_steps_per_iteration,
                hmc_config=hmc_cfg,
                step_fn=self._hmc_step_fn,
            )

            # --- B: Estimate model expectations ---
            # # vmap feature_vector over chains: (C, D) → (C, K)
            chain_features = self._batch_feature_fn(buffer.states)    # (C, K)
            model_expectations = chain_features.mean(axis=0)          # (K,)

            # # --- C: Compute SMM gradient ---
            # # ∇_θ E(θ, x_i) at each chain state: (C, K)
            # chain_grad_theta = self._batch_grad_theta_fn(theta, buffer.states)

            # # Efficient covariance estimator (O(N(p+K)), no matrix formed):
            # #   w_i = -dot(delta, f(x_i) - g_hat)
            # #   ∇J ≈ (1/N) Σ_i w_i · ∇_θ E(θ, x_i)
            # delta    = model_expectations - targets               # (p,)  E_θ[f] - μ
            # centered = chain_features - model_expectations        # (C, p) centred features
            # w        = -(centered @ delta)                        # (C,)  scalar weight per chain
            # grad     = (w[:, None] * chain_grad_theta).mean(axis=0)   # (K,)  ≈ ∇_θ J

            grad = self._smm_grad_fn(theta, buffer.states, targets)
            grad     = grad + cfg.l2_regularization * theta       # L2 regularisation
            if self._R is not None:
                grad = grad + cfg.roughness_gamma * (self._R @ theta)
            grad = jnp.clip(grad, -cfg.grad_clip, cfg.grad_clip)

            # --- D: Update parameters via optax ---
            # grad is ∇_θ J (uphill on J); optax minimises by subtracting lr * grad.
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
            history["max_error"].append(max_err)
            history["mean_error"].append(mean_err)
            history["mean_squared_error"].append(mean_squared_err)
            history["accept_rate"].append(float(buffer.accept_rate))
            history["step_size"].append(float(buffer.step_size))
            history["runtime_sec"].append(time.time() - t0)

            if cfg.verbose and it % 50 == 0:
                print(
                    f"[MaxEntSolver] iter {it:4d}  "
                    f"max_err={max_err:.6f}  mean_err={mean_err:.6f}  mean_squared_err={mean_squared_err:.6f}  "
                    f"accept={buffer.accept_rate:.3f}  step_size={buffer.step_size:.5f}"
                )

            if max_err < cfg.tolerance:
                if cfg.verbose:
                    print(f"[MaxEntSolver] Converged at iteration {it}.")
                break

        # Store final state
        self._theta = theta
        self._buffer = buffer
        self._opt_state = opt_state

        info = {
            "history": history,
            "chain_states": np.asarray(buffer.states),    # (C, D) in [0,1]
            "theta": np.asarray(theta),
            "feature_targets": np.asarray(targets),
            "final_model_expectations": np.asarray(model_expectations),
            "n_iterations": it,
            "converged": max_err < cfg.tolerance,
            "roughness_matrix_used": self._R is not None,
        }
        return theta, info