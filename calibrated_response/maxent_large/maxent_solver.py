"""Core MaxEnt solver using persistent HMC and optax optimisation.

The solver learns weights θ for an energy-based model::

    p_θ(x) = (1/Z(θ)) exp( Σ_k θ_k f_k(x) )

by matching feature expectations via stochastic gradient ascent on the dual
objective, using persistent HMC chains to estimate model expectations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from calibrated_response.models.variable import Variable
from calibrated_response.maxent_large.features import FeatureSpec, compile_feature_vector
from calibrated_response.maxent_large.mcmc import (HMCConfig, PersistentBuffer, 
                                                    build_hmc_step_fn, advance_buffer)


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

    # HMC kernel
    hmc_step_size: float = 0.01
    hmc_leapfrog_steps: int = 10
    adapt_step_size: bool = True
    target_accept_rate: float = 0.65

    prior: str = "uniform"  # "uniform" or "gaussian"
    prior_std: float = 0.5   # only used if prior="gaussian"

    # Misc
    seed: int = 0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class MaxEntSolver:
    """JAX-based multivariate maximum entropy solver using persistent HMC.

    Usage::

        solver = MaxEntSolver(config)
        solver.build(n_vars, feature_specs, feature_targets)
        theta, info = solver.solve()
    """

    def __init__(self, config: Optional[JAXSolverConfig] = None):
        self.config = config or JAXSolverConfig()

        # Populated by build()
        self._feature_vector_fn: Optional[Callable] = None
        self._energy_fn: Optional[Callable] = None
        self._grad_energy_fn: Optional[Callable] = None
        self._theta: Optional[jnp.ndarray] = None
        self._feature_targets: Optional[jnp.ndarray] = None
        self._buffer: Optional[PersistentBuffer] = None
        self._opt_state: Optional[optax.OptState] = None
        self._optimizer: Optional[optax.GradientTransformation] = None
        self._n_features: int = 0
        self._var_specs: Optional[Sequence[Variable]] = None
        self._n_vars: int = 0
        self._hmc_config: Optional[HMCConfig] = None
        self._built = False

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(
        self,
        var_specs: Sequence[Variable],
        feature_specs: Sequence[FeatureSpec],
        feature_targets: jnp.ndarray | np.ndarray,
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
        """
        cfg = self.config
        self._var_specs = var_specs
        self._n_vars = len(var_specs)
        self._n_features = len(feature_specs)
        self._feature_targets = jnp.asarray(feature_targets, dtype=jnp.float32)

        # Compile feature vector function: x → (K,)
        self._feature_vector_fn = compile_feature_vector(feature_specs)

        # Batch feature function: (C, D) → (C, K)
        self._batch_feature_fn = jax.jit(jax.vmap(self._feature_vector_fn))

        # Energy and gradient as functions of (theta, x) — traced once,
        # never recompiled as theta values change across iterations.
        fv_fn = self._feature_vector_fn

        # self.is_continuous = jnp.array([var.type == "continuous" for var in feature_specs], dtype=bool)

        def _energy(theta, x):
            return jnp.dot(theta, fv_fn(x))
        if cfg.prior == "gaussian":
            def _energy(theta, x):
                return jnp.dot(theta, fv_fn(x)) + jnp.sum((x - 0.5) ** 2 / (2 * cfg.prior_std ** 2))

        self._energy_fn = jax.jit(_energy)
        self._grad_energy_fn = jax.jit(jax.grad(_energy, argnums=1))

        # Initial parameters
        self._theta = jnp.zeros(self._n_features, dtype=jnp.float32)

        # HMC config
        self._hmc_config = HMCConfig(
            step_size=cfg.hmc_step_size,
            num_leapfrog_steps=cfg.hmc_leapfrog_steps,
            target_accept_rate=cfg.target_accept_rate,
            adapt_step_size=cfg.adapt_step_size,
        )

        hmc_step_fn =  build_hmc_step_fn(self._energy_fn, self._grad_energy_fn)
        self._hmc_step_fn = hmc_step_fn

        # Persistent buffer
        rng = jax.random.PRNGKey(cfg.seed)
        self._buffer = PersistentBuffer.initialize(
            n_chains=cfg.num_chains,
            var_specs=self._var_specs,
            rng_key=rng,
            step_size=cfg.hmc_step_size,
        )
        
        # Optax optimiser (we *negate* gradients since optax minimises)
        self._optimizer = optax.adam(cfg.learning_rate)
        self._opt_state = self._optimizer.init(self._theta)

        self._built = True

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

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
            "max_error": [],
            "mean_error": [],
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
            # vmap feature_vector over chains: (C, D) → (C, K)
            chain_features = self._batch_feature_fn(buffer.states)    # (C, K)
            model_expectations = chain_features.mean(axis=0)          # (K,)

            # --- C: Compute gradient ---
            grad = targets - model_expectations + cfg.l2_regularization * theta
            grad = jnp.clip(grad, -cfg.grad_clip, cfg.grad_clip)

            # --- D: Update parameters via optax ---
            # optax minimises; our gradient points in the ascent direction,
            # so we negate it to turn maximise → minimise.
            updates, opt_state = optimizer.update(grad, opt_state, theta)
            theta = optax.apply_updates(theta, updates)

            # --- Diagnostics ---
            err = jnp.abs(targets - model_expectations)
            max_err = float(err.max())
            mean_err = float(err.mean())

            history["iteration"].append(it)
            history["max_error"].append(max_err)
            history["mean_error"].append(mean_err)
            history["accept_rate"].append(float(buffer.accept_rate))
            history["step_size"].append(float(buffer.step_size))
            history["runtime_sec"].append(time.time() - t0)

            if cfg.verbose and it % 50 == 0:
                print(
                    f"[MaxEntSolver] iter {it:4d}  "
                    f"max_err={max_err:.6f}  mean_err={mean_err:.6f}  "
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
        }
        return theta, info