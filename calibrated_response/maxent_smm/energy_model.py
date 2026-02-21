"""Trained MaxEnt energy model with HMC sampling.

An ``EnergyModel`` wraps the learned feature weights θ, the compiled
energy / gradient functions, and the domain normaliser so that users can:

* evaluate the (unnormalised) log-probability at arbitrary points,
* draw new samples via HMC in either the normalised [0, 1]^D domain or
  the original variable domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from calibrated_response.maxent_smm.mcmc import (
    HMCConfig,
    PersistentBuffer,
    advance_buffer,
    build_hmc_step_fn,
)
from calibrated_response.maxent_smm.normalizer import ContinuousDomainNormalizer
from calibrated_response.models.variable import Variable


class EnergyModel:
    """A trained MaxEnt energy model  p_θ(x) ∝ exp(θ · f(x)).

    Parameters
    ----------
    theta : jnp.ndarray, shape (K,)
        Learned feature weights.
    energy_fn : callable  ``(theta, x) -> scalar``
        JIT-compiled energy function.
    grad_energy_fn : callable  ``(theta, x) -> (D,)``
        JIT-compiled gradient of energy w.r.t. position.
    feature_vector_fn : callable  ``(x,) -> (K,)``
        Compiled feature vector function.
    hmc_step_fn : callable
        JIT-compiled HMC chain-step function (from ``build_hmc_step_fn``).
    normalizer : ContinuousDomainNormalizer
        Maps between original variable domains and [0, 1].
    variables : list of Variable
        The variable definitions (for metadata / names).
    hmc_config : HMCConfig
        Default HMC hyperparameters for sampling.
    """

    def __init__(
        self,
        theta: jnp.ndarray,
        energy_fn: Callable,
        grad_energy_fn: Callable,
        feature_vector_fn: Callable,
        hmc_step_fn: Callable,
        normalizer: ContinuousDomainNormalizer,
        variables: Sequence[Variable],
        hmc_config: Optional[HMCConfig] = None,
    ):
        self.theta = theta
        self._energy_fn = energy_fn
        self._grad_energy_fn = grad_energy_fn
        self._feature_vector_fn = feature_vector_fn
        self._hmc_step_fn = hmc_step_fn
        self.normalizer = normalizer
        self.variables = list(variables)
        self.hmc_config = hmc_config or HMCConfig()
        self._var_names = [v.name for v in self.variables]
        self.n_vars = len(self.variables)

    # ------------------------------------------------------------------
    # Energy / log-probability
    # ------------------------------------------------------------------

    def energy(self, x: np.ndarray | jnp.ndarray) -> float:
        """Evaluate E_θ(x) = -θ·f(x) in the *normalised* [0,1]^D domain.

        Parameters
        ----------
        x : array-like, shape (D,)
            Point in [0, 1]^D.

        Returns
        -------
        float
        """
        return float(self._energy_fn(self.theta, jnp.asarray(x, dtype=jnp.float32)))

    def energy_original(self, x: np.ndarray | dict[str, float]) -> float:
        """Evaluate E_θ at a point in the *original* variable domain.

        Parameters
        ----------
        x : array-like shape (D,)  **or**  dict mapping variable name → value.
        """
        x_norm = self._to_normalized(x)
        return self.energy(x_norm)

    def log_prob(self, x: np.ndarray | jnp.ndarray) -> float:
        """Unnormalised log p_θ(x) = θ·f(x) = -E_θ(x).  Normalised domain."""
        return -self.energy(x)

    def log_prob_original(self, x: np.ndarray | dict[str, float]) -> float:
        """Unnormalised log p_θ(x) in original domain."""
        return -self.energy_original(x)

    def feature_vector(self, x: np.ndarray | jnp.ndarray) -> np.ndarray:
        """Evaluate the feature vector f(x) in normalised domain."""
        return np.asarray(self._feature_vector_fn(jnp.asarray(x, dtype=jnp.float32)))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        *,
        n_chains: int = 128,
        n_warmup: int = 100,
        n_steps_per_draw: int = 5,
        step_size: Optional[float] = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Draw samples via HMC in the normalised [0, 1]^D domain.

        Parameters
        ----------
        n_samples : int
            Desired number of samples (actual count may differ slightly
            depending on ``n_chains``; at most ``n_samples`` are returned).
        n_chains : int
            Number of parallel HMC chains.
        n_warmup : int
            Warm-up iterations (discarded).
        n_steps_per_draw : int
            HMC transitions between successive draws.
        step_size : float or None
            HMC step size (defaults to ``hmc_config.step_size``).
        seed : int
            RNG seed.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, D)
            Samples in [0, 1]^D.
        """
        ss = step_size or self.hmc_config.step_size
        hmc_cfg = HMCConfig(
            step_size=ss,
            num_leapfrog_steps=self.hmc_config.num_leapfrog_steps,
            target_accept_rate=self.hmc_config.target_accept_rate,
            adapt_step_size=self.hmc_config.adapt_step_size,
            adapt_rate=self.hmc_config.adapt_rate,
        )

        rng = jax.random.PRNGKey(seed)
        buffer = PersistentBuffer.initialize(
            n_chains=n_chains,
            rng_key=rng,
            step_size=ss,
            n_vars=self.n_vars,
        )

        # Warm-up
        for _ in range(n_warmup):
            buffer = advance_buffer(
                buffer, self.theta, n_steps=1,
                hmc_config=hmc_cfg, step_fn=self._hmc_step_fn,
            )

        # Collect draws
        n_draws = max(1, (n_samples + n_chains - 1) // n_chains)
        all_states = []
        for _ in range(n_draws):
            buffer = advance_buffer(
                buffer, self.theta, n_steps=n_steps_per_draw,
                hmc_config=hmc_cfg, step_fn=self._hmc_step_fn,
            )
            all_states.append(np.asarray(buffer.states))

        samples = np.concatenate(all_states, axis=0)[:n_samples]
        return samples

    def sample_original(
        self,
        n_samples: int,
        **kwargs,
    ) -> np.ndarray:
        """Draw samples and denormalize to original variable domains.

        Same parameters as ``sample()``.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, D)
            Samples in original domains.
        """
        samples_norm = self.sample(n_samples, **kwargs)
        return self._denormalize_samples(samples_norm)

    def sample_original_dict(
        self,
        n_samples: int,
        **kwargs,
    ) -> list[dict[str, float]]:
        """Draw samples as a list of ``{var_name: value}`` dicts.

        Same parameters as ``sample()``.
        """
        samples = self.sample_original(n_samples, **kwargs)
        return [
            {name: float(row[i]) for i, name in enumerate(self._var_names)}
            for row in samples
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_normalized(self, x) -> jnp.ndarray:
        """Convert original-domain input (array or dict) to [0, 1]^D."""
        if isinstance(x, dict):
            arr = np.array([
                self.normalizer.normalize_value(name, x[name])
                for name in self._var_names
            ], dtype=np.float32)
        else:
            arr = np.asarray(x, dtype=np.float32)
            # Normalise each dimension
            normed = np.array([
                self.normalizer.normalize_value(self._var_names[i], float(arr[i]))
                for i in range(self.n_vars)
            ], dtype=np.float32)
            arr = normed
        return jnp.asarray(arr)

    def _denormalize_samples(self, samples: np.ndarray) -> np.ndarray:
        """Denormalise (N, D) samples from [0, 1] back to original domains."""
        out = np.empty_like(samples)
        for i, name in enumerate(self._var_names):
            out[:, i] = np.array([
                self.normalizer.denormalize_value(name, float(v))
                for v in samples[:, i]
            ])
        return out
