"""JIT-compiled Hamiltonian Monte Carlo (HMC) with persistent chains.

All chains live in a continuous [0, 1]^n_vars state space.  Boundary handling
uses reflection so that samples never leave the unit hypercube.

Energy and gradient functions take ``(theta, x)`` so that JAX traces them once
with abstract shapes and reuses the compiled XLA kernel as ``theta`` values
change across training iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

from calibrated_response.models.variable import Variable


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HMCConfig:
    """Configuration for the HMC kernel."""
    step_size: float = 0.01
    num_leapfrog_steps: int = 10
    target_accept_rate: float = 0.65
    adapt_step_size: bool = True
    adapt_rate: float = 0.05          # multiplicative adaptation factor


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def _reflect_with_momentum(
    q: jnp.ndarray, p: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reflect position into [0, 1] **and** negate momentum for reflected dims.

    Without the momentum flip, a particle that crosses a boundary keeps its
    momentum pointing into the wall, causing samples to pile up at 0 and 1.

    Handles single boundary crossings correctly.  For excursions > 1.0
    (rare with reasonable step sizes, max 0.5), position is still valid
    but momentum parity may be wrong.
    """
    q = jnp.mod(q, 2.0)
    reflected = (q > 1.0)

    q = jnp.where(reflected, 2.0 - q, q)
    p = jnp.where(reflected, -p, p)
    return q, p


# ---------------------------------------------------------------------------
# Leapfrog integrator (JIT-compatible via lax.fori_loop)
# ---------------------------------------------------------------------------

def _leapfrog(
    grad_energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    theta: jnp.ndarray,
    q: jnp.ndarray,
    p: jnp.ndarray,
    step_size: float,
    num_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Leapfrog integrator with reflection at [0, 1] boundaries.

    Parameters
    ----------
    grad_energy_fn : callable  ``(theta, x) -> grad_x``
        Gradient of the energy w.r.t. position.
    theta : jnp.ndarray, shape (K,)
        Current feature weights (passed through, not closed over).
    q : jnp.ndarray, shape (D,)
        Initial position.
    p : jnp.ndarray, shape (D,)
        Initial momentum.
    step_size : float
    num_steps : int

    Returns
    -------
    q, p : updated position and momentum.
    """
    eps = step_size

    # Half-step for momentum
    p = p - 0.5 * eps * grad_energy_fn(theta, q)

    # First position step
    q = q + eps * p
    q, p = _reflect_with_momentum(q, p)

    def _inner_step(i, carry):
        q, p = carry
        grad = grad_energy_fn(theta, q)
        p = p - eps * grad
        q = q + eps * p
        q, p = _reflect_with_momentum(q, p)
        return (q, p)

    q, p = lax.fori_loop(0, num_steps - 1, _inner_step, (q, p))

    # Final half-step for momentum
    p = p - 0.5 * eps * grad_energy_fn(theta, q)

    # Negate momentum for reversibility
    p = -p
    return q, p


# ---------------------------------------------------------------------------
# Single HMC transition
# ---------------------------------------------------------------------------

def _hmc_step(
    energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    grad_energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    theta: jnp.ndarray,
    state: jnp.ndarray,
    rng_key: jnp.ndarray,
    step_size: float,
    num_leapfrog: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One HMC transition (propose + accept/reject).

    Returns (new_state, accepted) where accepted is 0. or 1.
    """
    key_mom, key_accept = jax.random.split(rng_key)

    # Sample momentum
    p0 = jax.random.normal(key_mom, shape=state.shape)

    # Current Hamiltonian
    current_E = energy_fn(theta, state)
    current_K = 0.5 * jnp.sum(p0 ** 2)

    # Leapfrog
    q_prop, p_prop = _leapfrog(grad_energy_fn, theta, state, p0, step_size, num_leapfrog)

    # Proposed Hamiltonian
    proposed_E = energy_fn(theta, q_prop)
    proposed_K = 0.5 * jnp.sum(p_prop ** 2)

    # Metropolis accept/reject
    log_alpha = (current_E + current_K) - (proposed_E + proposed_K)
    log_u = jnp.log(jax.random.uniform(key_accept))
    accepted = (log_u < log_alpha).astype(jnp.float32)

    new_state = jnp.where(accepted, q_prop, state)
    return new_state, accepted


# ---------------------------------------------------------------------------
# Vectorized chain step  (vmap over chains)
# ---------------------------------------------------------------------------

def _hmc_chain_step(
    energy_fn: Callable,
    grad_energy_fn: Callable,
    theta: jnp.ndarray,
    states: jnp.ndarray,
    rng_keys: jnp.ndarray,
    step_size: float,
    num_leapfrog: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Advance C chains in parallel via ``vmap``.

    Parameters
    ----------
    theta : (K,) feature weights — explicit arg so JIT traces once.
    states : (C, D)
    rng_keys : (C, 2)

    Returns
    -------
    new_states : (C, D)
    accepted : (C,)
    """
    def _single(state, key):
        return _hmc_step(energy_fn, grad_energy_fn, theta, state, key,
                         step_size, num_leapfrog)

    return jax.vmap(_single)(states, rng_keys)


# ---------------------------------------------------------------------------
# Persistent buffer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersistentBuffer:
    """Immutable container for persistent HMC chains.

    All states live in [0, 1]^n_vars (normalized domain).
    """
    states: jnp.ndarray       # (C, D)
    rng_key: jnp.ndarray
    step_size: float
    accept_rate: float = 0.0  # running acceptance rate

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @staticmethod
    def initialize(
        n_chains: int,
        rng_key: jnp.ndarray,
        step_size: float = 0.01,
        *,
        n_vars: Optional[int] = None,
        var_specs: Optional[Sequence[Variable]] = None,
    ) -> PersistentBuffer:
        """Create a buffer with uniform [0, 1] initial states.

        Provide either ``n_vars`` (int) or ``var_specs`` (sequence of
        Variable objects).  All variables are treated as continuous in
        [0, 1] for HMC purposes.
        """
        if var_specs is not None:
            n = len(var_specs)
        elif n_vars is not None:
            n = n_vars
        else:
            raise ValueError("Must provide either n_vars or var_specs")
        key_init, key_next = jax.random.split(rng_key)
        states = jax.random.uniform(key_init, shape=(n_chains, n))
        return PersistentBuffer(states=states, rng_key=key_next, step_size=step_size)
    
def build_hmc_step_fn(energy_fn, grad_energy_fn) -> Callable:
    """Build a chain step function with energy and gradient closed over."""
    def step_fn(theta, states, rng_keys, step_size, num_leapfrog):
        return _hmc_chain_step(energy_fn, grad_energy_fn, theta, states, rng_keys,
                               step_size, num_leapfrog)

    return jax.jit(step_fn)
    
def advance_buffer(buffer: PersistentBuffer,
            theta: jnp.ndarray,
            n_steps: int,
            hmc_config: Optional[HMCConfig],
            step_fn: Callable) -> PersistentBuffer:
    """Advance the buffer by running HMC steps."""
    config = hmc_config or HMCConfig()
    step_size = buffer.step_size
    states = buffer.states
    rng_key = buffer.rng_key
    total_accepted = jnp.zeros(())
    for _ in range(n_steps):
        rng_key, step_key = jax.random.split(rng_key)
        n_chains = states.shape[0]
        chain_keys = jax.random.split(step_key, n_chains)

        new_states, accepted = step_fn(theta, states, chain_keys,
            step_size, config.num_leapfrog_steps,
        )
        states = new_states
        total_accepted = total_accepted + accepted.mean()

    avg_accept = total_accepted / max(n_steps, 1)

    # Simple step-size adaptation
    if config.adapt_step_size:
        if avg_accept > config.target_accept_rate:
            step_size = step_size * (1.0 + config.adapt_rate)
        else:
            step_size = step_size * (1.0 - config.adapt_rate)
        # Clamp to reasonable range
        step_size = float(jnp.clip(step_size, 1e-6, 0.5))

    return PersistentBuffer(
        states=states,
        rng_key=rng_key,
        step_size=step_size,
        accept_rate=float(avg_accept),
    )
    
    # # ------------------------------------------------------------------
    # # Advance
    # # ------------------------------------------------------------------
    # def advance(
    #     self,
    #     energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    #     grad_energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    #     theta: jnp.ndarray,
    #     n_steps: int = 1,
    #     hmc_config: Optional[HMCConfig] = None,
    # ) -> "PersistentBuffer":
    #     """Run ``n_steps`` HMC transitions on all chains.

    #     Parameters
    #     ----------
    #     energy_fn, grad_energy_fn : callables with signature ``(theta, x)``
    #     theta : current feature weights

    #     Returns a **new** ``PersistentBuffer`` (immutable update).
    #     """
    #     config = hmc_config or HMCConfig()
    #     step_size = self.step_size
    #     states = self.states
    #     rng_key = self.rng_key
    #     total_accepted = jnp.zeros(())

    #     for _ in range(n_steps):
    #         rng_key, step_key = jax.random.split(rng_key)
    #         n_chains = states.shape[0]
    #         chain_keys = jax.random.split(step_key, n_chains)

    #         new_states, accepted = _hmc_chain_step(
    #             energy_fn, grad_energy_fn, theta, states, chain_keys,
    #             step_size, config.num_leapfrog_steps,
    #         )
    #         states = new_states
    #         total_accepted = total_accepted + accepted.mean()

    #     avg_accept = total_accepted / max(n_steps, 1)

    #     # Simple step-size adaptation
    #     if config.adapt_step_size:
    #         if avg_accept > config.target_accept_rate:
    #             step_size = step_size * (1.0 + config.adapt_rate)
    #         else:
    #             step_size = step_size * (1.0 - config.adapt_rate)
    #         # Clamp to reasonable range
    #         step_size = float(jnp.clip(step_size, 1e-6, 0.5))

    #     return PersistentBuffer(
    #         states=states,
    #         rng_key=rng_key,
    #         step_size=step_size,
    #         accept_rate=float(avg_accept),
    #     )
