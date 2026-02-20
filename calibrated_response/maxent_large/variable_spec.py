"""Declarative variable specifications for the MaxEnt energy model.

Each variable type is a frozen dataclass that declaratively describes a function
f_k(x) → scalar.  variables are compiled to pure JAX functions via
``compile_variable`` / ``compile_variable_vector`` so they can be JIT-compiled and
differentiated by JAX for use in HMC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Union, Literal

import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class GaussianPriorSpec:
    """Specification for a Gaussian prior on a continuous variable."""
    prior_type: Literal["gaussian"] = "gaussian"
    mean: float = 0.5
    std: float = 0.5

@dataclass(frozen=True)
class UniformPriorSpec:
    """Specification for a uniform prior on a variable."""
    prior_type: Literal["uniform"] = "uniform"

@dataclass(frozen=True)
class BetaPriorSpec:
    """Specification for a Beta prior on a continuous variable in [0, 1].

    The Beta(alpha, beta) distribution concentrates mass near 0 when alpha < beta,
    near 1 when alpha > beta, and at the center when alpha == beta.  Both
    parameters must be positive.
    """
    prior_type: Literal["beta"] = "beta"
    alpha: float = 2.0
    beta: float = 2.0

prior_spec = Union[GaussianPriorSpec, UniformPriorSpec, BetaPriorSpec]

@dataclass(frozen=True)
class VariableSpec:
    """Specification for a variable in the MaxEnt model."""
    name: str
    description: str
    type: Literal["binary", "continuous", "discrete", "ordinal"]
    prior: prior_spec = field(default_factory=UniformPriorSpec)
