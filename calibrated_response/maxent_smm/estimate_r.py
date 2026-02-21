"""Estimate the roughness matrix R for the augmented MaxEnt dual objective.

The roughness matrix is defined as:

    R_{ij} = int_{[0,1]^d} (nabla f_i(x)) . (nabla f_j(x)) dx

and enters the augmented dual objective as a smoothing penalty:

    loss = log_Z(theta) - theta @ mu + roughness_gamma * theta @ R @ theta

Minimising this loss encourages solutions where neighboring features have
correlated weights, producing smoother estimated distributions.

The matrix is estimated by Monte Carlo sampling uniformly over [0, 1]^d,
using the Jacobian of the compiled feature vector function so that all
feature gradients are computed in a single vectorised pass.  Samples are
processed in mini-batches to keep peak memory proportional to
``batch_size * n_features * d`` rather than ``n_samples * n_features * d``.
"""

import jax
import jax.numpy as jnp
from typing import Callable


def estimate_R(
    feature_vector_fn: Callable,
    d: int,
    n_samples: int = 10_000,
    batch_size: int = 1_000,
    key: jnp.ndarray = jax.random.PRNGKey(0),
) -> jnp.ndarray:
    """Estimate the roughness matrix R by Monte Carlo integration.

    Parameters
    ----------
    feature_vector_fn : callable  ``x -> (n_features,)``
        The compiled feature vector function, compatible with ``jax.jacobian``.
        This is ``MaxEntSolver._feature_vector_fn`` after calling ``build()``.
    d : int
        Number of variables (dimensionality of ``x``).
    n_samples : int
        Total number of uniform samples drawn from ``[0, 1]^d``.
    batch_size : int
        Samples processed per JIT-compiled call.  Trades memory for
        (negligible) Python-loop overhead.  Defaults to 1 000.
    key : jax.random.PRNGKey
        Random key for reproducibility.

    Returns
    -------
    R : jnp.ndarray, shape (n_features, n_features)
        Estimated roughness (Gram) matrix.  Symmetric positive semi-definite.
    """
    # J_fn(x) -> (n_features, d): Jacobian of the feature vector w.r.t. x.
    J_fn = jax.jacobian(feature_vector_fn)

    @jax.jit
    def batch_gram(X_batch: jnp.ndarray) -> jnp.ndarray:
        """Compute the mean Gram matrix over a batch.

        Parameters
        ----------
        X_batch : (B, d)

        Returns
        -------
        mean_gram : (n_features, n_features)
        """
        def gram(x):
            J = J_fn(x)    # (n_features, d)
            return J @ J.T  # (n_features, n_features)

        return jax.vmap(gram)(X_batch).mean(axis=0)

    # Infer n_features from a single forward pass.
    n_features = int(feature_vector_fn(jnp.zeros(d)).shape[0])

    X = jax.random.uniform(key, shape=(n_samples, d))

    # Accumulate a weighted sum over mini-batches.
    R_acc = jnp.zeros((n_features, n_features))
    total = 0
    for start in range(0, n_samples, batch_size):
        X_batch = X[start : start + batch_size]
        n_batch = int(X_batch.shape[0])
        R_acc = R_acc + batch_gram(X_batch) * n_batch
        total += n_batch

    return R_acc / total
