# maxent_pgmax notes

This package is a pgmax-based alternative to the existing `maxent` implementation.

## What is implemented

- Continuous and binary variables are discretized into finite buckets.
- Query estimates are converted into soft unary and pairwise factor preferences.
- Inference uses loopy belief propagation through `pgmax.infer.build_inferer(..., backend="bp")`.

## What is intentionally skipped

- Laplacian smoothing regularization from the JAX solver is not implemented as a direct pgmax factor.
- Exact global MaxEnt matching over all higher-order constraints is not implemented.
- Multi-condition conditional probabilities and conditional expectations are not fully supported.

## Practical alternatives

- For smoothness, increase bucket count moderately and apply post-hoc smoothing to marginals.
- For stability, use stronger unary priors and smaller conditional strengths.
- If exact calibration is required, combine pgmax marginals with an outer optimization loop (iterative scaling) or switch to a differentiable objective over full log-potentials.
