# Scalable Maximum Entropy Model Training Blueprint

This document describes an implementation-ready procedure for training a scalable maximum entropy (MaxEnt) or energy-based model using persistent MCMC. This approach supports thousands of variables, arbitrary feature constraints, and marginal priors.

---

# 0. Core Model Definition

Define a set of feature functions:

    f_k(x),    k = 1, ..., K

Each feature may represent:

- Moment constraint (e.g. X * Y)
- Indicator bin: 1(X ∈ B_i, Y ∈ B_j)
- Polynomial or threshold feature
- Basis expansion feature
- Arbitrary structured feature

The model is an exponential family distribution:

    p_theta(x) = (1 / Z(theta)) * exp( sum_k theta_k * f_k(x) )

This defines an energy-based model or factor graph.

Energy function:

    E_theta(x) = - sum_k theta_k * f_k(x)

Partition function:

    Z(theta) = sum_x exp(sum_k theta_k * f_k(x))

---

# 1. Objective Function

Use the maximum entropy dual objective with optional marginal priors:

    L(theta) =
        theta^T * m_target
        - log Z(theta)
        - sum_{S in P} lambda_S * Phi_S(p_theta(x_S))
        - R(theta)

Where:

- m_target: target feature expectations
- Phi_S: marginal prior penalty (KL, L2, etc.)
- R(theta): parameter regularization (L2 recommended)
- P: set of marginals with priors
- lambda_S: prior strengths

If penalties are convex, the objective is convex.

---

# 2. Persistent MCMC Chains

Maintain C persistent chains:

    x^(1), ..., x^(C)  ~  p_theta(x)

These chains evolve gradually as theta changes.

Use Gibbs sampling or Metropolis-Hastings updates.

Persistent chains avoid repeated burn-in and dramatically improve efficiency.

---

# 3. Maintain Feature Values for Chains

Store feature values for each chain:

    f_k^(c) = f_k(x^(c))

Maintain running sums:

    S_k = sum_{c=1..C} f_k^(c)

This allows efficient expectation estimation.

Update incrementally when chains change.

---

# 4. Training Iteration

Each iteration consists of four steps.

---

## Step A: Advance MCMC Chains

For each chain:

- Select variable(s) to update
- Perform Gibbs or Metropolis-Hastings update
- Recompute affected feature values
- Update running feature sums

This maintains approximate samples from p_theta.

---

## Step B: Estimate Model Expectations

Estimate expectations using sample means:

    mu_hat_k = S_k / C

This approximates:

    E_theta[f_k(X)]

Also estimate marginals if needed for priors.

---

## Step C: Compute Gradient

Base moment-matching gradient:

    g_k = m_target_k - mu_hat_k

Add marginal prior gradient:

    g_k -= sum_S lambda_S * dPhi_S / dtheta_k

Add regularization gradient:

    g_k -= dR(theta) / dtheta_k

All terms computable from samples.

---

## Step D: Update Parameters

Update parameters using optimizer (SGD, Adam, etc.):

    theta <- theta + eta * g

---

# 5. Repeat Until Convergence

Monitor:

- Feature moment mismatch: |m_target - mu_hat|
- Marginal prior penalties
- Parameter stability

Stop when stable.

---

# 6. Computational Complexity

Each iteration costs approximately:

    O(number_of_MCMC_updates × features_per_update)

Not exponential in number of variables.

Scales to thousands of variables and features.

---

# 7. Marginal Priors Integration

Marginal priors require only:

- Estimating marginals from chains
- Computing additional gradient terms

No partition function computation required.

No architectural changes needed.

---

# 8. Exact Marginal Matching Special Case

If exact marginal matching is desired:

- Use indicator features for desired marginals
- Use moment matching objective only
- No penalty term needed

The model will match those marginals at convergence.

---

# 9. Data Structures

Recommended structure:

theta:
    shape: (K,)

features:
    scopes: list of variable indices per feature
    type: feature type identifier
    params: feature-specific parameters

chains:
    X:
        shape: (C, n_variables)
    feature_values:
        shape: (C, K)

running_sum:
    S:
        shape: (K,)

adjacency_map:
    variable_index -> list of affected feature indices

This enables efficient incremental updates.

---

# 10. Why This Works

This performs stochastic gradient ascent on:

    theta^T * m_target - log Z(theta) - priors

without computing the partition function.

Gradient identity:

    d/dtheta log Z(theta) = E_theta[f(X)]

So gradient is:

    m_target - E_theta[f(X)]

Samples provide expectation estimates.

This converges to the maximum entropy distribution satisfying constraints.

---

# 11. Conceptual Interpretation

The learned distribution is the unique maximum entropy distribution satisfying:

- Feature expectation constraints
- Marginal priors
- Structural assumptions encoded in features

This is the correct probabilistic solution.

---

# 12. One-Sentence Summary

Training consists of maintaining persistent MCMC samples, estimating expectations from them, computing the maximum entropy gradient, and updating parameters until convergence.
