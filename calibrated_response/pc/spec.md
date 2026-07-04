# Requirements: Constraint-Trained Tensorized Region-Graph SPN

A specification for hand-rolling a probabilistic circuit over ~1000 variables, trained to
match a set of marginal constraints (with no known a-priori structure) and regularized to
stay well-behaved. Targets a static circuit structure and custom gradient-based losses, in
PyTorch or JAX.

---

## 1. Goal and scope

**Goal.** Learn a distribution $p_\theta(X)$ over $X = (X_1,\dots,X_n)$, $n \approx 1000$, such
that for a given set of constraints $\{q_\alpha(x_{S_\alpha})\}_{\alpha}$ (target marginals over
variable subsets $S_\alpha \subseteq \{1,\dots,n\}$), the model marginals
$p_\theta(x_{S_\alpha})$ match the targets, while remaining a "reasonable" distribution where the
constraints leave it underdetermined.

**Primary deliverables.**
- A fixed (non-dynamic) tractable circuit whose marginals over arbitrary subsets are exact and
  cheap.
- A training loop minimizing a marginal-matching loss plus regularizers, by gradient descent
  (not EM).
- Custom inference passes the standard libraries do not expose: subset marginals, projected
  moments/cumulants of $a^\top X$, and (optionally) the projected characteristic function.

**Non-goals.**
- Structure learning, or any data- or iteration-dependent structure mutation. Structure is built
  once, before training, and frozen.
- Exact entropy regularization (requires determinism; see §6 and §10).
- MAP/MPE inference (requires determinism).

---

## 2. Model definition

The circuit is a smooth, decomposable probabilistic circuit (an SPN). Three node types, each
computing a function evaluated bottom-up; the root returns $p_\theta(x)$.

- **Input (leaf)** over a single variable $X_v$: a univariate density/mass $f(x_v)$.
  - Continuous: Gaussian, parameters $(\mu, \log\sigma^2)$.
  - Discrete: Categorical, parameters = logits over the domain.
- **Sum** (mixture): $\sum_c w_c\, p_c$, weights $w_c \ge 0$, $\sum_c w_c = 1$.
- **Product** (factorization): $\prod_c p_c$ over children with disjoint scopes.

**Structural invariants (must hold by construction, assert in tests):**
- *Smoothness*: all children of a sum node share the same scope.
- *Decomposability*: children of a product node have pairwise-disjoint scopes.

These two suffice for linear-time marginalization, which is the whole point. The model is
deliberately **not** deterministic and **not** structured-decomposable across repetitions — that
is acceptable for every query in §5 and is the reason exact-entropy regularization is excluded.

**Normalization.** Parametrize each sum node's weights on the simplex (softmax of free logits) and
each leaf as a normalized density. With locally normalized weights and normalized leaves, the
circuit is globally normalized: marginalizing out all variables yields $Z = 1$ exactly. This is a
unit test (§8), not something to enforce with a loss.

---

## 3. Architecture: tensorized region graph (RAT-SPN form)

The structure is a region graph: a recursive partition of the variable set, tensorized so each
layer is a single batched `einsum`. This is the standard answer to "unknown dependency structure":
random/heuristic partitions, hedged by width and repetitions, all fixed once chosen.

### 3.1 Region graph

- A **region** is a subset of variables (a scope). The **root region** is all $n$ variables.
- A **partition** of a region splits it into disjoint child regions (use binary, balanced splits).
- Recurse until a region has $\le \ell$ variables (**leaf threshold**, e.g. $\ell = 1$–$4$).
  Depth $\approx \log_2(n/\ell) \approx 8$ for $n=1000$.
- Build $R$ independent partition hierarchies (**repetitions**) over the same leaves; mix them at
  the root with one sum node of $R$ weights.

### 3.2 Tensorization

- Each region carries a value **vector of length $C$** (the **width**: $C$ sum nodes per region).
- A leaf region of size 1 carries $K$ input distributions for its variable.
- A binary partition combining child value-vectors $u, v \in \mathbb{R}^C$ into the parent's
  $\mathbb{R}^C$ is one **einsum layer**:
  $$\text{out}[p] \;=\; \sum_{i,j} W[p,i,j]\, u[i]\, v[j], \qquad W \ge 0,\ \textstyle\sum_{i,j} W[p,i,j] = 1.$$
  This fuses the product (outer combination over $i,j$) and the sum (contraction with $W$) into one
  operation. In log-space (§7) it is a `logsumexp` over the $(i,j)$ grid.

### 3.3 Sizing

- Parameters per einsum layer $\approx C^3$; reducible to $\sim C^2$ via the EiNet mixing-layer
  factorization (split the $C\times C\times C$ tensor into a product step + a $C\times C$ mixing
  step). Use the factorization if memory is tight.
- Internal partitions per repetition $\approx n/\ell \approx 250$; total layers $\approx R \cdot n/\ell$.
- Order-of-magnitude parameter counts at $n=1000$: $C=16, R=16 \Rightarrow \sim 10^7$;
  $C=32 \Rightarrow \sim 10^8$. Both fit a single modern GPU.

### 3.4 Defaults (starting point)

| Symbol | Meaning | Default |
|---|---|---|
| $C$ | width (sum nodes/region) | 16–32 |
| $R$ | repetitions | 8–16 |
| $K$ | input distributions/variable | 4–8 |
| $\ell$ | leaf threshold | 1–4 |

Scale $C$ first if constraints are underfit.

---

## 4. Structure construction (one-shot, from constraints only)

You know which variables co-occur in each constraint, even if not the pattern. That defines a
**constraint hypergraph** $H = (V, E)$ with one hyperedge $S_\alpha$ per constraint. Use it to
choose the partitions — still fully static, computed once before training.

**Requirement: a constraint $q_\alpha(x_{S_\alpha})$ is cheap to fit exactly only when $S_\alpha$
stays within a shared scope deep in the hierarchy.** If a partition splits $S_\alpha$ near the
root, representing that marginal demands large width. So partitions should keep co-constrained
variables together.

**Construction (cut-based repetition):**
1. Build $H$ from the constraint scopes.
2. Recursive balanced min-cut bisection of $H$ (METIS-style): at each region, split to cut as few
   hyperedges as possible while keeping the split balanced. This yields one partition hierarchy.
3. Use this as one repetition; add $R-1$ **random** balanced partitions as repetitions to hedge
   dependencies the cut misses.


## 5. Required inference passes

Each region is a tensor of shape `(batch, C)` over a known scope. Every pass below is a bottom-up
traversal reusing the same layer weights; all are differentiable in $\theta$ and $O(\text{layers}\cdot C^2)$.

### 5.1 Log-likelihood / forward pass (required)
- Leaf: log-density at the evidence.
- Product step: `out_log[i,j] = uL_log[i] + vR_log[j]` (outer sum).
- Sum step: `out_log[p] = logsumexp_{i,j}(log W[p,i,j] + out_log[i,j])`.
- Root: mix repetitions. Returns $\log p_\theta(x)$.

### 5.2 Subset marginals (required)
To compute $p_\theta(x_S)$, marginalizing out $\bar S$: set the leaf log-value to $0$ ($\log 1$)
for every variable in $\bar S$ (its normalized density integrates to 1), evaluate at evidence for
variables in $S$, run §5.1 unchanged. Conditionals are two marginal passes.

For a discrete constraint over $S_\alpha$, the full marginal **table** costs
$\prod_{v\in S_\alpha}|\mathrm{dom}(v)|$ forward passes (one per cell). Tractable only for small
$|S_\alpha|$; this bounds the constraint order you can match exactly.

### 5.3 Projected moments of $Y = a^\top X$ (required for the regularizer)
Carry per-component mean and variance vectors, shape `(C,)`, per region:
- **Leaf** over $X_v$, component $d$: $m_1[d] = a_v\,\mathbb{E}_d[X_v]$,
  $\mathrm{var}[d] = a_v^2\,\mathrm{Var}_d[X_v]$.
- **Product step** (independent sub-scopes, $Y = Y_L + Y_R$): means add, variances add:
  $m_1[i,j] = m_1^L[i] + m_1^R[j]$, $\mathrm{var}[i,j] = \mathrm{var}^L[i] + \mathrm{var}^R[j]$.
- **Sum step** (mixture — variance does **not** add): mix *raw* moments,
  $m_1[p] = \sum_{i,j} W[p,i,j]\,m_1[i,j]$,
  $m_2[p] = \sum_{i,j} W[p,i,j]\,(\mathrm{var}[i,j] + m_1[i,j]^2)$,
  then $\mathrm{var}[p] = m_2[p] - m_1[p]^2$.
- **Root**: mixture over repetitions, same rule as the sum step.

Output: exact $\mathbb{E}[a^\top X]$ and $\mathrm{Var}(a^\top X)$. For higher cumulants, add
cumulants at product steps and mix raw moments at sum steps (convert between the two as you go).

### 5.4 Projected characteristic function (optional — full projected density)
Carry a complex value per component at frequency $\omega$:
- Leaf: $\varphi[d] = \mathbb{E}_d[e^{i\omega a_v X_v}]$ (leaf's MGF at $i\omega a_v$; closed form
  for Gaussian/categorical).
- Product step: $\varphi[i,j] = \varphi^L[i]\cdot\varphi^R[j]$ (independent ⇒ multiply).
- Sum step: $\varphi[p] = \sum_{i,j} W[p,i,j]\,\varphi[i,j]$ (mixture ⇒ linear).
Evaluate on an $\omega$-grid (one pass each), inverse-FFT to a density. Only needed if you want the
full distribution of a projection rather than its moments.

---

## 6. Training objective

$$
L(\theta) = \underbrace{\sum_\alpha D\!\big(q_\alpha \,\|\, p_\theta(x_{S_\alpha})\big)}_{\text{constraint matching}}
\;+\; \lambda_{\text{dir}} R_{\text{dir}}(\theta)
\;+\; \lambda_{\text{proj}} R_{\text{proj}}(\theta)
\;+\; \lambda_{\text{kl}} R_{\text{kl}}(\theta)
$$

- **Constraint matching**: $D$ = KL or squared error between target and model marginal (§5.2).
  Overlapping constraints must be mutually consistent (agree on shared sub-marginals); if not, the
  loss returns a compromise.
- **Dirichlet weight prior** $R_{\text{dir}} = -\sum_{\text{sum nodes}} (\alpha_0-1)\sum_c \log w_c$,
  $\alpha_0 > 1$, pushes sum weights toward uniform (anti-degeneracy). Cheapest useful regularizer;
  enable first.
- **Projection isotropy** $R_{\text{proj}} = \mathbb{E}_{a}\big[(\mathrm{Var}(a^\top X)/\|a\|^2 - 1)^2\big]$,
  estimated with a small batch of random directions $a$ per step via §5.3. Guards against
  collapse without imposing a Gaussianizing pressure; use a two-sided target (penalize both
  under- and over-concentration). Each direction's gradient is rank-1 in $\Sigma$, so use several
  directions per step to reduce variance.
- **KL to a reference** $R_{\text{kl}} = D_{\text{KL}}(p_\theta \,\|\, r)$ or cross-entropy, with
  $r$ a fixed, simple reference (e.g. fully-factorized product of per-variable marginals). Shrinks
  the underdetermined part toward a sane default. Tractable when $r$ is structured-decomposable and
  deterministic relative to $p_\theta$; a factorized $r$ satisfies this trivially.

**Excluded**: exact entropy regularization — needs a deterministic circuit (the log-of-circuit
operation is tractable only under determinism; for a non-deterministic SPN, exact entropy is
\#P-hard). The isotropy + Dirichlet + KL-to-reference combination covers "reasonable under the
constraints" without it.

**Optimizer**: Adam-style gradient descent. Non-convex (mixture nonidentifiability); use multiple
restarts / careful init if matching quality varies.

---

## 7. Numerical requirements

- **Log-space** for the likelihood pass; sum layers use `logsumexp`, never raw `exp`/`sum`.
- **Simplex via softmax**: store sum-node weights as free logits, softmax at use. Keeps weights
  normalized and the model globally normalized automatically.
- **Leaf parametrization**: Gaussian as $(\mu, \log\sigma^2)$ (unconstrained); categorical as logits.
- **Moment/CF passes** run in linear space (not log); guard against catastrophic cancellation in
  $m_2 - m_1^2$ (clamp variance at $\ge 0$).
- Keep `float32`; switch to `float64` only if normalization/gradient tests show drift.

---

## 8. Validation requirements

Implement these as automated tests on a small instance ($n \approx 8$–$12$, small $C, R$) where
brute force is possible:

1. **Normalization**: marginalize all variables ⇒ $\log Z = 0$ within tolerance ($10^{-5}$).
2. **Marginal consistency**: a computed marginal $p(x_S)$ sums/integrates to 1; marginalizing a
   variable out of $p(x_S)$ matches $p(x_{S\setminus v})$ computed directly.
3. **Brute-force marginal check**: for small $n$, compare $p_\theta(x_S)$ from §5.2 against
   enumeration/quadrature of the full joint.
4. **Moment-pass check**: draw samples from the circuit (ancestral sampling), compare empirical
   $\mathbb{E}[a^\top X]$, $\mathrm{Var}(a^\top X)$ against §5.3 for random $a$.
5. **Gradient check**: finite-difference vs autograd on a handful of parameters.
6. **Structure invariants**: assert smoothness and decomposability after construction.

---

## 9. Engineering requirements

- **Framework**: plain PyTorch or JAX. Rationale for hand-rolling rather than using PyJuice: the
  passes in §5.3–5.4 are not standard library operations and need direct access to per-region value
  tensors; a transparent tensorized implementation exposes every intermediate, where a compiled
  library does not. Port to PyJuice only if you hit a scale wall the hand-rolled version cannot
  clear.
- **Representation**: store the region graph as layers grouped by depth so each depth is one batched
  `einsum`; index child→parent via gather/scatter on precomputed integer maps. No Python-level
  per-node loops in the hot path.
- **Static shapes**: structure frozen ⇒ all tensor shapes known at build time (important for JAX
  `jit`).
- **Sampling**: ancestral — at each sum node pick a child by its weights, recurse, sample leaves.
  Needed for §8 test 4 and for any downstream generative use.

---

## 10. Open questions / to verify before committing

- **Constraint order vs. exactness**: the per-constraint marginal-table cost (§5.2) grows with
  $|S_\alpha|$ for discrete variables. Confirm your constraints are low-order, or budget for
  approximate matching of high-order ones.
- **Representational feasibility**: a fixed decomposable structure encodes specific independencies.
  If the constraints come from a joint whose higher-order dependencies conflict with the chosen
  partitions, the loss can still go down on the specified marginals while the joint is far from the
  source. Wider $C$ and more repetitions $R$ mitigate; verify by holding out some constraints.
- **Whether this exact recipe (marginal-constraint fitting + projection regularization on a PC) has
  a published precedent** — not confirmed. Worth a literature check before claiming novelty.
- **Library region-graph APIs**: if you do lean on EiNet/PyJuice for the structure, confirm the
  region-graph constructor accepts a user-supplied partition (for the §4 min-cut variant) rather
  than only random/grid splits. Not verified here.