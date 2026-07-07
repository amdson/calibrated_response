# Smoothness Regularization on Tensor Chains and Trees

Penalize roughness of the represented function in a single continuous coordinate $x_i$, discretized on a grid of $d$ points with spacing $h$. The penalty approximates $\gamma \int (\partial_x^2 f_i)^2\, dx$ and is quadratic in one core, so it drops into an ALS/DMRG sweep as a Tikhonov term.

## Ingredients

Second-difference matrix $D \in \mathbb{R}^{(d-2)\times d}$:

$$
D_{k,k}=1,\quad D_{k,k+1}=-2,\quad D_{k,k+2}=1.
$$

Roughness metric $L = D^\top D \in \mathbb{R}^{d\times d}$: symmetric PSD, pentadiagonal, interior stencil $(1,-4,6,-4,1)$, boundary-corrected diagonal

$$
\mathrm{diag}(L) = (1,\,5,\,6,\,6,\dots,6,\,6,\,5,\,1).
$$

Null space $\ker L = \mathrm{span}\{\mathbf{1},\,(1,2,\dots,d)\}$: constant and linear ramps along the grid cost zero, only curvature is penalized. (For a slope penalty instead, use first differences; $L$ becomes the tridiagonal path Laplacian, $\ker L = \{\mathbf 1\}$.)

## Where to apply it

The penalty on the *represented function* is $R_i = \gamma\, \langle p | L_i | p\rangle$, a two-copy contraction with $L$ inserted at site $i$. It collapses to a single-core expression **only when that core is the orthogonality center** (all other cores isometric toward it). Off-center, the isometric environments contract to identity; on-center, nothing else survives.

- **Chain (MPS):** center at site $i$ means sites $1..i{-}1$ left-canonical, $i{+}1..N$ right-canonical.
- **Tree (TTN):** center at the node holding $x_i$; every other node isometric toward the center along its unique tree path, i.e. flattening with the toward-center edge as columns gives $T_{e_0}^\dagger T_{e_0} = I$.

## Formula

Flatten the center core with the physical index as rows: $\tilde A \in \mathbb{R}^{d \times M}$, where $M = \prod_{\text{bonds}} \chi$ over all bonds incident to the center node.

- Chain, interior site: $M = \chi_{i-1}\chi_i$.
- Chain, boundary site / TTN leaf: $M = \chi$.
- TTN internal node of degree $z$: $M = \prod_{k=1}^{z}\chi_{e_k}$.

Matrix form:

$$
R_i = \frac{\gamma}{h^3}\, \mathrm{tr}\!\big(\tilde A^\top L\, \tilde A\big).
$$

Componentwise (bond multi-index $\alpha$):

$$
R_i = \frac{\gamma}{h^3} \sum_{\alpha}\ \sum_{k=1}^{d-2}\Big( A[v_k]_\alpha - 2 A[v_{k+1}]_\alpha + A[v_{k+2}]_\alpha \Big)^2.
$$

The two cases differ only in how many components $\alpha$ has.

## Spacing factor

$h^{-3}$ is the only place $h$ enters: $(f_{k-1}-2f_k+f_{k+1})/h^2 \approx f''$, and $\int (f'')^2 dx \approx h\sum_k (f'')^2$ gives $h \cdot h^{-4} = h^{-3}$. If $h$ is uniform across the chain/tree and only relative strength matters, fold it into $\gamma$. Keep it explicit only when grids differ between sites, else fine grids are over-penalized. (Per-site: weight $\gamma\, h_i^{-3}$.)

## ALS drop-in

With $c = \mathrm{vec}(\tilde A)$, the local normal equations $Nc = b$ from the environment gain one additive term:

$$
\Big(N + \frac{\gamma}{h^3}\,(I_M \otimes L)\Big)\, c = b.
$$

Generalized Tikhonov with roughness metric $L$. Since $L \succeq 0$, it only improves conditioning of the local solve. Per sweep: solve at the center, QR to move the center to the next node, repeat. To smooth several coordinates, each is penalized when its own node is the center during the sweep.

## Caveats

- **Born parameterization ($p = \psi^2$):** the exact curvature of $p$ is quartic in the cores and breaks the quadratic solve. Penalize the curvature of $\psi$ instead; each local subproblem stays Tikhonov and $p'' = 2(\psi'^2 + \psi\psi'')$ is still controlled.
- **Scope:** this suppresses high-frequency ripple in each coordinate. It does not raise global entropy — a sharply peaked $p$ and a uniform $p$ are both smooth by this metric. Entropy pressure is a separate $-\beta H_2$ term.