# Cached marginals on a tensor tree — the contractions, in Einstein notation

A map of what `all_site_marginals`, `all_edge_marginals`, and `subtree_marginal`
actually compute, so the code (`tree.py`) reads as notation, not magic. Everything
here is the **nonneg** kind (a tree-structured graphical model, exact sum-product).

## 0. The object

A tree with nodes `i = 0..n-1`. Each node carries one **physical** index `x_i`
(the discretised bin, dimension `d_i`) and one **bond** index per tree neighbour
(dimension `r`). The node's core is a tensor

$$T^{(i)}_{x_i,\; b_{i n_1},\, b_{i n_2},\, \dots}$$

with one bond leg `b_{ij}` shared with each neighbour `j ∈ N(i)`. (In code the
physical axis is first: `core.shape == (d_i, r, r, …)`, one `r` per neighbour, in
`self.adj[i]` order.)

The unnormalised density is the full contraction — every bond summed along its edge,
every physical index kept:

$$\psi(x_0,\dots,x_{n-1}) \;=\; \sum_{\{b\}} \prod_{i} T^{(i)}_{x_i,\,\{b_{i\cdot}\}},
\qquad p(x) = \psi(x)/Z,\quad Z=\sum_x \psi(x).$$

A **marginal** over a set `S` sums out every physical index *not* in `S`:
`p(x_S) = (1/Z) Σ_{x∉S} ψ`. Doing that naïvely is one contraction of the whole tree
per query — `O(n)` each. Belief propagation computes the pieces once and reuses them.

Convention below: repeated bond indices are summed (Einstein); an index left on the
left-hand side is **open** (kept as an output axis). Every contraction in the code
also divides by `stop_grad(max|·|)` — a positive constant that cancels under the final
normalisation, so it's exact and just keeps float32 in range. I omit it from the math.

## 1. Messages (belief propagation)

The **message** from `i` to a neighbour `j` is the contraction of `i`'s *entire
subtree on the far side of the `i–j` edge*, leaving the shared bond `b_{ij}` open:

$$m_{i\to j}[b_{ij}] \;=\; \sum_{x_i}\; \sum_{\{b_{ik}:\,k\ne j\}} T^{(i)}_{x_i,\,b_{ij},\,\{b_{ik}\}}\;\prod_{k\in N(i)\setminus j} m_{k\to i}[b_{ik}].$$

In words: take `i`'s core, fold in the incoming message on **every bond except the
one going to `j`**, sum the physical index — what's left is a length-`r` vector on the
`i–j` bond. That is exactly `_bp_contract(core_i, adj_i, incoming, open_nb=j)`, whose
einsum for, say, a degree-3 node with neighbours `(j,k,l)` and open bond `j` is

```
einsum("x j k l, k, l -> j",  core_i,  m_{k->i},  m_{l->i})
#       ^physical x summed (absent from output); j left open; k,l folded
```

Two sweeps fill the cache (`_bp_messages`):

- **Upward** (leaves → root): `up[i] = m_{i→parent(i)}`, each node using its children's
  up-messages. A leaf has no children, so `up[leaf] = Σ_{x} T^{(leaf)}_{x, b}` — just its
  core with the physical summed.
- **Downward** (root → leaves): `down[i] = m_{parent(i)→i}`, each node sending to child
  `c` by folding in its parent-message and its *other* children's up-messages:

$$m_{i\to c}[b_{ic}] = \sum_{x_i}\sum_{\{b_{ik}:k\ne c\}} T^{(i)}_{x_i,b_{ic},\{b_{ik}\}}\;\Big(\!\!\prod_{\substack{k\in\text{children}(i)\\k\ne c}}\!\! m_{k\to i}\Big)\, m_{\text{parent}(i)\to i}.$$

After both passes, **every directed edge** `w→v` has a cached message
`msg(w,v) = up[w] if w is a child of v else down[v]`. Total work `O(n·r^3)`, once.

## 2. Site marginal — fold *all* incident messages

$$p(x_i) \;\propto\; \sum_{\{b_{ik}\}} T^{(i)}_{x_i,\,\{b_{ik}\}}\;\prod_{k\in N(i)} m_{k\to i}[b_{ik}].$$

Same as a message, but keep the physical index **open** and leave *no* bond open —
`_bp_contract(core_i, adj_i, {all neighbours}, open_phys=True)`:

```
einsum("x j k l, j, k, l -> x",  core_i,  m_{j->i}, m_{k->i}, m_{l->i})
```

`all_site_marginals` does this for every `i` off the one cache → all `n` marginals in
`O(n)`, versus `O(n)` *per site* for `site_marginal`.

## 3. Adjacent-pair (edge) marginal — two half-open cores across the shared bond

For a tree edge `(i,j)`, keep the `i–j` bond open on each side:

$$A_{x_i,\,b} = \!\!\sum_{k\in N(i)\setminus j}\!\! T^{(i)}_{x_i,b,\{b_{ik}\}}\!\prod m_{k\to i},
\qquad
B_{x_j,\,b} = \!\!\sum_{k\in N(j)\setminus i}\!\! T^{(j)}_{x_j,b,\{b_{jk}\}}\!\prod m_{k\to j},$$

$$p(x_i,x_j)\;\propto\; \sum_b A_{x_i,b}\,B_{x_j,b} \;=\;\texttt{einsum("xr,yr->xy", A, B)}.$$

`A`,`B` are `_node_open(..., i, exclude=j)` and `_node_open(..., j, exclude=i)` —
"core folded with all messages *except* the partner's, physical + partner-bond open".
That's `all_edge_marginals`.

## 4. Arbitrary pair — a matrix product along the path

Non-adjacent `i,j`: let the unique path be `i = v_0 - v_1 - \dots - v_L = j`. Each node
*on* the path is closed off by its **off-path** cached messages; interior path nodes
sum their physical (they're not queried), endpoints keep theirs open.

- Endpoints (physical open, one path-bond open):
$$A_{x_i,\,b_0}=\!\!\sum_{k\notin\text{path}}\!\!T^{(v_0)}_{x_i,b_0,\{b\}}\!\prod m_{k\to v_0},\qquad
  B_{x_j,\,b_{L-1}}=\dots\;(\text{same at }v_L).$$
- Interior node `v_m` (physical summed, **both** path-bonds open) → an `r×r` transfer:
$$M^{(m)}_{b_{m-1},\,b_m}=\sum_{x_{v_m}}\ \sum_{k\notin\text{path}} T^{(v_m)}_{x_{v_m},\,b_{m-1},\,b_m,\{b\}}\ \prod_{k\notin\text{path}} m_{k\to v_m}.$$

Then the pair marginal is the **matrix product** down the path with the two open
physical legs hanging off the ends:

$$p(x_i,x_j)\ \propto\ \sum_{b_0\dots b_{L-1}} A_{x_i,b_0}\,M^{(1)}_{b_0 b_1}\,M^{(2)}_{b_1 b_2}\cdots M^{(L-1)}_{b_{L-2}b_{L-1}}\,B_{x_j,b_{L-1}}.$$

Cost `O(L·r^3)` — proportional to the tree distance, **not** `n`. On a bounded-degree
latent tree, `L = O(\log n)`.

## 5. Arbitrary triple — a "Y", contracted at its branch node

Three nodes `i,j,k` span a Steiner subtree shaped like a **Y**: three paths (arms)
meeting at one branch node `m`. Each arm is a matrix product exactly as in §4, producing
a tensor with the arm's open physical leg and the arm's bond into `m`:

$$P_{x_i,\,a} \ (\text{arm to }i),\quad Q_{x_j,\,b}\ (\text{arm to }j),\quad R_{x_k,\,c}\ (\text{arm to }k).$$

The branch node `m` (physical summed if `m∉\{i,j,k\}`, else open) ties the three arms:

$$p(x_i,x_j,x_k)\ \propto\ \sum_{a,b,c}\; C^{(m)}_{a,b,c}\,P_{x_i,a}\,Q_{x_j,b}\,R_{x_k,c},
\qquad C^{(m)}_{a,b,c}=\!\!\sum_{x_m}\sum_{k\notin\text{subtree}}\!\! T^{(m)}_{x_m,a,b,c,\{b\}}\prod m_{k\to m}.$$

The general version of §4–§5 is one recursion, `_subtree_contract`'s `rec(i, tparent)`:
at each subtree node it opens the physical if the node is queried, leaves the bond to its
subtree-parent open, **recurses** into subtree-children, and **folds the cached message**
for every neighbour outside the subtree — i.e. it builds exactly the `A`/`M`/`C` pieces
above and einsum-contracts them. It is `joint_marginal` restricted to the Steiner
subtree, with the rest of the tree replaced by its cached boundary messages.

## 6. Why this is the whole scaling story

- One BP pass: `O(n·r^3)`, shared across the entire constraint list.
- Each constraint on a set `S`: `O(|\text{Steiner}(S)|·r^3)` — a pair/triple is
  `O(\text{depth}·r^3)`.
- Total for `C` constraints: `O((n + Σ|\text{Steiner}|)·r^3)` instead of `O(C·n·r^3)`.

Two things keep it cheap, both controlled by **topology + `r`**: keep coupled variables
a few hops apart (short Steiner subtrees) and keep `r` modest (low factor-rank). A
bounded-degree *latent* tree — factors as internal hubs, observables at leaves — buys
both, and makes the per-pair contractions homogeneous so they `vmap` into one batched
op (small compile graph, pairs with `reusable_adam`).

Reading efficiency is fully solved by the above; whether the model can *fit* a coupling
is the separate capacity question, set by `r` on the bonds the correlation must cross.
