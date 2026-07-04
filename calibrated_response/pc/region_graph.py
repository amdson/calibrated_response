"""Static region-graph construction for the tensorized circuit (spec §3).

A *region* is a subset of variables. The root region is all ``n`` variables; we
recursively split it into two balanced halves (leaf threshold = 1 variable) to
get one binary partition hierarchy, and build ``R`` such hierarchies
(*repetitions*) over independent random variable orderings to hedge against
dependencies a single partition misses (spec §3.1, simple/random variant of §4).

The hierarchy is compiled into **depth-batched layers**: all binary partitions
whose children are already evaluated are grouped together so each depth is a
single batched tensor op (spec §9). This keeps the traced JAX graph shallow
(~log2(n) layers) regardless of ``n`` — avoiding the per-node unrolling that
blows up compile time.

Pure-numpy, evaluated once at build time. No dependency on the rest of the repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegionGraph:
    """Frozen region-graph structure shared by every circuit pass.

    Region ids are assigned in post-order, so every child id is smaller than its
    parent id. Leaf regions hold exactly one variable.
    """

    n_vars: int
    C: int                     # width (sum nodes / region)
    K: int                     # input distributions per leaf variable
    R: int                     # repetitions
    num_regions: int

    # Leaf regions (one variable each), indexed by a leaf-local index ``l``.
    leaf_region_id: np.ndarray   # (L,) region id of leaf l
    leaf_var: np.ndarray         # (L,) variable index of leaf l

    # Depth-batched internal partitions. ``layers[d]`` describes all binary
    # partitions at depth d as parallel arrays of region ids.
    layers: list                 # list of dict(parent, left, right) int arrays

    rep_root_id: np.ndarray      # (R,) root region id of each repetition

    # For (numpy) ancestral sampling: region id -> ('leaf', var) or
    # ('internal', left_id, right_id, layer_idx, local_idx).
    region_info: dict = field(default_factory=dict)


def build_region_graph(n_vars, C, K, R, seed=0):
    """Build a random balanced binary region graph with ``R`` repetitions."""
    rng = np.random.default_rng(seed)

    regions = []          # region id -> ('leaf', var) | ('internal', l, r)
    partitions = []       # (parent_id, left_id, right_id)
    leaf_region_id = []
    leaf_var = []

    def new_region(payload):
        rid = len(regions)
        regions.append(payload)
        return rid

    def build_rep(var_list):
        if len(var_list) == 1:
            rid = new_region(("leaf", int(var_list[0])))
            leaf_region_id.append(rid)
            leaf_var.append(int(var_list[0]))
            return rid
        mid = len(var_list) // 2
        left = build_rep(var_list[:mid])
        right = build_rep(var_list[mid:])
        rid = new_region(("internal", left, right))
        partitions.append((rid, left, right))
        return rid

    rep_root_id = []
    for _ in range(R):
        order = rng.permutation(n_vars)
        rep_root_id.append(build_rep(list(order)))

    num_regions = len(regions)

    # ---- compute depth (level) of each region for batching ----
    level = np.zeros(num_regions, dtype=np.int64)
    for rid in range(num_regions):                      # post-order => children first
        kind = regions[rid][0]
        if kind == "internal":
            _, l, r = regions[rid]
            level[rid] = max(level[l], level[r]) + 1

    # ---- group partitions by parent level into batched layers ----
    by_level: dict[int, list] = {}
    for (parent, left, right) in partitions:
        by_level.setdefault(level[parent], []).append((parent, left, right))

    layers = []
    region_info: dict = {}
    for l, var in zip(leaf_region_id, leaf_var):
        region_info[l] = ("leaf", var)

    for lev in sorted(by_level):
        group = by_level[lev]
        parent = np.array([g[0] for g in group], dtype=np.int64)
        left = np.array([g[1] for g in group], dtype=np.int64)
        right = np.array([g[2] for g in group], dtype=np.int64)
        layer_idx = len(layers)
        for local, (p, le, ri) in enumerate(group):
            region_info[p] = ("internal", le, ri, layer_idx, local)
        layers.append({"parent": parent, "left": left, "right": right})

    return RegionGraph(
        n_vars=n_vars,
        C=C,
        K=K,
        R=R,
        num_regions=num_regions,
        leaf_region_id=np.array(leaf_region_id, dtype=np.int64),
        leaf_var=np.array(leaf_var, dtype=np.int64),
        layers=layers,
        rep_root_id=np.array(rep_root_id, dtype=np.int64),
        region_info=region_info,
    )
