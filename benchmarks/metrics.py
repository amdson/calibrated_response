"""Scores for a fitted engine against a held-out split.

The headline numbers:

- ``heldout_nll`` — mean negative log p of held-out binned rows (nats/row).
  Lower is better; ``independent`` gives the reference the joint must beat.
- ``pair_tv_unseen`` — mean total-variation distance between model and
  empirical held-out pairwise marginals, over pairs *no constraint touched*.
  This is the generalisation question: does the engine's inductive bias fill
  in unconstrained correlations sensibly, or invent junk?
- ``pair_tv_seen`` — same over constrained pairs (did it even use the bag?).
- ``marginal_tv`` — mean TV over single-variable marginals (basic sanity).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def _empirical_pair(Xi, enc, a, b):
    i, j = enc.site[a], enc.site[b]
    h = np.zeros((enc.dims[i], enc.dims[j]))
    np.add.at(h, (Xi[:, i], Xi[:, j]), 1.0)
    return h / h.sum()


def score(fitted, enc, Xi_test, constrained_pairs, max_pairs: int = 200,
          seed: int = 0) -> dict:
    out = {"heldout_nll": float(-np.mean(fitted.log_prob_rows(Xi_test)))}

    tvs = []
    for var in enc.names:
        emp = np.bincount(Xi_test[:, enc.site[var]],
                          minlength=enc.dims[enc.site[var]])
        tvs.append(_tv(fitted.marginal(var), emp / emp.sum()))
    out["marginal_tv"] = float(np.mean(tvs))

    all_pairs = [frozenset(p) for p in combinations(enc.names, 2)]
    seen = [p for p in all_pairs if p in constrained_pairs]
    unseen = [p for p in all_pairs if p not in constrained_pairs]
    rng = np.random.default_rng(seed)
    for label, pairs in (("seen", seen), ("unseen", unseen)):
        if len(pairs) > max_pairs:
            pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs,
                                                  replace=False)]
        tvs = [_tv(fitted.pair_marginal(a, b), _empirical_pair(Xi_test, enc, a, b))
               for a, b in (sorted(p) for p in pairs)]
        out[f"pair_tv_{label}"] = float(np.mean(tvs)) if tvs else float("nan")
    return out
