"""Engine-agnostic constraint bags.

A *bag* is a list of constraint dataclasses whose targets are empirical
statistics of the train split (the "oracle" forecaster). :func:`noisy_bag` then
degrades the oracle: probabilities are jittered on the log-odds scale,
expectations in encoded units, and a ``conflict_frac`` of constraints is
replaced with outright wrong values to exercise robustness machinery.

Events are half-open bin ranges ``[lo, hi)`` on the shared grid, so every
engine interprets a constraint identically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .encoding import TableEncoder


@dataclass(frozen=True)
class Interval:
    """Event ``X_var in bins [lo, hi)``."""
    var: str
    lo: int
    hi: int


@dataclass(frozen=True)
class MarginalConstraint:
    var: str
    probs: tuple                   # full histogram, len n_bins


@dataclass(frozen=True)
class ProbConstraint:
    events: tuple                  # tuple[Interval, ...], conjunction
    target: float


@dataclass(frozen=True)
class CondProbConstraint:
    event: tuple                   # tuple[Interval, ...]
    given: tuple                   # tuple[Interval, ...]
    target: float


@dataclass(frozen=True)
class CondExpectConstraint:
    var: str                       # target in encoded [0,1] units (bin centres)
    given: tuple                   # tuple[Interval, ...]
    target: float


# ----------------------------------------------------------------------
def _rows_in(Xi: np.ndarray, enc: TableEncoder, events) -> np.ndarray:
    m = np.ones(len(Xi), dtype=bool)
    for ev in events:
        col = Xi[:, enc.site[ev.var]]
        m &= (col >= ev.lo) & (col < ev.hi)
    return m


def _tail(enc: TableEncoder, var: str, rng: np.random.Generator) -> Interval:
    """A random one-sided tail event with a not-too-extreme split point."""
    d = enc.dims[enc.site[var]]
    s = int(rng.integers(1, d))
    return Interval(var, s, d) if rng.random() < 0.5 else Interval(var, 0, s)


def true_bag(Xi: np.ndarray, enc: TableEncoder, seed: int = 0,
             n_pair: int = 40, n_cond: int = 20, n_cond_expect: int = 10,
             min_given: int = 30):
    """Empirical constraint bag. Returns ``(bag, constrained_pairs)`` where
    ``constrained_pairs`` is the set of frozenset({var_i, var_j}) touched by any
    2-variable constraint — metrics use its complement as the *unseen* pairs."""
    rng = np.random.default_rng(seed)
    bag, pairs = [], set()

    for var in enc.names:
        d = enc.dims[enc.site[var]]
        counts = np.bincount(Xi[:, enc.site[var]], minlength=d)
        bag.append(MarginalConstraint(var, tuple(counts / counts.sum())))

    def draw_pair():
        i, j = rng.choice(len(enc.names), size=2, replace=False)
        return enc.names[i], enc.names[j]

    for _ in range(n_pair):
        a, b = draw_pair()
        evs = (_tail(enc, a, rng), _tail(enc, b, rng))
        bag.append(ProbConstraint(evs, float(_rows_in(Xi, enc, evs).mean())))
        pairs.add(frozenset((a, b)))

    for _ in range(n_cond):
        a, b = draw_pair()
        ev, gv = (_tail(enc, a, rng),), (_tail(enc, b, rng),)
        sel = _rows_in(Xi, enc, gv)
        if sel.sum() < min_given:
            continue
        bag.append(CondProbConstraint(
            ev, gv, float(_rows_in(Xi[sel], enc, ev).mean())))
        pairs.add(frozenset((a, b)))

    for _ in range(n_cond_expect):
        a, b = draw_pair()
        gv = (_tail(enc, b, rng),)
        sel = _rows_in(Xi, enc, gv)
        if sel.sum() < min_given:
            continue
        centers = enc.centers(a)
        bag.append(CondExpectConstraint(
            a, gv, float(centers[Xi[sel, enc.site[a]]].mean())))
        pairs.add(frozenset((a, b)))

    return bag, pairs


# ----------------------------------------------------------------------
def _logit_jitter(p, sd, rng, lo=1e-3):
    p = np.clip(np.asarray(p, dtype=np.float64), lo, 1 - lo)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + rng.normal(0, sd, np.shape(p)))))


def noisy_bag(bag, seed: int = 0, prob_logit_sd: float = 0.0,
              expect_sd: float = 0.0, conflict_frac: float = 0.0):
    """Degrade an oracle bag. ``prob_logit_sd`` jitters every probability on the
    log-odds scale (marginal histograms are jittered per-bin and renormalised);
    ``expect_sd`` is Gaussian noise in encoded [0,1] units; ``conflict_frac``
    replaces that fraction of non-marginal constraints with uniform-random
    targets (deliberately wrong, for robustness tests)."""
    rng = np.random.default_rng(seed)
    out = []
    for c in bag:
        conflicted = (not isinstance(c, MarginalConstraint)
                      and rng.random() < conflict_frac)
        if isinstance(c, MarginalConstraint):
            p = _logit_jitter(c.probs, prob_logit_sd, rng) if prob_logit_sd else np.asarray(c.probs)
            out.append(replace(c, probs=tuple(p / p.sum())))
        elif isinstance(c, (ProbConstraint, CondProbConstraint)):
            t = rng.uniform(0.05, 0.95) if conflicted else \
                float(_logit_jitter(c.target, prob_logit_sd, rng))
            out.append(replace(c, target=t))
        elif isinstance(c, CondExpectConstraint):
            t = rng.uniform(0.0, 1.0) if conflicted else \
                float(np.clip(c.target + rng.normal(0, expect_sd), 0.0, 1.0))
            out.append(replace(c, target=t))
        else:
            raise TypeError(f"unknown constraint {type(c).__name__}")
    return out
