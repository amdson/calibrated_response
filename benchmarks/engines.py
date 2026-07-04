"""Engine adapters: constraint bag in, fitted joint distribution out.

Every engine implements

    fit(enc, bag, seed) -> fitted

and every fitted model implements

    log_prob_rows(Xi)          (N,) log p of integer-binned rows
    marginal(var)              (d,) bin mass
    pair_marginal(var_a, var_b)  (d_a, d_b) bin mass

so metrics.py never sees engine internals. Add an engine by appending to
ENGINES at the bottom.
"""

from __future__ import annotations

import numpy as np

from .constraints import (CondExpectConstraint, CondProbConstraint,
                          Interval, MarginalConstraint, ProbConstraint)
from .encoding import TableEncoder


# ======================================================================
# Independent null: product of the (noisy) marginal constraints.
# Any engine that can't beat this is not using the correlation constraints.
# ======================================================================

class IndependentEngine:
    name = "independent"

    def fit(self, enc: TableEncoder, bag, seed: int = 0):
        margs = {c.var: np.asarray(c.probs) for c in bag
                 if isinstance(c, MarginalConstraint)}
        # uniform for any variable the bag doesn't constrain
        table = [margs.get(v, np.full(d, 1.0 / d))
                 for v, d in zip(enc.names, enc.dims)]
        return _IndependentFit(enc, table)


class _IndependentFit:
    def __init__(self, enc, table):
        self.enc, self.table = enc, [np.clip(t, 1e-12, None) for t in table]

    def log_prob_rows(self, Xi):
        return sum(np.log(t[Xi[:, i]]) for i, t in enumerate(self.table))

    def marginal(self, var):
        return self.table[self.enc.site[var]]

    def pair_marginal(self, a, b):
        return np.outer(self.marginal(a), self.marginal(b))


# ======================================================================
# Tensor chain
# ======================================================================

class TensorChainEngine:
    """Born (or nonneg) TensorChain fit to the bag by constraint SSE + regs.

    ``regularizers`` is passed straight to
    :func:`calibrated_response.tn.losses.combined_loss` — entries are
    ``(name_or_fn, weight)``.
    """

    def __init__(self, bond_dim: int = 8, kind: str = "born",
                 regularizers=(("entropy", 1e-3),), marginal_weight: float = 1.0,
                 backend: str = "adam", steps: int = 2000, lr: float = 2e-2,
                 init: str = "uniform"):
        self.name = f"tn_{kind}_r{bond_dim}"
        self.bond_dim, self.kind = bond_dim, kind
        self.regularizers = list(regularizers)
        self.marginal_weight = marginal_weight
        self.backend, self.steps, self.lr, self.init = backend, steps, lr, init

    # ---- bag -> native constraint tuples -----------------------------
    def _mask(self, enc, iv: Interval):
        m = np.zeros(enc.dims[enc.site[iv.var]], dtype=np.float32)
        m[iv.lo:iv.hi] = 1.0
        return m

    def _event(self, enc, ivs):
        return {enc.site[iv.var]: self._mask(enc, iv) for iv in ivs}

    def _convert(self, enc, bag):
        csts = []
        for c in bag:
            if isinstance(c, MarginalConstraint):
                csts.append(("kl", enc.site[c.var], np.asarray(c.probs),
                             self.marginal_weight))
            elif isinstance(c, ProbConstraint):
                csts.append(("prob", self._event(enc, c.events), c.target))
            elif isinstance(c, CondProbConstraint):
                csts.append(("cond", self._event(enc, c.event),
                             self._event(enc, c.given), c.target))
            elif isinstance(c, CondExpectConstraint):
                csts.append(("cond_expect", enc.site[c.var],
                             self._event(enc, c.given), c.target))
            else:
                raise TypeError(f"unknown constraint {type(c).__name__}")
        return csts

    def fit(self, enc: TableEncoder, bag, seed: int = 0):
        from calibrated_response.tn.chain import TensorChain
        from calibrated_response.tn.losses import combined_loss

        model = TensorChain(enc.tn_vars(), bond_dim=self.bond_dim, kind=self.kind)
        loss = combined_loss(model, self._convert(enc, bag), self.regularizers)
        kw = dict(steps=self.steps, lr=self.lr) if self.backend == "adam" else {}
        params, history = model.optimize(
            loss, backend=self.backend, seed=seed,
            init=model.init_params(seed=seed, init=self.init), **kw)
        return _TensorChainFit(enc, model, params, history)


class _TensorChainFit:
    def __init__(self, enc, model, params, history):
        self.enc, self.model, self.params, self.history = enc, model, params, history

    def log_prob_rows(self, Xi):
        return np.asarray(self.model.log_prob_idx(self.params, Xi))

    def marginal(self, var):
        return np.asarray(self.model.joint_marginal(
            self.params, (self.enc.site[var],)))

    def pair_marginal(self, a, b):
        i, j = self.enc.site[a], self.enc.site[b]
        m = np.asarray(self.model.joint_marginal(self.params, (i, j)))
        return m if i < j else m.T           # joint_marginal orders by site index


# ======================================================================
# TODO stubs — wire in the existing implementations for the bake-off.
# ======================================================================

class GaussianEngine:
    """Single N(mu, Sigma) over encoded [0,1] values, fit to the same
    functionals in closed form. Port from
    examples/maxent_tests/gaussian_baseline_toy_problems.ipynb."""
    name = "gaussian"

    def fit(self, enc, bag, seed: int = 0):
        raise NotImplementedError


class PCEngine:
    """Probabilistic-circuit adapter over calibrated_response.pc.Circuit with
    losses from calibrated_response.pc.losses (match_prob / match_cond_prob /
    match_marginal / match_expectation map 1:1 onto the bag)."""
    name = "pc"

    def fit(self, enc, bag, seed: int = 0):
        raise NotImplementedError


ENGINES = {
    "independent": IndependentEngine,
    "tn": TensorChainEngine,
    "gaussian": GaussianEngine,
    "pc": PCEngine,
}
