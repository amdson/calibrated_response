"""Sweep driver: (dataset x engine x constraint budget x noise x seed) -> JSONL.

Each cell fits one engine on one noisy bag and appends one result row to
``benchmarks/results/<dataset>.jsonl``, so runs are resumable and plots are a
groupby away. Usage:

    python -m benchmarks.run --dataset synthetic_chain --engines independent tn
    python -m benchmarks.run --dataset adult --n-pair 20 60 --noise 0.0 0.3 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .constraints import noisy_bag, true_bag
from .datasets import DATASETS
from .encoding import TableEncoder
from .engines import ENGINES, TensorChainEngine
from .metrics import score

RESULTS_DIR = Path(__file__).parent / "results"


def run_cell(engine, enc, Xi_train, Xi_test, *, n_pair, noise, conflict, seed):
    bag, pairs = true_bag(Xi_train, enc, seed=seed, n_pair=n_pair,
                          n_cond=n_pair // 2, n_cond_expect=n_pair // 4)
    bag = noisy_bag(bag, seed=seed, prob_logit_sd=noise,
                    expect_sd=0.05 * noise, conflict_frac=conflict)
    t0 = time.perf_counter()
    fitted = engine.fit(enc, bag, seed=seed)
    row = score(fitted, enc, Xi_test, pairs, seed=seed)
    row.update(engine=engine.name, n_pair=n_pair, noise=noise,
               conflict=conflict, seed=seed,
               fit_seconds=round(time.perf_counter() - t0, 2))
    return row


class _FastFit:
    """A fitted-model view over *shared, cached* jitted query functions.

    :func:`benchmarks.metrics.score` calls ``log_prob_rows`` once and
    ``marginal`` / ``pair_marginal`` many times per cell. Building the jitted
    contractions here (keyed by site tuple) and reusing them across every cell of
    a sweep means each distinct query shape is compiled once for the whole sweep,
    not once per cell — the scoring analog of the compile-once fit.
    """

    def __init__(self, model, params, enc, logp_fn, jm_fn):
        self.model, self.params, self.enc = model, params, enc
        self._logp, self._jm = logp_fn, jm_fn

    def log_prob_rows(self, Xi):
        import jax.numpy as jnp
        return np.asarray(self._logp(self.params, jnp.asarray(Xi, jnp.int32)))

    def marginal(self, var):
        return np.asarray(self._jm(self.params, (self.enc.site[var],)))

    def pair_marginal(self, a, b):
        i, j = self.enc.site[a], self.enc.site[b]
        m = np.asarray(self._jm(self.params, (i, j)))
        return m if i < j else m.T


def sweep_tn_fast(engine, enc, Xi_train, Xi_test, *, n_pair, noise_grid, seeds,
                  conflict=0.0, events_seed=0, progress=True):
    """Sweep a :class:`TensorChainEngine` over (noise, seed) with **one** compile.

    The constraint *structure* (which events/marginals the bag pins) is drawn once
    from ``events_seed`` and held fixed; each cell only re-draws the *noisy target
    values* (:func:`noisy_bag`), which are fed to a reusable, batched loss
    (:func:`losses.batched_constraint_loss` + :func:`chain.reusable_adam`). So the
    ~10-30 s XLA compile is paid a single time for the whole sweep instead of once
    per cell, and scoring reuses cached jitted contractions (:class:`_FastFit`).

    Returns the same result rows as :func:`run_cell`. Because events are fixed,
    ``seed`` here reseeds *the noise and the model init only* (not the constraint
    draw) — the intended axis for a noise/robustness sweep.
    """
    import jax
    import jax.numpy as jnp
    from calibrated_response.tn.chain import TensorChain, reusable_adam
    from calibrated_response.tn.losses import batched_constraint_loss

    model = TensorChain(enc.tn_vars(), bond_dim=engine.bond_dim, kind=engine.kind)
    base_bag, pairs = true_bag(Xi_train, enc, seed=events_seed, n_pair=n_pair,
                               n_cond=n_pair // 2, n_cond_expect=n_pair // 4)
    loss_fn, pack = batched_constraint_loss(
        model, engine._convert(enc, base_bag), engine.regularizers)
    fit = reusable_adam(loss_fn, steps=engine.steps, lr=engine.lr)

    # Cached jitted scoring queries, shared across every cell (#4).
    logp = jax.jit(lambda p, X: model.log_prob_idx(p, X))
    jm_cache: dict = {}

    def jm(p, sites):
        key = tuple(sorted(sites))
        if key not in jm_cache:
            jm_cache[key] = jax.jit(lambda p, s=key: model.joint_marginal(p, s))
        return jm_cache[key](p)

    rows = []
    for noise in noise_grid:
        for seed in seeds:
            bag = noisy_bag(base_bag, seed=seed, prob_logit_sd=noise,
                            expect_sd=0.05 * noise, conflict_frac=conflict)
            targets = pack(engine._convert(enc, bag))
            t0 = time.perf_counter()
            init = model.init_params(seed=seed, init=engine.init)
            params, _ = fit(init, targets)
            fitted = _FastFit(model, params, enc, logp, jm)
            row = score(fitted, enc, Xi_test, pairs, seed=seed)
            row.update(engine=engine.name, n_pair=n_pair, noise=noise,
                       conflict=conflict, seed=seed,
                       fit_seconds=round(time.perf_counter() - t0, 2))
            rows.append(row)
            if progress:
                print(f"  {row['engine']:>14}  noise={noise:<4} seed={seed}  "
                      f"nll={row['heldout_nll']:.4f}  "
                      f"pair_tv_unseen={row['pair_tv_unseen']:.4f}  "
                      f"({row['fit_seconds']}s)")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic_chain", choices=DATASETS)
    ap.add_argument("--engines", nargs="+", default=["independent", "tn"],
                    choices=ENGINES)
    ap.add_argument("--n-pair", nargs="+", type=int, default=[40])
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--conflict", nargs="+", type=float, default=[0.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-bins-cont", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args(argv)

    kw = {"max_rows": args.max_rows} if args.max_rows else {}
    train, test = DATASETS[args.dataset](**kw)
    enc = TableEncoder(train, n_bins_cont=args.n_bins_cont)
    Xi_train, Xi_test = enc.bin_indices(train), enc.bin_indices(test)
    print(f"{args.dataset}: {len(train)} train / {len(test)} test rows, "
          f"{len(enc.names)} vars, dims={enc.dims}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.dataset}.jsonl"
    with open(out_path, "a") as fh:
        for name in args.engines:
            engine = ENGINES[name]()
            for n_pair in args.n_pair:
                for noise in args.noise:
                    for conflict in args.conflict:
                        for seed in args.seeds:
                            row = run_cell(engine, enc, Xi_train, Xi_test,
                                           n_pair=n_pair, noise=noise,
                                           conflict=conflict, seed=seed)
                            fh.write(json.dumps(row) + "\n")
                            fh.flush()
                            print(f"  {row['engine']:>14}  n_pair={n_pair:<3} "
                                  f"noise={noise:<4} conflict={conflict:<4} "
                                  f"seed={seed}  nll={row['heldout_nll']:.4f}  "
                                  f"pair_tv_unseen={row['pair_tv_unseen']:.4f}  "
                                  f"({row['fit_seconds']}s)")
    print(f"appended to {out_path}")


if __name__ == "__main__":
    main()
