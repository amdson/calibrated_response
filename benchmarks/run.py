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

from .constraints import noisy_bag, true_bag
from .datasets import DATASETS
from .encoding import TableEncoder
from .engines import ENGINES
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
