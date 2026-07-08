# Constraint-reconstruction benchmark

Can an engine recover a joint distribution from **many noisy low-order
statistics**, without ever seeing rows? This isolates the *integration engine*
from the *constraint generator* (the LLM), with unlimited ground truth.

## Protocol

1. Split a real table (train/test), bin every column onto a shared grid
   ([encoding.py](encoding.py) — quantile bins for continuous, one bin per
   categorical level).
2. Compute an oracle constraint bag from the **train split only**
   ([constraints.py](constraints.py)): all 1-way marginals + random pairwise
   tail probabilities, conditional probabilities, and conditional expectations.
3. Degrade it: log-odds noise on probabilities, Gaussian noise on expectations,
   and optionally a `conflict_frac` of deliberately wrong targets.
4. Fit each engine from the bag alone ([engines.py](engines.py)) and score
   against the held-out split ([metrics.py](metrics.py)):
   - `heldout_nll` — nats/row on test rows (headline).
   - `pair_tv_unseen` — TV error on pairwise marginals **no constraint
     touched**: does the inductive bias fill in unconstrained correlations
     sensibly? This is where regularization choices show up.
   - `pair_tv_seen`, `marginal_tv` — sanity checks.

The `independent` engine (product of the noisy marginals, closed form) is the
null: an engine only demonstrates value by beating it on `heldout_nll`, and it
can only do so by exploiting the correlation constraints.

## Running

```
python -m benchmarks.run --dataset synthetic_chain            # fast smoke
python -m benchmarks.run --dataset synthetic_chain --engines independent tn tree
python -m benchmarks.run --dataset adult --engines independent tn \
    --n-pair 20 40 80 --noise 0.0 0.3 1.0 --conflict 0.0 0.2
```

Results append to `results/<dataset>.jsonl` (one row per cell — engine,
sweep coordinates, seed, scores), so sweeps are resumable and plotting is a
pandas groupby.

## The curves that matter

- **nll vs n_pair** (noise=0): value of each additional constraint.
- **nll vs noise** (fixed n_pair): degradation under realistic estimate error —
  LLM estimates live around logit-sd 0.3–1.0.
- **nll vs conflict_frac**: robustness; the spike-and-slab machinery should
  flatten this curve.
- Everything above with `pair_tv_unseen` on the y-axis: the regularizer
  comparison (entropy / amplitude_roughness / …) lives here.

## Extending

- New engine: implement `fit(enc, bag, seed) -> fitted` with
  `log_prob_rows / marginal / pair_marginal`, register in `ENGINES`.
  Available: `independent`, `tn` (born tensor *chain*), `tree` (tensor *tree* —
  variables at the leaves of a balanced binary latent tree; defaults to
  `kind="nonneg"`, which scores constraints through the batched
  `SteinerMarginals`/`BPPlan` sweep), `flow` (invertible flow maxent — exact
  density via the inverse pass), `gaussian` (single joint Gaussian; weak here
  by construction — quantile encoding makes marginals uniform, which a
  Gaussian is misspecified for), `copula` (Gaussian copula: exact histogram
  marginals + one correlation matrix, the strong linear baseline).
  `PCEngine` is a stub awaiting a port.

  `tn` vs `tree` isolates *topology* (a line vs a balanced binary latent tree)
  at fixed constraint machinery; set the same `kind` on both for an
  apples-to-apples comparison. Unlike `tn`, `tree` has no compile-once fast
  sweep (`sweep_tn_fast` walks a linear chain), so each `tree` cell pays its own
  compile via `run_cell`.
- New dataset: add a loader to `DATASETS` returning `(train_df, test_df)`.
