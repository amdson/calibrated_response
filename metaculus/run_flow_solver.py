"""Fit the flow maxent solver on cached elicitations and save predictions.

Consumes the elicitation cache written by ``run_elicitation.py`` (no LLM
calls here — this is the free, re-runnable half of the pipeline) and writes
one prediction row per entry to ``--out``:

    {key, p_target, direct_llm, market, resolved_to, source, round,
     entropy, n_constraints, n_skipped, n_clip_warnings, max_abs_error,
     min_p_cond, seconds, config}

- Resume-safe: rows are keyed by entry and saved after every fit; re-running
  skips existing keys. --out is REQUIRED and should be a fresh
  ``runs/<date>-<slug>/`` directory per solver run, so a resume can never
  cross a code change (the mechanism behind the 2026-07-14 stale-row
  contamination). A ``manifest.json`` with the config + git SHA is written
  next to each output file.
- Repeated estimates of one quantity are collapsed by default (duplicates
  count once, disagreement widens — see ``protocol.collapse_repeats``);
  ``--no-collapse`` is the escape hatch that lets k repeats sharpen the
  solver by sqrt(k).
- ``--no-corr`` drops CorrelationEstimate before building — the ablation that
  separates "better solver" from "more information used".
- ``--robust`` fits with per-estimate credence gates.

Usage
-----
    python metaculus/run_flow_solver.py \
        --out metaculus/runs/2026-07-15-mytest/flow_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from pydantic import TypeAdapter

from calibrated_response.models.query import (CorrelationEstimate,
                                              EstimateUnion,
                                              ProbabilityEstimate)
from calibrated_response.models.variable import (BinaryVariable,
                                                 ContinuousVariable)
from calibrated_response.maxent_sampler import DistributionBuilder
from calibrated_response.generation.protocol import collapse_repeats

from run_elicitation import TARGET_NAME, entry_key, load_dataset

_ADAPTER = TypeAdapter(EstimateUnion)
_VAR_CLASSES = {"BinaryVariable": BinaryVariable,
                "ContinuousVariable": ContinuousVariable}
MARKET_SOURCES = {"metaculus", "manifold", "polymarket", "infer"}


def deserialize(entry: dict):
    variables = []
    for d in entry["variables"]:
        d = dict(d)
        cls = _VAR_CLASSES[d.pop("__class__")]
        variables.append(cls.model_validate(d))
    estimates = [_ADAPTER.validate_python(d) for d in entry["estimates"]]
    return variables, estimates


def direct_llm_estimate(estimates) -> float | None:
    """The LLM's own unconditional P(target = True) — the paired baseline."""
    for est in estimates:
        if isinstance(est, ProbabilityEstimate) and \
                est.proposition.variable == TARGET_NAME:
            p = float(est.probability)
            return p if est.proposition.value else 1.0 - p
    return None


def market_value(e: dict) -> float | None:
    """Market probability at freeze — only meaningful for market sources
    (for data-series questions freeze_datetime_value is the series value)."""
    if e.get("source") not in MARKET_SOURCES:
        return None
    try:
        v = float(e.get("freeze_datetime_value"))
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path(__file__).parent
                                             / "data" / "full_dataset.json"))
    ap.add_argument("--cache", default=str(Path(__file__).parent / "caches"
                                           / "full" / "llm_cache_full.json"))
    ap.add_argument("--out", required=True,
                    help="prediction file; use a fresh runs/<date>-<slug>/ "
                         "directory per solver run so resume never crosses "
                         "a code change")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--entropy-reg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-estimates", type=int, default=None,
                    help="give the solver only the first N estimates (elicited "
                         "order, direct target estimate always kept) — an "
                         "information-density sweep over one cache, so arms "
                         "differ only in how much the solver sees")
    ap.add_argument("--no-collapse", action="store_true",
                    help="do NOT fold repeated estimates of the same quantity "
                         "into one spread-derived-sd estimate. Collapse is the "
                         "default (defense in depth: even a cache that still "
                         "contains echoed duplicates cannot sqrt(k)-sharpen "
                         "the solver); this flag restores k independent "
                         "penalties per quantity")
    ap.add_argument("--no-corr", action="store_true",
                    help="drop CorrelationEstimate (information-diet ablation)")
    ap.add_argument("--prob-penalty", default="logit", choices=["logit", "abs"],
                    help="probability-constraint residual space: 'logit' "
                         "(multiplicative slack, kills rare-event inflation) "
                         "or 'abs' (legacy absolute)")
    ap.add_argument("--prob-logit-sd", type=float, default=0.3)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--protect-anchor", action="store_true",
                    help="keep the direct target estimate ungated in robust "
                         "mode (off by default: the joint should be allowed "
                         "to outweigh the anchor)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None,
                    help="'i/n': process every n-th todo entry starting at i. "
                         "Poor-man's parallelism across sessions — each shard "
                         "MUST write to its own --out file; merge afterwards "
                         "(the files are plain {key: row} dicts).")
    args = ap.parse_args(argv)

    _, entries = load_dataset(args.dataset)
    by_key = {entry_key(e): e for e in entries}
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))

    out_path = Path(args.out)
    preds = json.loads(out_path.read_text(encoding="utf-8")) \
        if out_path.exists() else {}
    config = {"steps": args.steps, "n_samples": args.n_samples,
              "entropy_reg": args.entropy_reg, "no_corr": args.no_corr,
              "collapse_repeats": not args.no_collapse,
              "prob_penalty": args.prob_penalty,
              "prob_logit_sd": args.prob_logit_sd,
              "robust": args.robust, "protect_anchor": args.protect_anchor,
              "max_estimates": args.max_estimates,
              "cache": Path(args.cache).name, "seed": args.seed}

    # audit trail: config + git SHA per output file, readable without
    # loading a prediction file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent).stdout.strip() or None
    except OSError:
        sha = None
    prev = manifest.get(out_path.name)
    if prev and preds and (prev.get("git_sha") != sha
                           or prev.get("config") != config):
        print(f"WARNING: resuming {out_path.name} over rows fit at "
              f"sha={prev.get('git_sha')} config={prev.get('config')} — "
              f"stale-row contamination; use a fresh runs/<date>-<slug>/ "
              f"directory instead")
    manifest[out_path.name] = {
        "config": config, "git_sha": sha,
        "dataset": Path(args.dataset).name,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest_path.write_text(json.dumps(manifest, indent=1),
                             encoding="utf-8")

    todo = [k for k in cache if k in by_key and k not in preds]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
    print(f"cache: {len(cache)}  already predicted: {len(preds)}  "
          f"todo: {len(todo)}{f'  (shard {args.shard})' if args.shard else ''}"
          f"  config: {config}")

    n_done = 0
    for key in todo:
        if args.limit is not None and n_done >= args.limit:
            break
        e = by_key[key]
        t0 = time.time()
        try:
            variables, estimates = deserialize(cache[key])
            direct = direct_llm_estimate(estimates)  # from raw repeats, so the
            #        paired baseline is identical across collapse on/off arms
            if not args.no_collapse:
                estimates = collapse_repeats(
                    estimates, prob_logit_sd=args.prob_logit_sd)
            if args.no_corr:
                estimates = [x for x in estimates
                             if not isinstance(x, CorrelationEstimate)]
            if args.max_estimates is not None and \
                    len(estimates) > args.max_estimates:
                di = next((i for i, x in enumerate(estimates)
                           if isinstance(x, ProbabilityEstimate)
                           and x.proposition.variable == TARGET_NAME), None)
                keep = [] if di is None else [estimates[di]]
                rest = [x for i, x in enumerate(estimates) if i != di]
                estimates = keep + rest[:args.max_estimates - len(keep)]
            builder = DistributionBuilder(variables, estimates,
                                          prob_penalty=args.prob_penalty,
                                          prob_logit_sd=args.prob_logit_sd,
                                          robust=args.robust,
                                          anchor_variable=(
                                              TARGET_NAME if args.protect_anchor
                                              else None))
            dist, info = builder.build(
                target_variable=TARGET_NAME, steps=args.steps,
                n_samples=args.n_samples, entropy_reg=args.entropy_reg,
                seed=args.seed)
            report = info["report"]
            p_conds = [r["p_cond"] for r in report if r["p_cond"] is not None]
            preds[key] = {
                "p_target": float(dist.probability),
                "direct_llm": direct,
                "market": market_value(e),
                "resolved_to": e.get("resolved_to"),
                "source": e.get("source"),
                "round": e.get("round_due_date"),
                "entropy": info["entropy"],
                "n_constraints": info["n_constraints"],
                "n_skipped": len(info["skipped_constraints"]),
                "n_clip_warnings": len([w for w in info["warnings"]
                                        if "clipped" in w]),
                "max_abs_error": (max(abs(r["error_rel"]) for r in report)
                                  if report else None),
                "min_p_cond": min(p_conds) if p_conds else None,
                "seconds": round(time.time() - t0, 1),
                "config": config,
            }
            n_done += 1
            print(f"[{n_done}] {key[:30]}...  P={preds[key]['p_target']:.3f} "
                  f"(llm {direct if direct is None else round(direct, 3)}, "
                  f"market {preds[key]['market']}, "
                  f"resolved {preds[key]['resolved_to']}) "
                  f"({preds[key]['seconds']}s)")
        except Exception as ex:
            preds[key] = {"error": f"{type(ex).__name__}: {ex}",
                          "trace": traceback.format_exc()[-1000:],
                          "config": config}
            print(f"[FAIL] {key[:30]}...  {type(ex).__name__}: {ex}")
        out_path.write_text(json.dumps(preds, indent=1), encoding="utf-8")

    ok = [p for p in preds.values() if "p_target" in p]
    print(f"\n{len(ok)} predictions in {out_path}")


if __name__ == "__main__":
    main()
