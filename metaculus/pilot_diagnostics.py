"""Aggregate health report for a pilot: elicitation cache + solver predictions.

Two sections:

1. **Elicitation health** (cache only): coverage of the target contract,
   estimate-type mix, variable sanity, retry/latency stats, failure count.
2. **Solver health + scoring** (per predictions file from run_flow_solver.py):
   skip/clip rates, residuals, conditioning budgets, and Brier scores —
   flow vs the LLM's own direct estimate (paired) and vs the market at freeze
   (market-source subset). Pass several --predictions files to compare
   configs (e.g. the --no-corr ablation) side by side.

Usage
-----
    python metaculus/pilot_diagnostics.py
    python metaculus/pilot_diagnostics.py \
        --predictions metaculus/flow_predictions.json \
                      metaculus/flow_predictions_nocorr.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from calibrated_response.models.query import EstimateUnion
from run_elicitation import TARGET_NAME, target_coverage

_ADAPTER = TypeAdapter(EstimateUnion)


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def elicitation_report(cache: dict, failures: dict) -> None:
    print("=" * 72)
    print(f"ELICITATION: {len(cache)} cached, {len(failures)} failed "
          f"({100 * len(failures) / max(len(cache) + len(failures), 1):.1f}%)")
    if not cache:
        return

    n_vars, n_ests, types, attempts, secs = [], [], Counter(), Counter(), []
    n_direct, couplings, bad_bounds, no_unit = 0, [], 0, 0
    for entry in cache.values():
        variables, raw_ests = entry["variables"], entry["estimates"]
        ests = [_ADAPTER.validate_python(d) for d in raw_ests]
        n_vars.append(len(variables))
        n_ests.append(len(ests))
        types.update(e.estimate_type for e in ests)
        attempts[entry.get("attempts", 1)] += 1
        if entry.get("seconds"):
            secs.append(entry["seconds"])
        has_direct, n_coupled = target_coverage(ests)
        n_direct += has_direct
        couplings.append(n_coupled)
        for v in variables:
            if v["__class__"] == "ContinuousVariable":
                if not v["upper_bound"] > v["lower_bound"]:
                    bad_bounds += 1
                if not v.get("unit"):
                    no_unit += 1

    n = len(cache)
    print(f"  variables/entry: mean {np.mean(n_vars):.1f}   "
          f"estimates/entry: mean {np.mean(n_ests):.1f}")
    print(f"  estimate types: {dict(types)}")
    print(f"  direct P(target): {n_direct}/{n}   "
          f"target couplings/entry: mean {np.mean(couplings):.1f} "
          f"min {min(couplings)}")
    print(f"  attempts: {dict(attempts)}   "
          f"seconds/entry: median {_pct(secs, 50):.0f} p90 {_pct(secs, 90):.0f}")
    print(f"  continuous vars with bad bounds: {bad_bounds}, "
          f"missing unit: {no_unit}")


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def scoring_report(label: str, preds: dict) -> None:
    ok = {k: v for k, v in preds.items() if "p_target" in v}
    failed = len(preds) - len(ok)
    print("=" * 72)
    print(f"SOLVER [{label}]: {len(ok)} fits, {failed} failures")
    if not ok:
        return
    rows = list(ok.values())
    cfg = rows[0].get("config", {})
    print(f"  config: {cfg}")
    secs = [r["seconds"] for r in rows]
    print(f"  fit seconds: median {_pct(secs, 50):.0f} p90 {_pct(secs, 90):.0f}")
    print(f"  skipped constraints/entry: mean "
          f"{np.mean([r['n_skipped'] for r in rows]):.2f}   "
          f"clip warnings/entry: mean "
          f"{np.mean([r['n_clip_warnings'] for r in rows]):.2f}")
    maxe = [r["max_abs_error"] for r in rows if r["max_abs_error"] is not None]
    pcs = [r["min_p_cond"] for r in rows if r["min_p_cond"] is not None]
    print(f"  max |residual|/entry: median {_pct(maxe, 50):.3f} "
          f"p90 {_pct(maxe, 90):.3f}   min p_cond: median {_pct(pcs, 50):.3f} "
          f"p10 {_pct(pcs, 10):.3f}")

    # ---- scoring on resolved entries ------------------------------------
    scored = [r for r in rows if r.get("resolved_to") in (0.0, 1.0)]
    if not scored:
        print("  (no resolved entries to score)")
        return
    y = [r["resolved_to"] for r in scored]
    p_flow = [r["p_target"] for r in scored]
    print(f"  scored on {len(scored)} resolved  "
          f"(base rate {np.mean(y):.2f})")
    print(f"    Brier flow:   {brier(p_flow, y):.4f}")

    paired = [r for r in scored if r.get("direct_llm") is not None]
    if paired:
        d = (np.array([(r['p_target'] - r['resolved_to']) ** 2 for r in paired])
             - np.array([(r['direct_llm'] - r['resolved_to']) ** 2
                         for r in paired]))
        print(f"    Brier direct: "
              f"{brier([r['direct_llm'] for r in paired], [r['resolved_to'] for r in paired]):.4f} "
              f"on {len(paired)}  | paired delta (flow - direct): "
              f"{np.mean(d):+.4f} ± {np.std(d) / np.sqrt(len(d)):.4f}")

    mkt = [r for r in scored if r.get("market") is not None]
    if mkt:
        d = (np.array([(r['p_target'] - r['resolved_to']) ** 2 for r in mkt])
             - np.array([(r['market'] - r['resolved_to']) ** 2 for r in mkt]))
        print(f"    Brier market: "
              f"{brier([r['market'] for r in mkt], [r['resolved_to'] for r in mkt]):.4f} "
              f"on {len(mkt)}  | paired delta (flow - market): "
              f"{np.mean(d):+.4f} ± {np.std(d) / np.sqrt(len(d)):.4f}")

    # ---- calibration ------------------------------------------------------
    edges = [0, .1, .25, .5, .75, .9, 1.0001]
    print("    calibration (flow):")
    p_arr, y_arr = np.asarray(p_flow), np.asarray(y, float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_arr >= lo) & (p_arr < hi)
        if m.sum():
            print(f"      p in [{lo:.2f},{hi:.2f}): n={int(m.sum()):3d}  "
                  f"mean p={p_arr[m].mean():.2f}  freq={y_arr[m].mean():.2f}")


def main(argv=None):
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(here / "llm_cache_full.json"))
    ap.add_argument("--predictions", nargs="*", default=None,
                    help="run_flow_solver output file(s); compared side by side")
    args = ap.parse_args(argv)

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}
    fail_path = Path(str(cache_path) + ".failures.json")
    failures = json.loads(fail_path.read_text(encoding="utf-8")) \
        if fail_path.exists() else {}
    elicitation_report(cache, failures)

    pred_paths = args.predictions
    if pred_paths is None:
        default = here / "flow_predictions.json"
        pred_paths = [str(default)] if default.exists() else []
    for p in pred_paths:
        preds = json.loads(Path(p).read_text(encoding="utf-8"))
        scoring_report(Path(p).stem, preds)


if __name__ == "__main__":
    main()
