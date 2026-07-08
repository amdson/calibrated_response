"""Run LLM elicitation (variables + estimates) over a dataset, cache-first.

The elicitation cache is the shared, reusable asset of the whole eval: every
solver (maxent_smm, flow DistributionBuilder, ...) consumes the same cached
variables/estimates, so solver comparisons stay paired. This script only
fills the cache — it runs no solver.

- Async: entries run concurrently on one asyncio event loop
  (``--concurrency`` in-flight at a time, default 8) via the generators'
  ``agenerate`` / ``OpenRouterClient.aquery_structured``. Everything stays
  single-threaded — cache updates need no locking.
- Resume-safe: the cache is saved every ``--save-every`` completions (and on
  exit/failure); re-running skips ids already present.
- Serialization matches ``run_maxent_smm.ipynb`` exactly
  (``{"variables": [...], "estimates": [...]}`` keyed by entry id).
- Failures are recorded to ``<cache>.failures.json`` and skipped on rerun
  unless ``--retry-failures``.

Selecting a subset (filters compose; --dry-run shows what would run):

    # everything resolved
    --only-resolved
    # specific rounds / sources
    --rounds 2026-06-07 2026-06-21 --sources metaculus polymarket
    # a deterministic random sample of 100 after the filters above
    --sample 100 --sample-seed 0
    # at most 25 per round (balanced across time)
    --max-per-round 25
    # explicit ids (from the dataset's 'id' field), or one id per line in a file
    --ids hughuOcEOl ZkT8UYLu356zdHGRjRqR
    --ids-file picked.txt
    # stop after N new elicitations regardless of selection size
    --limit 10

Usage
-----
    python metaculus/run_elicitation.py --dataset metaculus/full_dataset.json --dry-run
    python metaculus/run_elicitation.py --dataset metaculus/full_dataset.json \
        --only-resolved --sample 100 --dry-run
    python metaculus/run_elicitation.py --dataset metaculus/full_dataset.json --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path

import dotenv

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

from calibrated_response.llm.openrouter import OpenRouterClient
from calibrated_response.generation.variable_generator import VariableGenerator
from calibrated_response.generation.natural_estimate_generator import (
    NaturalEstimateGenerator,
)
from calibrated_response.models.variable import BinaryVariable

TARGET_NAME = "target"

DEFAULT_MODEL = "google/gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# dataset / cache helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("meta", {}), data["questions"]
    return {}, data                       # bare list (tiny_dataset.json style)


def entry_question(e: dict) -> str:
    """The question text with date templates filled from the entry itself."""
    question = str(e.get("question", ""))
    for field, key in (("{resolution_date}", "resolution_date"),
                       ("{forecast_due_date}", "round_due_date")):
        if e.get(key):
            question = question.replace(field, str(e[key]))
    return question


def entry_context(e: dict) -> str:
    """Prompt context for one entry — same layout as run_maxent_smm's
    predict_binary, plus explicit calendar grounding (the model's implicit
    'now' is its training time, which wrecks every 'by June' question)."""
    return (
        f"Today's date: {e.get('round_due_date', 'unknown')}\n"
        f"Question: {entry_question(e)}\n\n"
        f"Background:\n{e.get('background', '')}\n\n"
        f"Resolution criteria:\n{e.get('resolution_criteria', '')}\n\n"
        f"Additional context:\n{e.get('source_intro', '')}"
    )


def entry_key(e: dict) -> str:
    """Cache key: question id, disambiguated by round (the same market
    question can appear in several rounds with different freeze values)."""
    return f"{e['id']}@{e.get('round_due_date', '')}"


def target_coverage(estimates) -> tuple[bool, int]:
    """(has_direct, n_coupled) — whether a direct unconditional P(target=...)
    exists, and how many estimates link target to another variable (either
    side of a conditional, or a correlation). If nothing couples the target,
    the maxent joint degenerates to the direct estimate and the solver has
    nothing to work with."""
    has_direct, n_coupled = False, 0
    for est in estimates:
        names = set()
        if hasattr(est, "proposition"):
            names.add(est.proposition.variable)
        if hasattr(est, "variable"):
            names.add(est.variable)
        if hasattr(est, "variable_a"):
            names |= {est.variable_a, est.variable_b}
        cond_names = {c.variable for c in getattr(est, "conditions", []) or []}
        if est.estimate_type == "probability" and \
                est.proposition.variable == TARGET_NAME:
            has_direct = True
        if TARGET_NAME in (names | cond_names) and \
                len(names | cond_names) > 1:
            n_coupled += 1
    return has_direct, n_coupled


def _serialize_variables(variables):
    return [{"__class__": type(v).__name__, **v.model_dump()} for v in variables]


def _serialize_estimates(estimates):
    return [x.model_dump() for x in estimates]


# ---------------------------------------------------------------------------

def select_entries(entries: list[dict], args) -> list[dict]:
    """Apply the subset filters, in order; deterministic given the args."""
    if args.only_resolved:
        entries = [e for e in entries if e.get("resolved")]
    if args.rounds:
        want = set(args.rounds)
        entries = [e for e in entries if e.get("round_due_date") in want]
    if args.sources:
        want = set(args.sources)
        entries = [e for e in entries if e.get("source") in want]
    ids = set(args.ids or [])
    if args.ids_file:
        ids |= {ln.strip() for ln in Path(args.ids_file).read_text().splitlines()
                if ln.strip()}
    if ids:
        # bare ids match any round; 'id@round' pins one exact entry (immune
        # to dataset refreshes changing the resolved set)
        entries = [e for e in entries
                   if e["id"] in ids or entry_key(e) in ids]
    if args.max_per_round is not None:
        by_round: dict[str, int] = {}
        kept = []
        for e in entries:
            r = e.get("round_due_date", "")
            if by_round.get(r, 0) < args.max_per_round:
                by_round[r] = by_round.get(r, 0) + 1
                kept.append(e)
        entries = kept
    if args.sample is not None and args.sample < len(entries):
        import random
        rng = random.Random(args.sample_seed)
        entries = rng.sample(entries, args.sample)
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", default=str(Path(__file__).parent
                                           / "llm_cache_full.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-variables", type=int, default=4)
    ap.add_argument("--n-estimates", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW elicitations")
    ap.add_argument("--only-resolved", action="store_true",
                    help="skip unresolved questions (default: elicit all — "
                         "forecasting before resolution is the cleanest "
                         "anti-contamination protocol)")
    ap.add_argument("--rounds", nargs="+", default=None,
                    help="only these round_due_dates (YYYY-MM-DD)")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="only these sources (metaculus, polymarket, manifold, "
                         "infer, acled, dbnomics, fred, wikipedia, yfinance)")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="only these question ids")
    ap.add_argument("--ids-file", default=None,
                    help="file with one question id per line")
    ap.add_argument("--max-per-round", type=int, default=None,
                    help="cap entries per round (balanced across time)")
    ap.add_argument("--sample", type=int, default=None,
                    help="deterministic random sample of this size, applied "
                         "after all other filters")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="max in-flight questions on the event loop")
    ap.add_argument("--retries", type=int, default=2,
                    help="retries per question (schema-validation failures at "
                         "temperature 0.7 are common and usually transient)")
    ap.add_argument("--reasoning-effort", default="low",
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="bound the thinking budget of reasoning models "
                         "(thinking tokens count against max_tokens; unbounded "
                         "thinking is the dominant truncation failure)")
    ap.add_argument("--save-every", type=int, default=10,
                    help="write the cache to disk every N completions")
    ap.add_argument("--retry-failures", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would run, no LLM calls")
    args = ap.parse_args(argv)

    meta, entries = load_dataset(args.dataset)
    entries = select_entries(entries, args)

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}
    fail_path = Path(str(cache_path) + ".failures.json")
    failures = json.loads(fail_path.read_text(encoding="utf-8")) \
        if fail_path.exists() else {}
    if args.retry_failures:
        failures = {}

    todo = [e for e in entries
            if entry_key(e) not in cache and entry_key(e) not in failures]
    print(f"selected: {len(entries)} entries "
          f"(cutoff {meta.get('cutoff', '?')}, retrieved {meta.get('retrieved_at', '?')})")
    print(f"cached: {len(cache)}  failed: {len(failures)}  todo: {len(todo)}")
    if todo:
        from collections import Counter
        print("todo by round:", dict(Counter(e.get('round_due_date') for e in todo)))
        print("todo by source:", dict(Counter(e.get('source') for e in todo)))
    if args.dry_run or not todo:
        return

    if args.limit is not None:
        todo = todo[:args.limit]

    reasoning = ({"enabled": False} if args.reasoning_effort == "none"
                 else {"effort": args.reasoning_effort})
    client = OpenRouterClient(model=args.model, reasoning=reasoning)
    var_gen = VariableGenerator(client)
    est_gen = NaturalEstimateGenerator(client)

    def save_cache():
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    async def elicit_one(e, sem):
        """(entry, payload, error): never raises, so gather stays simple."""
        async with sem:
            t1 = time.time()
            # the target is INJECTED, not elicited: the readout convention
            # (variables[0] is the question outcome) becomes a guarantee
            target = BinaryVariable(
                name=TARGET_NAME,
                description=f"The literal answer to the main question: "
                            f"{entry_question(e)}")
            for attempt in range(args.retries + 1):
                try:
                    ctx_vars = list(await var_gen.agenerate(
                        question=entry_context(e),
                        n_variables=args.n_variables))
                    variables = [target] + [v for v in ctx_vars
                                            if v.name != TARGET_NAME]
                    estimates = await est_gen.agenerate(
                        question=entry_context(e), variables=variables,
                        num_estimates=args.n_estimates)
                    # salvage validators may legally drop items; an empty
                    # result is a retryable failure, not a cacheable answer
                    if not ctx_vars:
                        raise ValueError("no context variables survived "
                                         "validation")
                    if not estimates:
                        raise ValueError("no estimates survived parsing")
                    # coverage failures raise INSIDE the bounded retry loop:
                    # total spend per question is hard-capped at
                    # 2 * (retries + 1) LLM calls, never a re-query loop
                    has_direct, n_coupled = target_coverage(estimates)
                    if not has_direct:
                        raise ValueError(
                            "missing direct P(target = ...) estimate")
                    if n_coupled < 1:
                        raise ValueError(
                            "no estimate links target to another variable")
                    payload = {
                        "variables": _serialize_variables(variables),
                        "estimates": _serialize_estimates(estimates),
                        "model": args.model,
                        "elicited_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "seconds": round(time.time() - t1, 1),
                        "attempts": attempt + 1,
                    }
                    return e, payload, None
                except Exception as ex:
                    if attempt >= args.retries:
                        return e, None, {"error": f"{type(ex).__name__}: {ex}",
                                         "trace": traceback.format_exc()[-1500:],
                                         "attempts": attempt + 1}
                    # provider "high demand" spikes last seconds-to-minutes;
                    # short backoffs just re-hit the same spike
                    await asyncio.sleep(5.0 * (attempt + 1))

    async def run():
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(elicit_one(e, sem)) for e in todo]
        n_done = 0
        try:
            for fut in asyncio.as_completed(tasks):
                e, payload, err = await fut
                key = entry_key(e)
                if err is not None:
                    failures[key] = err
                    fail_path.write_text(json.dumps(failures, indent=1),
                                         encoding="utf-8")
                    print(f"[FAIL] {key[:24]}...  {err['error'][:120]}")
                    continue
                cache[key] = payload
                n_done += 1
                if n_done % args.save_every == 0:
                    save_cache()
                print(f"[{n_done}/{len(todo)}] {key[:24]}...  "
                      f"{len(payload['variables'])} vars, "
                      f"{len(payload['estimates'])} est "
                      f"({payload['seconds']:.0f}s)  "
                      f"{str(e.get('question', ''))[:60]}")
        finally:
            for t in tasks:
                t.cancel()
            save_cache()
        return n_done

    t0 = time.time()
    try:
        n_done = asyncio.run(run())
    except KeyboardInterrupt:
        print("\ninterrupted — cache saved; rerun to resume")
        return

    dt = time.time() - t0
    print(f"\ndone: {n_done} new elicitations in {dt/60:.1f} min "
          f"({dt/max(n_done,1):.1f}s/question effective at "
          f"concurrency {args.concurrency}), {len(failures)} failures, "
          f"cache now {len(cache)} entries")


if __name__ == "__main__":
    main()
