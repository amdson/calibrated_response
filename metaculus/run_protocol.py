"""Run a multi-pass elicitation protocol over a dataset, cache-first.

The protocol version of ``run_elicitation.py``: each entry's elicitation is
an explicit list of enricher nodes over one shared state (see
``calibrated_response.generation.protocol``). The cache stays consumable by
``run_flow_solver.py`` unchanged — same ``{variables, estimates}`` layout,
estimates carry an extra ``provenance`` field naming the node that produced
them (pydantic ignores it on deserialize; diagnostics can group by it).

Resource bound: worst-case LLM calls per question = (# llm nodes) x
(retries + 1), printed before launch. ``propose_requests`` nodes are pure
code and free. There is no re-query loop.

Usage
-----
    python metaculus/run_protocol.py --dataset metaculus/data/full_dataset.json \
        --protocol v1 --ids-file metaculus/data/pilot_ids.txt --limit 2
    python metaculus/run_protocol.py ... --show 2      # pretty-print cache
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
from calibrated_response.models.variable import BinaryVariable
from calibrated_response.generation.protocol import (ENRICHERS, State, merge,
                                                     n_llm_calls)

from run_elicitation import (DEFAULT_MODEL, TARGET_NAME, entry_context,
                             entry_key, entry_question, load_dataset,
                             select_entries, target_coverage,
                             _serialize_variables, _serialize_estimates)

# ---------------------------------------------------------------------------
# protocols: explicit node lists — the whole experiment definition
# ---------------------------------------------------------------------------

PROTOCOLS = {
    # the current single-pass pipeline, reproduced in the new machinery
    # (2 calls/question, same as run_elicitation.py)
    "baseline": [
        ("gen_variables", {"mode": "drivers", "n": 4}),
        ("gen_estimates", {"scope": "free", "n": 10}),
    ],
    # v1 multi-pass: variables -> marginal battery -> target couplings ->
    # programmatic requests (only quantities with NO existing estimate:
    # missing direct/marginals + complement arms) -> one fill pass.
    # 4 calls/question.
    "v1": [
        ("gen_variables", {"mode": "drivers", "n": 4}),
        ("gen_estimates", {"scope": "marginals"}),
        ("gen_estimates", {"scope": "target_connections"}),
        ("propose_requests", {"rules": ["direct", "marginals",
                                        "complements"]}),
        ("fill_requests", {}),
    ],
    # v1 + a second fill of the same request battery in a FRESH context
    # (prior estimates hidden) = one genuinely independent repeat of every
    # requested quantity (5 calls/question). A repeat that can see the
    # first answer just echoes it — see protocol.py.
    "v1x2": [
        ("gen_variables", {"mode": "drivers", "n": 4}),
        ("gen_estimates", {"scope": "marginals"}),
        ("gen_estimates", {"scope": "target_connections"}),
        ("propose_requests", {"rules": ["direct", "marginals",
                                        "complements"]}),
        ("fill_requests", {}),
        ("fill_requests", {"hide_estimates": True}),
    ],
    # fermi decomposition variant of v1
    "v1_fermi": [
        ("gen_variables", {"mode": "fermi", "n": 4}),
        ("gen_estimates", {"scope": "marginals"}),
        ("gen_estimates", {"scope": "target_connections"}),
        ("propose_requests", {"rules": ["direct", "marginals",
                                        "complements"]}),
        ("fill_requests", {}),
    ],
    # v1 + a spread battery: p10/p50/p90 for every continuous variable.
    # E[x] = 48 says nothing about whether x clears 50 — that is decided by
    # the maxent default width unless the spread is elicited explicitly.
    # 5 calls/question.
    "v1_spread": [
        ("gen_variables", {"mode": "drivers", "n": 4}),
        ("gen_estimates", {"scope": "marginals"}),
        ("gen_estimates", {"scope": "spreads"}),
        ("gen_estimates", {"scope": "target_connections"}),
        ("propose_requests", {"rules": ["direct", "marginals",
                                        "complements"]}),
        ("fill_requests", {}),
    ],
}


def serialize_state(state: State, extra: dict) -> dict:
    ests = _serialize_estimates(state.estimates)
    for d, prov in zip(ests, state.provenance):
        d["provenance"] = prov
    return {"variables": _serialize_variables(state.variables),
            "estimates": ests,
            "requests": list(state.requests),
            "node_log": state.node_log,
            **extra}


def show_entry(key: str, payload: dict) -> None:
    print(f"\n=== {key} ===")
    print(f"protocol: {payload.get('protocol')}   "
          f"nodes: {len(payload.get('node_log', []))}   "
          f"vars: {len(payload['variables'])}   "
          f"estimates: {len(payload['estimates'])}   "
          f"requests: {len(payload.get('requests', []))}")
    for row in payload.get("node_log", []):
        print(f"  [{row['node']}] +{row['added']['vars']}v "
              f"+{row['added']['estimates']}e +{row['added']['requests']}r "
              f"({row['seconds']:.0f}s, attempt {row['attempts']})")
    print(" variables:")
    for v in payload["variables"]:
        bounds = ""
        if "lower_bound" in v:
            bounds = f"  [{v['lower_bound']}, {v['upper_bound']}] {v.get('unit', '')}"
        print(f"  - {v['name']} ({v['type']}){bounds}")
    print(" estimates (by node):")
    by_node: dict[str, list] = {}
    for d in payload["estimates"]:
        by_node.setdefault(d.get("provenance", "?"), []).append(d)
    from calibrated_response.models.query import EstimateUnion
    from pydantic import TypeAdapter
    adapter = TypeAdapter(EstimateUnion)
    for node, ds in by_node.items():
        print(f"  {node}:")
        for d in ds:
            est = adapter.validate_python(
                {k: v for k, v in d.items() if k != "provenance"})
            print(f"    {est.to_query_estimate()}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", default="v1", choices=sorted(PROTOCOLS))
    ap.add_argument("--cache", default=None,
                    help="default: metaculus/caches/<protocol>/"
                         "llm_cache_<protocol>.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-resolved", action="store_true")
    ap.add_argument("--rounds", nargs="+", default=None)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--max-per-round", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2,
                    help="retries PER NODE (bounded; a node that exhausts its "
                         "retries fails the whole entry)")
    ap.add_argument("--reasoning-effort", default="low",
                    choices=["none", "minimal", "low", "medium", "high"])
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--retry-failures", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", type=int, default=None, metavar="N",
                    help="pretty-print N cached entries and exit (no LLM)")
    args = ap.parse_args(argv)

    protocol = PROTOCOLS[args.protocol]
    cache_path = Path(args.cache) if args.cache else \
        Path(__file__).parent / "caches" / args.protocol / \
        f"llm_cache_{args.protocol}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}

    if args.show is not None:
        for key in list(cache)[:args.show]:
            show_entry(key, cache[key])
        return

    meta, entries = load_dataset(args.dataset)
    entries = select_entries(entries, args)
    fail_path = Path(str(cache_path) + ".failures.json")
    failures = json.loads(fail_path.read_text(encoding="utf-8")) \
        if fail_path.exists() else {}
    if args.retry_failures:
        failures = {}

    todo = [e for e in entries
            if entry_key(e) not in cache and entry_key(e) not in failures]
    calls = n_llm_calls(protocol)
    print(f"protocol '{args.protocol}': {len(protocol)} nodes, "
          f"{calls} LLM calls/question "
          f"(worst case {calls * (args.retries + 1)} with retries)")
    print(f"selected: {len(entries)}  cached: {len(cache)}  "
          f"failed: {len(failures)}  todo: {len(todo)}  -> {cache_path.name}")
    if args.dry_run or not todo:
        return
    if args.limit is not None:
        todo = todo[:args.limit]

    reasoning = ({"enabled": False} if args.reasoning_effort == "none"
                 else {"effort": args.reasoning_effort})
    client = OpenRouterClient(model=args.model, reasoning=reasoning)

    def save_cache():
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    async def run_nodes(e):
        state = State(
            question=entry_context(e),
            variables=[BinaryVariable(
                name=TARGET_NAME,
                description=f"The literal answer to the main question: "
                            f"{entry_question(e)}")])
        for node_name, params in protocol:
            fn, needs_llm = ENRICHERS[node_name]
            qualifier = params.get('mode') or params.get('scope') or \
                ("fresh" if params.get('hide_estimates') else None)
            tag = f"{node_name}" + (f"[{qualifier}]" if qualifier else "")
            t1 = time.time()
            for attempt in range(args.retries + 1):
                try:
                    if needs_llm:
                        delta = await fn(state, client, **params)
                    else:
                        delta = fn(state, **params)
                    added = merge(state, tag, *delta)
                    state.node_log.append(
                        {"node": tag, "added": added,
                         "seconds": round(time.time() - t1, 1),
                         "attempts": attempt + 1})
                    break
                except Exception:
                    if attempt >= args.retries:
                        raise
                    await asyncio.sleep(5.0 * (attempt + 1))
        has_direct, n_coupled = target_coverage(state.estimates)
        if not has_direct:
            raise ValueError("missing direct P(target = ...) estimate")
        if n_coupled < 1:
            raise ValueError("no estimate links target to another variable")
        return state

    async def elicit_one(e, sem):
        async with sem:
            t0 = time.time()
            try:
                state = await run_nodes(e)
            except Exception as ex:
                return e, None, {"error": f"{type(ex).__name__}: {ex}",
                                 "trace": traceback.format_exc()[-1500:]}
            payload = serialize_state(state, {
                "protocol": args.protocol,
                "model": args.model,
                "elicited_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seconds": round(time.time() - t0, 1)})
            return e, payload, None

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
                      f"{len(payload['estimates'])} est, "
                      f"{len(payload['requests'])} req "
                      f"({payload['seconds']:.0f}s)")
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
    print(f"\ndone: {n_done} new in {(time.time() - t0) / 60:.1f} min, "
          f"{len(failures)} failures, cache now {len(cache)} entries")


if __name__ == "__main__":
    main()
