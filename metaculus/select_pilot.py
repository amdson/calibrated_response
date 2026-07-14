"""Select a stratified pilot set and pin it to a reproducible ids file.

Stratification: resolved entries only, spread evenly across rounds, and
round-robin across sources within each round (so no single source or time
slice dominates). The output file contains one ``id@round`` key per line —
exact entries, immune to later dataset refreshes growing the resolved set.

Usage
-----
    python metaculus/select_pilot.py --n 150                 # writes data/pilot_ids.txt
    python metaculus/run_elicitation.py --dataset metaculus/data/full_dataset.json \
        --ids-file metaculus/data/pilot_ids.txt --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def select(entries: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    resolved = [e for e in entries if e.get("resolved")]

    by_round: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in resolved:
        by_round[e["round_due_date"]][e.get("source", "?")].append(e)

    rounds = sorted(by_round)
    quota, extra = divmod(n, len(rounds))
    picked = []
    for i, r in enumerate(rounds):
        want = quota + (1 if i < extra else 0)
        # round-robin across sources within the round
        pools = {s: rng.sample(v, len(v)) for s, v in by_round[r].items()}
        order = sorted(pools)
        rng.shuffle(order)
        k = 0
        while want > 0 and any(pools.values()):
            s = order[k % len(order)]
            k += 1
            if pools[s]:
                picked.append(pools[s].pop())
                want -= 1
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path(__file__).parent
                                             / "data" / "full_dataset.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent
                                         / "data" / "pilot_ids.txt"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    entries = data["questions"] if isinstance(data, dict) else data
    picked = select(entries, args.n, args.seed)

    keys = [f"{e['id']}@{e['round_due_date']}" for e in picked]
    Path(args.out).write_text("\n".join(keys) + "\n", encoding="utf-8")

    print(f"wrote {len(keys)} entries to {args.out}")
    print("by round: ", dict(Counter(e["round_due_date"] for e in picked)))
    print("by source:", dict(Counter(e["source"] for e in picked)))
    print("outcomes: ", dict(Counter(round(e["resolved_to"]) for e in picked)))


if __name__ == "__main__":
    main()
