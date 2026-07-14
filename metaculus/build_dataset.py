"""Build the full forecasting dataset from ForecastBench rounds.

ForecastBench (github.com/forecastingresearch/forecastbench-datasets) publishes
a ~500-question set every two weeks (half prediction-market questions with the
market value frozen at the due date, half real-world data-series questions) and
continuously-updated resolution sets. This script downloads every round whose
forecast_due_date is on/after ``--cutoff`` (the LLM's knowledge-cutoff guard),
joins questions with their resolutions, and emits one merged dataset in the
same per-entry schema as ``tiny_dataset.json``.

Contamination protocol
----------------------
- ``--cutoff`` must be >= the elicitation LLM's training cutoff. Every entry's
  ``freeze_datetime`` / round due date is checked against it.
- Unresolved questions are INCLUDED (``resolved: false``): eliciting on them
  now and scoring when they resolve is the cleanest possible protocol.
- Re-run with ``--refresh`` to re-download resolution sets and update
  ``resolved`` / ``resolved_to`` in place; entry ids and question text are
  stable under refresh (the earliest resolved horizon is kept, and unresolved
  entries use their first future resolution date for the template).

Usage
-----
    python metaculus/build_dataset.py --cutoff 2026-02-01
    python metaculus/build_dataset.py --cutoff 2026-02-01 --refresh
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from collections import Counter
from pathlib import Path

import requests

REPO = "forecastingresearch/forecastbench-datasets"
API = f"https://api.github.com/repos/{REPO}/contents/datasets"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/datasets"
HEADERS = {"User-Agent": "calibrated-response-dataset-builder"}

ROUNDS_DIR = Path(__file__).parent.parent / "data" / "forecast_data" / "rounds"
MARKET_SOURCES = {"metaculus", "manifold", "polymarket", "infer"}


def _get_json(url: str, timeout: int = 120):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def list_round_dates() -> list[str]:
    """Round dates (YYYY-MM-DD) that have an llm question set."""
    items = _get_json(f"{API}/question_sets")
    dates = []
    for it in items:
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-llm\.json", it["name"])
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def fetch_round(date: str, refresh: bool = False):
    """(question_set, resolution_set_or_None) for one round, disk-cached.

    Question sets are immutable -> cached forever. Resolution sets grow over
    time -> re-downloaded when ``refresh`` (or missing). The newest rounds may
    have no resolution set yet.
    """
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    qpath = ROUNDS_DIR / f"{date}-llm.json"
    rpath = ROUNDS_DIR / f"{date}_resolution_set.json"

    if qpath.exists():
        qs = json.loads(qpath.read_text(encoding="utf-8"))
    else:
        qs = _get_json(f"{RAW}/question_sets/{date}-llm.json")
        qpath.write_text(json.dumps(qs), encoding="utf-8")

    rs = None
    if rpath.exists() and not refresh:
        rs = json.loads(rpath.read_text(encoding="utf-8"))
    else:
        try:
            rs = _get_json(f"{RAW}/resolution_sets/{date}_resolution_set.json")
            rpath.write_text(json.dumps(rs), encoding="utf-8")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                rs = json.loads(rpath.read_text(encoding="utf-8")) \
                    if rpath.exists() else None
            else:
                raise
    return qs, rs


def _first_future_date(q: dict, due: str):
    """First scheduled resolution date after the round due date, for the
    question-text template of an unresolved entry."""
    dates = q.get("resolution_dates")
    if isinstance(dates, list) and dates:
        future = sorted(d for d in dates if str(d) > due)
        return future[0] if future else sorted(dates)[-1]
    return None


def build_entries(date: str, qs: dict, rs: dict | None) -> list[dict]:
    """One entry per question id: original fields + resolution + round tags.

    Resolution policy: the EARLIEST resolved horizon (stable under refresh —
    later refreshes never change an already-resolved entry). Multi-horizon
    data-series resolutions beyond the first are dropped rather than emitted
    as extra (heavily correlated) data points.
    """
    res_by_id: dict[str, list[dict]] = {}
    for r in (rs or {}).get("resolutions", []):
        if isinstance(r.get("id"), str) and r.get("resolved"):
            res_by_id.setdefault(r["id"], []).append(r)

    due = qs.get("forecast_due_date", date)
    entries = []
    for q in qs["questions"]:
        if not isinstance(q.get("id"), str):
            continue                                   # combo questions
        e = dict(q)
        e["round_due_date"] = due
        rlist = sorted(res_by_id.get(q["id"], []),
                       key=lambda r: str(r["resolution_date"]))
        if rlist:
            r = rlist[0]
            e["resolved"] = True
            e["resolution_date"] = r["resolution_date"]
            e["resolved_to"] = float(r["resolved_to"])
        else:
            e["resolved"] = False
            e["resolution_date"] = _first_future_date(q, due)
            e["resolved_to"] = None
        entries.append(e)
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True,
                    help="YYYY-MM-DD; only rounds with forecast_due_date >= "
                         "cutoff are used. Must be >= the elicitation LLM's "
                         "training cutoff.")
    ap.add_argument("--out", default=str(Path(__file__).parent
                                         / "data" / "full_dataset.json"))
    ap.add_argument("--refresh", action="store_true",
                    help="re-download resolution sets (pick up new resolutions)")
    args = ap.parse_args(argv)

    dates = [d for d in list_round_dates() if d >= args.cutoff]
    if not dates:
        raise SystemExit(f"no rounds on/after {args.cutoff}")
    print(f"{len(dates)} rounds on/after {args.cutoff}: {dates[0]} .. {dates[-1]}")

    all_entries = []
    for d in dates:
        qs, rs = fetch_round(d, refresh=args.refresh)
        assert qs.get("forecast_due_date", d) >= args.cutoff
        entries = build_entries(d, qs, rs)
        n_res = sum(e["resolved"] for e in entries)
        print(f"  {d}: {len(entries)} questions, {n_res} resolved"
              f"{'' if rs else '  (no resolution set yet)'}")
        all_entries.extend(entries)

    resolved = [e for e in all_entries if e["resolved"]]
    out = {
        "meta": {
            "source": f"github.com/{REPO}",
            "cutoff": args.cutoff,
            "rounds": dates,
            "retrieved_at": datetime.date.today().isoformat(),
            "n_questions": len(all_entries),
            "n_resolved": len(resolved),
        },
        "questions": all_entries,
    }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\nwrote {args.out}")
    print(f"total: {len(all_entries)} questions, {len(resolved)} resolved "
          f"({Counter(round(e['resolved_to']) for e in resolved)})")
    print("by source:", dict(Counter(e['source'] for e in all_entries)))
    print("resolved by source:", dict(Counter(e['source'] for e in resolved)))


if __name__ == "__main__":
    main()
