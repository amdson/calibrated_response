#!/usr/bin/env python3
"""
Scrape a ~500-question Metaculus dataset suitable for calibration experiments.

Outputs:
  - questions.jsonl : one record per question, incl metadata + prediction history
  - questions_meta.csv : flattened metadata for quick inspection
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = "https://www.metaculus.com"
QUESTIONS_ENDPOINT = f"{BASE_URL}/api2/questions/"
PRED_HISTORY_ENDPOINT = f"{BASE_URL}/api2/questions/{{id}}/prediction-history/"


# ----------------------------
# Inclusion criteria defaults
# ----------------------------
TARGET_N = 500
RANDOM_SEED = 7

# Pull this many question summaries before filtering.
# If your filters are strict, bump this (e.g., 15000).
CRAWL_LIMIT = 8000

# Forecast-history quality gates
MIN_HISTORY_POINTS = 30
MIN_SPAN_DAYS = 30

# Optional: include these possibility types
ALLOWED_TYPES = {"numeric", "date"}  # add "binary" if desired


@dataclass
class FetchConfig:
    api_token: Optional[str] = None  # if you have one; many endpoints work without auth
    timeout_s: int = 30
    sleep_s: float = 0.2  # light politeness delay
    user_agent: str = "metaculus-calibration-scraper/0.1"


def _headers(cfg: FetchConfig) -> Dict[str, str]:
    h = {"User-Agent": cfg.user_agent}
    # Metaculus API auth is used for some use cases; keep optional.
    if cfg.api_token:
        h["Authorization"] = f"Token {cfg.api_token}"
    return h


def _parse_dt(s: str) -> Optional[datetime]:
    """Parse ISO-ish timestamps found in API responses."""
    if not s:
        return None
    # Metaculus commonly uses Zulu timestamps with optional fractional seconds.
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    # Fallback: let fromisoformat try (handles offsets)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def list_questions_page(
    session: requests.Session,
    cfg: FetchConfig,
    *,
    page: Optional[int] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    status: str = "resolved",
    order_by: str = "-last_prediction_time",
) -> Dict[str, Any]:
    """
    Fetch a single paginated page from /api2/questions/.

    Metaculus uses Django REST Framework-style pagination; different deployments support
    different params. This function tries a common set: page, limit/offset.
    """
    params: Dict[str, Any] = {"status": status, "order_by": order_by}

    # Try 'page' pagination first (often works)
    if page is not None:
        params["page"] = page

    # Also include limit/offset if provided (some configs prefer this)
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset

    r = session.get(QUESTIONS_ENDPOINT, headers=_headers(cfg), params=params, timeout=cfg.timeout_s)
    r.raise_for_status()
    time.sleep(cfg.sleep_s)
    return r.json()


def get_prediction_history(
    session: requests.Session,
    cfg: FetchConfig,
    qid: int,
) -> Dict[str, Any]:
    url = PRED_HISTORY_ENDPOINT.format(id=qid)
    r = session.get(url, headers=_headers(cfg), timeout=cfg.timeout_s)
    r.raise_for_status()
    time.sleep(cfg.sleep_s)
    return r.json()


def question_possibility_type(q: Dict[str, Any]) -> Optional[str]:
    # Common location: q["possibilities"]["type"]
    t = _safe_get(q, ["possibilities", "type"])
    if isinstance(t, str):
        return t.lower()
    return None


def question_is_resolved(q: Dict[str, Any]) -> bool:
    # Common: q["status"] == "resolved"
    st = q.get("status")
    if isinstance(st, str) and st.lower() == "resolved":
        return True
    # Fallback: treat presence of resolution as resolved-ish
    return q.get("resolution") is not None or q.get("resolve_time") is not None


def looks_conditional_or_cancelled(q: Dict[str, Any]) -> bool:
    """
    Heuristic exclusions. Metaculus has multiple flags over time; we check a few common ones
    and also scan title for "conditional" markers.
    """
    title = (q.get("title") or "").lower()

    # Flags sometimes present
    for k in ("is_conditional", "conditional", "cancelled", "canceled", "annulled"):
        v = q.get(k)
        if isinstance(v, bool) and v:
            return True

    # Title heuristics
    bad_markers = ["conditional on", "if and only if", "if ", "unless ", "cancelled", "canceled", "annulled"]
    if any(m in title for m in bad_markers):
        # note: "if " is broad; you can remove it if it over-filters
        return True

    return False


def history_stats(hist: Dict[str, Any]) -> Tuple[int, Optional[datetime], Optional[datetime]]:
    """
    Try to extract timeseries points from prediction-history response.
    The schema can vary; we attempt common shapes.
    """
    # Common shapes seen historically:
    # - {"prediction_timeseries": [{"t": <unix seconds>, "community_prediction": ...}, ...], ...}
    # - {"history": [{"time": "...", "community_prediction": ...}, ...], ...}
    series = None

    if isinstance(hist.get("prediction_timeseries"), list):
        series = hist["prediction_timeseries"]
        # time in unix seconds in key 't'
        times = []
        for p in series:
            t = p.get("t")
            if isinstance(t, (int, float)):
                times.append(datetime.fromtimestamp(t, tz=timezone.utc))
        if times:
            return len(series), min(times), max(times)
        return len(series), None, None

    if isinstance(hist.get("history"), list):
        series = hist["history"]
        times = []
        for p in series:
            ts = p.get("time") or p.get("timestamp") or p.get("t")
            if isinstance(ts, str):
                dt = _parse_dt(ts)
                if dt:
                    times.append(dt)
            elif isinstance(ts, (int, float)):
                times.append(datetime.fromtimestamp(ts, tz=timezone.utc))
        if times:
            return len(series), min(times), max(times)
        return len(series), None, None

    # If nothing matches, treat as empty
    return 0, None, None


def passes_filters(q: Dict[str, Any], hist: Dict[str, Any]) -> bool:
    if not question_is_resolved(q):
        return False

    ptype = question_possibility_type(q)
    if ptype not in ALLOWED_TYPES:
        return False

    if looks_conditional_or_cancelled(q):
        return False

    n_pts, t0, t1 = history_stats(hist)
    if n_pts < MIN_HISTORY_POINTS:
        return False

    if t0 and t1:
        span_days = (t1 - t0).total_seconds() / 86400.0
        if span_days < MIN_SPAN_DAYS:
            return False

    return True


def main() -> None:
    cfg = FetchConfig(api_token=os.getenv("METACULUS_TOKEN"))
    random.seed(RANDOM_SEED)

    session = requests.Session()

    # 1) Crawl question summaries
    questions: List[Dict[str, Any]] = []
    page = 1
    while len(questions) < CRAWL_LIMIT:
        data = list_questions_page(session, cfg, page=page, status="resolved")
        results = data.get("results") or []
        if not results:
            break
        questions.extend(results)
        page += 1

        # Stop if API indicates no next page
        if not data.get("next"):
            break

    print(f"Crawled {len(questions)} resolved question summaries.")

    # 2) Fetch prediction histories + filter
    kept: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        qid = q.get("id")
        if not isinstance(qid, int):
            continue

        try:
            hist = get_prediction_history(session, cfg, qid)
        except requests.HTTPError as e:
            # Some questions may have restricted endpoints; skip
            print(f"Skipping qid={qid} (history fetch failed): {e}")
            continue

        if passes_filters(q, hist):
            kept.append({"question": q, "prediction_history": hist})

        if i % 200 == 0:
            print(f"Processed {i}/{len(questions)}; kept so far: {len(kept)}")

        if len(kept) >= TARGET_N * 3:
            # You can stop early once you have a big pool to sample from
            break

    if len(kept) < TARGET_N:
        print(f"WARNING: only found {len(kept)} passing filters; consider relaxing thresholds.")
        sample = kept
    else:
        sample = random.sample(kept, k=TARGET_N)

    # 3) Write outputs
    with open("questions.jsonl", "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write a quick CSV for inspection
    with open("questions_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "title",
                "type",
                "status",
                "created_time",
                "close_time",
                "resolve_time",
                "n_history_points",
                "history_span_days",
            ],
        )
        w.writeheader()
        for rec in sample:
            q = rec["question"]
            hist = rec["prediction_history"]
            n_pts, t0, t1 = history_stats(hist)
            span_days = ""
            if t0 and t1:
                span_days = (t1 - t0).total_seconds() / 86400.0
            w.writerow(
                {
                    "id": q.get("id"),
                    "title": q.get("title"),
                    "type": question_possibility_type(q),
                    "status": q.get("status"),
                    "created_time": q.get("created_time"),
                    "close_time": q.get("close_time"),
                    "resolve_time": q.get("resolve_time"),
                    "n_history_points": n_pts,
                    "history_span_days": span_days,
                }
            )

    print(f"Wrote {len(sample)} questions to questions.jsonl and questions_meta.csv")


if __name__ == "__main__":
    main()
