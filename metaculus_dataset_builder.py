#!/usr/bin/env python3
"""Build a Metaculus dataset of questions and answers.

This script reuses the same API style as `metaculus_example.py`:
- Lists posts from `/api/posts/` with tournament/status filters
- Extracts question payloads from each post
- Writes JSONL records with question metadata and resolution fields

Examples:

  # Resolved questions from one tournament slug
  python metaculus_dataset_builder.py \
    --tournament fall-aib-2025 \
    --statuses resolved \
    --output data/fall_aib_2025_resolved.jsonl

  # Include open + resolved questions from multiple tournaments
  python metaculus_dataset_builder.py \
    --tournament minibench \
    --tournament spring-aib-2026 \
    --statuses open,resolved \
    --output data/mixed_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

API_BASE_URL = "https://www.metaculus.com/api"
API2_BASE_URL = "https://www.metaculus.com/api2"
DEFAULT_FORECAST_TYPES = "binary,multiple_choice,numeric,discrete,date"


@dataclass
class DatasetStats:
    n_posts_scanned: int = 0
    n_questions_written: int = 0
    n_resolved_questions: int = 0
    n_open_questions: int = 0
    n_questions_hydrated: int = 0
    n_hydration_failures: int = 0


def _auth_headers(token: str | None) -> dict[str, str]:
    if token:
        return {"Authorization": f"Token {token}"}
    return {}


def _get_json_with_retries(
    session: requests.Session,
    url: str,
    *,
    token: str | None,
    params: dict[str, Any] | None = None,
    timeout_s: int = 30,
    max_retries: int = 12,
) -> requests.Response:
    backoff_s = 5.0
    for attempt in range(max_retries):
        response = session.get(
            url,
            headers=_auth_headers(token),
            params=params,
            timeout=timeout_s,
        )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_s = max(float(retry_after), backoff_s) if retry_after else backoff_s
            time.sleep(wait_s)
            backoff_s = min(backoff_s * 2.0, 30.0)
            continue

        if response.status_code >= 500:
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)
            continue

        return response

    # Final attempt (raise on failure)
    response = session.get(
        url,
        headers=_auth_headers(token),
        params=params,
        timeout=timeout_s,
    )
    return response


def _extract_questions_from_post(post: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    # Single-question posts commonly use "question".
    if isinstance(post.get("question"), dict):
        questions.append(post["question"])

    # Group posts may include lists under "questions".
    if isinstance(post.get("questions"), list):
        questions.extend([q for q in post["questions"] if isinstance(q, dict)])

    return questions


def _question_record(post: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    qid = question.get("id")

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "post": {
            "id": post.get("id"),
            "title": post.get("title"),
            "url": f"https://www.metaculus.com/questions/{post.get('id')}/",
            "published_at": post.get("published_at"),
            "tournaments": post.get("tournaments", []),
            "status": post.get("status"),
        },
        "question": {
            "id": qid,
            "title": question.get("title"),
            "description": question.get("description"),
            "url": f"https://www.metaculus.com/questions/{qid}/" if qid else None,
            "status": question.get("status"),
            "resolve_time": question.get("resolve_time"),
            "scheduled_close_time": question.get("scheduled_close_time"),
            "scheduled_resolve_time": question.get("scheduled_resolve_time"),
            "created_time": question.get("created_time"),
            "forecast_type": question.get("type") or question.get("forecast_type"),
            "possibilities": question.get("possibilities"),
            "resolution": question.get("resolution"),
            "community_prediction": question.get("community_prediction"),
            "fine_print": question.get("fine_print"),
            "resolution_criteria": question.get("resolution_criteria"),
        },
    }


def _hydrate_question_from_api2(
    session: requests.Session,
    token: str | None,
    question_id: int,
    timeout_s: int = 30,
) -> dict[str, Any]:
    response = _get_json_with_retries(
        session,
        f"{API2_BASE_URL}/questions/{question_id}/",
        token=token,
        timeout_s=timeout_s,
    )
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    payload = response.json()
    # Some endpoints return {"question": {...}}, others return the object directly.
    if isinstance(payload, dict) and isinstance(payload.get("question"), dict):
        return payload["question"]
    return payload if isinstance(payload, dict) else {}


def _list_posts_page(
    session: requests.Session,
    token: str | None,
    *,
    tournaments: list[str],
    statuses: str,
    forecast_type: str,
    offset: int,
    limit: int,
    include_description: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": forecast_type,
        "statuses": statuses,
        "include_description": "true" if include_description else "false",
    }

    if tournaments:
        params["tournaments"] = tournaments

    response = _get_json_with_retries(
        session,
        f"{API_BASE_URL}/posts/",
        token=token,
        params=params,
        timeout_s=30,
    )
    response.raise_for_status()
    return response.json()

def build_dataset(
    *,
    output_path: Path,
    tournaments: list[str],
    statuses: str,
    forecast_type: str,
    max_posts: int,
    page_size: int,
    include_description: bool,
    token: str | None,
    hydrate_resolution: bool,
    hydration_sleep_s: float,
    require_resolution: bool,
) -> DatasetStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = DatasetStats()

    with requests.Session() as session, output_path.open("w", encoding="utf-8") as out:
        offset = 0

        while stats.n_posts_scanned < max_posts:
            page_limit = min(page_size, max_posts - stats.n_posts_scanned)
            page = _list_posts_page(
                session,
                token,
                tournaments=tournaments,
                statuses=statuses,
                forecast_type=forecast_type,
                offset=offset,
                limit=page_limit,
                include_description=include_description,
            )

            posts = page.get("results", [])
            if not posts:
                break

            for post in posts:
                stats.n_posts_scanned += 1
                for question in _extract_questions_from_post(post):
                    if hydrate_resolution and question.get("id") is not None:
                        # /api/posts often omits resolved values; fetch full question payload from /api2.
                        qid = int(question["id"])
                        details: dict[str, Any] = {}
                        had_failure = False
                        try:
                            details = _hydrate_question_from_api2(session, token, qid)
                        except requests.RequestException:
                            had_failure = True
                        if details:
                            question = {**question, **details}
                            stats.n_questions_hydrated += 1
                        elif had_failure:
                            stats.n_hydration_failures += 1
                        if hydration_sleep_s > 0:
                            time.sleep(hydration_sleep_s)

                    q_status = (question.get("status") or "").lower()
                    has_resolution = question.get("resolution") is not None
                    if q_status == "resolved" or has_resolution:
                        stats.n_resolved_questions += 1
                    elif q_status == "open":
                        stats.n_open_questions += 1

                    if require_resolution and not has_resolution:
                        continue

                    record = _question_record(post, question)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats.n_questions_written += 1

            offset += len(posts)

            # If API returned fewer rows than requested, pagination is exhausted.
            if len(posts) < page_limit:
                break

    return stats

DEFAULT_FORECAST_TYPES = "binary,multiple_choice,numeric,discrete,date"

"""
python metaculus_dataset_builder.py --tournament ""fall-aib-2025" --statuses "resolved" --forecast-type "binary,numeric" --max-posts 1000 --output data/fall_aib_2025.jsonl

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Metaculus question dataset (with resolution fields when available)."
    )
    parser.add_argument(
        "--tournament",
        action="append",
        default=[],
        help="Tournament ID/slug. Repeat for multiple tournaments.",
    )
    parser.add_argument(
        "--statuses",
        default="resolved",
        help="Comma-separated post statuses (e.g. resolved or open,resolved).",
    )
    parser.add_argument(
        "--forecast-type",
        default=DEFAULT_FORECAST_TYPES,
        help=f"Comma-separated forecast types. Default: {DEFAULT_FORECAST_TYPES}",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=1000,
        help="Maximum number of posts to scan.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of posts requested per page.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("questions.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--include-description",
        action="store_true",
        help="Request full descriptions from the API.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("METACULUS_TOKEN"),
        help="Metaculus API token. Defaults to METACULUS_TOKEN env var.",
    )
    parser.add_argument(
        "--hydrate-resolution",
        action="store_true",
        help="Fetch each question from /api2/questions/{id}/ to populate resolution/criteria fields.",
    )
    parser.add_argument(
        "--hydration-sleep-s",
        type=float,
        default=0.05,
        help="Optional delay between per-question hydration requests.",
    )
    parser.add_argument(
        "--require-resolution",
        action="store_true",
        help="Only write questions that have non-null `question.resolution`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = build_dataset(
        output_path=args.output,
        tournaments=args.tournament,
        statuses=args.statuses,
        forecast_type=args.forecast_type,
        max_posts=args.max_posts,
        page_size=args.page_size,
        include_description=args.include_description,
        token=args.token,
        hydrate_resolution=args.hydrate_resolution,
        hydration_sleep_s=args.hydration_sleep_s,
        require_resolution=args.require_resolution,
    )

    print(f"Wrote dataset to {args.output}")
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
