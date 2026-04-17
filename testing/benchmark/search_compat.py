from __future__ import annotations

import json
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FIELD_WEIGHTS = {
    "title": 4,
    "group": 3,
    "description": 2,
    "location": 1,
    "audience": 1,
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "be", "i", "me", "my", "this",
    "that", "it", "do", "want", "find", "looking", "something", "events",
    "event", "any", "some", "what", "show", "get", "can", "will", "like",
    "go", "going", "around", "near", "about", "up", "out", "next", "this",
    "weekend", "today", "tomorrow", "tonight", "week", "morning", "afternoon",
    "evening", "night", "midnight", "noon", "late", "early", "pm", "am", "oclock",
}


def base_terms(query: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def load_events(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return payload["events"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported events corpus format")


def filter_by_date(
    events: List[Dict[str, Any]],
    date_range: Tuple[date, date],
    time_range: Optional[Tuple[Optional[time], Optional[time]]] = None,
) -> List[Dict[str, Any]]:
    start_d, end_d = date_range
    t_start, t_end = time_range if time_range else (None, None)
    out: List[Dict[str, Any]] = []

    for event in events:
        raw = str(event.get("start") or "")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw[:25])
        except ValueError:
            continue

        if not (start_d <= dt.date() <= end_d):
            continue
        if t_start is not None and dt.time() < t_start:
            continue
        if t_end is not None and dt.time() > t_end:
            continue
        out.append(event)

    return out


def filter_by_time(
    events: List[Dict[str, Any]],
    time_range: Tuple[Optional[time], Optional[time]],
) -> List[Dict[str, Any]]:
    t_start, t_end = time_range
    out: List[Dict[str, Any]] = []

    for event in events:
        raw = str(event.get("start") or "")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw[:25])
        except ValueError:
            continue

        if t_start is not None and dt.time() < t_start:
            continue
        if t_end is not None and dt.time() > t_end:
            continue
        out.append(event)

    return out


def score_event(event: Dict[str, Any], terms: List[str]) -> int:
    score = 0
    for field, weight in FIELD_WEIGHTS.items():
        value = event.get(field)
        if not value:
            continue
        text = " ".join(value).lower() if isinstance(value, list) else str(value).lower()
        for term in terms:
            if term in text:
                score += weight
    return score


def search(
    events: List[Dict[str, Any]],
    terms: List[str],
    top_n: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    scored = [(score_event(e, terms), e) for e in events]
    scored = [(score, event) for score, event in scored if score > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]
