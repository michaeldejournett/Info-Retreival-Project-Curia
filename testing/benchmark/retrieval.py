from __future__ import annotations

from datetime import date, time
from typing import Dict, List, Optional, Tuple

from testing.benchmark.search_compat import filter_by_date, filter_by_time, load_events, search
from testing.benchmark.schemas import ModelInvocationResult


def load_event_corpus(path: str) -> List[Dict[str, object]]:
    return load_events(path)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_time(value: Optional[str]) -> Optional[time]:
    if not value:
        return None
    try:
        return time.fromisoformat(value)
    except ValueError:
        return None


def _date_range(inv: ModelInvocationResult) -> Optional[Tuple[date, date]]:
    start = _parse_date(inv.date_from)
    end = _parse_date(inv.date_to) or start
    if start and end:
        if end < start:
            start, end = end, start
        return start, end
    return None


def _time_range(inv: ModelInvocationResult) -> Optional[Tuple[Optional[time], Optional[time]]]:
    start = _parse_time(inv.time_from)
    end = _parse_time(inv.time_to)
    if start or end:
        return start, end
    return None


def retrieve_ranked_urls(
    events: List[Dict[str, object]],
    invocation: ModelInvocationResult,
    top_n: int = 10,
) -> List[str]:
    pool = list(events)

    date_range = _date_range(invocation)
    time_range = _time_range(invocation)

    if date_range:
        pool = filter_by_date(pool, date_range, time_range)
    elif time_range:
        pool = filter_by_time(pool, time_range)

    if invocation.keywords:
        ranked = search(pool, invocation.keywords, top_n)
        urls = [str(event.get("url")) for _, event in ranked if event.get("url")]
        return urls

    return [str(event.get("url")) for event in pool[:top_n] if event.get("url")]
