"""Stage 1: temporal extraction — chat few-shot + prefix KV cache + dateparser fallback."""
from __future__ import annotations

import json
import logging
from datetime import date, time, timedelta
from typing import Dict, List, Optional, Tuple

from local_models import generate_chat, generate_with_prefix, is_loaded, warmup_prefix_cache

logger = logging.getLogger(__name__)

LABEL = "temporal"

SYSTEM = (
    "You extract date and time ranges from event search queries. "
    "Given today's date and a query, first write ONE line starting with 'Reasoning: ' "
    "explaining the date math, then output ONE JSON object and nothing else. "
    'Schema: {"date_from":"YYYY-MM-DD"|null,"date_to":"YYYY-MM-DD"|null,'
    '"time_from":"HH:MM"|null,"time_to":"HH:MM"|null}. '
    "Use null when the query does not imply that field. "
    "Weekend=Sat+Sun. Tonight=today,17:00-21:00. Morning=06:00-12:00. "
    "Afternoon=12:00-17:00. Evening=17:00-21:00."
)

# 4 exemplars covering the main idiom classes that were failing
FEWSHOT: List[Tuple[str, str, str, str]] = [
    (
        "2026-04-17", "Friday", "sports games tonight",
        "Reasoning: Tonight means today 2026-04-17, evening 17:00-21:00.\n"
        '{"date_from":"2026-04-17","date_to":"2026-04-17","time_from":"17:00","time_to":"21:00"}',
    ),
    (
        "2026-04-13", "Monday", "this weekend concerts",
        "Reasoning: Next weekend from Monday 2026-04-13 is Sat 2026-04-18 and Sun 2026-04-19.\n"
        '{"date_from":"2026-04-18","date_to":"2026-04-19","time_from":null,"time_to":null}',
    ),
    (
        "2026-04-17", "Friday", "art exhibits next month",
        "Reasoning: Next month after April 2026 is May 2026, full month.\n"
        '{"date_from":"2026-05-01","date_to":"2026-05-31","time_from":null,"time_to":null}',
    ),
    (
        "2026-04-17", "Friday", "events two weeks from now",
        "Reasoning: Two weeks after 2026-04-17 is 2026-05-01.\n"
        '{"date_from":"2026-05-01","date_to":"2026-05-01","time_from":null,"time_to":null}',
    ),
]


def _prefix_messages() -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    for t, wd, q, a in FEWSHOT:
        msgs.append({"role": "user", "content": f"Today: {t} ({wd})\nQuery: {q}"})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def warmup() -> None:
    warmup_prefix_cache(LABEL, _prefix_messages())


def _build_user_content(query: str, today: date) -> str:
    return f"Today: {today.isoformat()} ({today.strftime('%A')})\nQuery: {query}"


def _coerce_date(s) -> Optional[date]:
    if s is None:
        return None
    try:
        return date.fromisoformat(str(s).strip())
    except (ValueError, TypeError):
        return None


def _coerce_time(s) -> Optional[time]:
    if s is None:
        return None
    raw = str(s).strip()
    try:
        return time.fromisoformat(raw) if raw else None
    except (ValueError, TypeError):
        return None


def _parse_temporal_json(
    raw: Optional[str],
) -> Tuple[Optional[Tuple[date, date]], Optional[Tuple[Optional[time], Optional[time]]]]:
    if not raw:
        return None, None
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        logger.info("Temporal: non-JSON output: %s", raw[:150])
        return None, None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.info("Temporal: invalid JSON: %s", raw[:150])
        return None, None
    if not isinstance(obj, dict):
        return None, None

    d_from = _coerce_date(obj.get("date_from"))
    d_to = _coerce_date(obj.get("date_to"))
    t_from = _coerce_time(obj.get("time_from"))
    t_to = _coerce_time(obj.get("time_to"))

    date_range: Optional[Tuple[date, date]] = None
    if d_from and d_to:
        date_range = (min(d_from, d_to), max(d_from, d_to))
    elif d_from:
        date_range = (d_from, d_from)
    elif d_to:
        date_range = (d_to, d_to)

    time_range: Optional[Tuple[Optional[time], Optional[time]]] = None
    if t_from or t_to:
        time_range = (t_from, t_to)

    return date_range, time_range


def _dateparser_fallback(
    query: str,
) -> Tuple[Optional[Tuple[date, date]], Optional[Tuple[Optional[time], Optional[time]]]]:
    """Catch explicit dates the model missed using dateparser + simple rules."""
    try:
        import dateparser.search
        from datetime import datetime
        today = date.today()
        q = query.lower()

        # Time-of-day rules
        time_range: Optional[Tuple[Optional[time], Optional[time]]] = None
        if any(w in q for w in ("morning", "am")):
            time_range = (time(6, 0), time(12, 0))
        elif any(w in q for w in ("afternoon",)):
            time_range = (time(12, 0), time(17, 0))
        elif any(w in q for w in ("evening",)):
            time_range = (time(17, 0), time(21, 0))
        elif any(w in q for w in ("tonight", "night")):
            time_range = (time(17, 0), time(21, 0))

        # Try dateparser for explicit dates ("April 25", "May 3rd")
        results = dateparser.search.search_dates(
            query,
            settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if results:
            parsed_dates = [r[1].date() for r in results if isinstance(r[1], datetime)]
            if parsed_dates:
                d = min(parsed_dates)
                if d >= today:
                    return (d, d), time_range

        return None, time_range if time_range else None
    except Exception:
        return None, None


def extract_temporal(
    query: str,
) -> Tuple[Optional[Tuple[date, date]], Optional[Tuple[Optional[time], Optional[time]]]]:
    """Return (date_range, time_range). Either may be None."""
    if not is_loaded():
        return _dateparser_fallback(query)

    user_content = _build_user_content(query, date.today())
    raw = generate_with_prefix(LABEL, user_content, max_new_tokens=128)
    if raw is None:
        # Prefix cache not warmed yet — fall back to full conversation
        msgs = _prefix_messages()
        msgs.append({"role": "user", "content": user_content})
        raw = generate_chat(msgs, max_new_tokens=128)

    date_range, time_range = _parse_temporal_json(raw)

    # Post-merge: if model returned nothing, try dateparser
    if date_range is None and time_range is None:
        date_range, time_range = _dateparser_fallback(query)
    elif date_range is None:
        fallback_dr, _ = _dateparser_fallback(query)
        if fallback_dr:
            date_range = fallback_dr

    return date_range, time_range
