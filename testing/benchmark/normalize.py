from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from testing.benchmark.search_compat import STOP_WORDS


def strip_json_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.removeprefix("```json").removeprefix("```")
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = strip_json_fences(text)
    if not cleaned:
        return None, "empty-response"
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, None
        return None, "response-not-object"
    except json.JSONDecodeError as exc:
        return None, f"json-decode-error: {exc}"


def normalize_keywords(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in values or []:
        token = re.sub(r"\s+", " ", str(raw).strip().lower())
        if not token:
            continue
        if token in STOP_WORDS:
            continue
        if len(token) <= 1:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _parse_date_text(value: Any) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    text = str(value).strip()
    if not text:
        return None

    # Strict first to avoid locale ambiguity.
    try:
        return datetime.fromisoformat(text[:10]).date().isoformat()
    except ValueError:
        pass

    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text.replace(",", ""), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_time_text(value: Any) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    # HH:MM 24-hour
    hhmm = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text)
    if hhmm:
        return f"{int(hhmm.group(1)):02d}:{hhmm.group(2)}"

    # H(am|pm) and H:MM(am|pm)
    ampm = re.fullmatch(r"(\d{1,2})(?::([0-5]\d))?\s*(am|pm)", text)
    if ampm:
        hour = int(ampm.group(1))
        minute = int(ampm.group(2) or "0")
        suffix = ampm.group(3)
        if hour == 12:
            hour = 0
        if suffix == "pm":
            hour += 12
        if 0 <= hour <= 23:
            return f"{hour:02d}:{minute:02d}"

    return None


def normalize_date_range(date_from: Any, date_to: Any) -> Tuple[Optional[str], Optional[str]]:
    start = _parse_date_text(date_from)
    end = _parse_date_text(date_to) if date_to not in (None, "", "null") else start
    if start and end and end < start:
        start, end = end, start
    return start, end


def normalize_time_range(time_from: Any, time_to: Any) -> Tuple[Optional[str], Optional[str]]:
    start = _parse_time_text(time_from)
    end = _parse_time_text(time_to)
    if start and end and end < start:
        # Keep as-is for potential overnight windows, but annotate via caller metadata if needed.
        return start, end
    return start, end


def normalize_from_model_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    keywords = normalize_keywords(obj.get("keywords") or [])
    date_from, date_to = normalize_date_range(obj.get("date_from"), obj.get("date_to"))
    time_from, time_to = normalize_time_range(obj.get("time_from"), obj.get("time_to"))
    return {
        "keywords": keywords,
        "date_from": date_from,
        "date_to": date_to,
        "time_from": time_from,
        "time_to": time_to,
    }
