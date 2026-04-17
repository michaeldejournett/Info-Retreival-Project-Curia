from __future__ import annotations

import ast
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


def _try_parse_object(candidate: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidate = (candidate or "").strip()
    if not candidate:
        return None, "empty-response"

    json_error: Optional[str] = None
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, None
        return None, "response-not-object"
    except json.JSONDecodeError as exc:
        json_error = f"json-decode-error: {exc}"

    # Many local models emit Python-like dict literals with single quotes.
    try:
        obj = ast.literal_eval(candidate)
    except Exception:
        return None, json_error

    if isinstance(obj, dict):
        return obj, None
    return None, "response-not-object"


def _extract_fenced_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text or "", flags=re.IGNORECASE):
        block = (match.group(1) or "").strip()
        if block:
            candidates.append(block)
    return candidates


def _extract_braced_candidates(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out

    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1

    return out


def _object_quality_score(obj: Dict[str, Any]) -> int:
    keys = {str(k).lower() for k in obj.keys()}
    expected = {"keywords", "date_from", "date_to", "time_from", "time_to"}
    score = len(keys.intersection(expected)) * 2

    keywords = obj.get("keywords")
    if isinstance(keywords, list):
        score += 2
        score += min(3, sum(1 for x in keywords if str(x).strip()))

    joined = " ".join(str(v) for v in obj.values())
    lowered = joined.lower()
    if "yyyy-mm-dd or null" in lowered or "hh:mm or null" in lowered:
        score -= 6
    if "..." in lowered:
        score -= 3

    return score


def _extract_labeled_value(text: str, label_pattern: str) -> Optional[str]:
    key_boundary = r"(?:keywords?|date[_\s-]*from|date[_\s-]*to|time[_\s-]*from|time[_\s-]*to)"
    pattern = re.compile(
        rf"\b{label_pattern}\b\s*[:=]\s*(.+?)(?=\b{key_boundary}\b\s*[:=]|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text or "")
    if not match:
        return None
    value = (match.group(1) or "").strip()
    return value or None


def _coerce_nullable_scalar(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    scalar = re.sub(r"\s+", " ", value).strip().strip(",;.")
    if scalar and scalar[0] == scalar[-1] and scalar[0] in {'"', "'"}:
        scalar = scalar[1:-1].strip()
    if scalar.lower() in {"", "null", "none", "n/a", "na"}:
        return None
    return scalar


def _coerce_keyword_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []

    bracket_match = re.search(r"\[[\s\S]*?\]", value)
    if bracket_match:
        candidate = bracket_match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]

    compact = re.sub(r"\s+", " ", value).strip().strip(",;.")
    if not compact:
        return []
    if compact and compact[0] == compact[-1] and compact[0] in {'"', "'"}:
        compact = compact[1:-1].strip()
    if compact.lower() in {"null", "none"}:
        return []

    parts = [p.strip(" \t\r\n\"'.,;") for p in re.split(r"[,|;]", compact)]
    return [p for p in parts if p]


def _parse_labeled_fallback(text: str) -> Optional[Dict[str, Any]]:
    keywords_raw = _extract_labeled_value(text, r"keywords?")
    date_from_raw = _extract_labeled_value(text, r"date[_\s-]*from")
    date_to_raw = _extract_labeled_value(text, r"date[_\s-]*to")
    time_from_raw = _extract_labeled_value(text, r"time[_\s-]*from")
    time_to_raw = _extract_labeled_value(text, r"time[_\s-]*to")

    parsed = {
        "keywords": _coerce_keyword_list(keywords_raw),
        "date_from": _coerce_nullable_scalar(date_from_raw),
        "date_to": _coerce_nullable_scalar(date_to_raw),
        "time_from": _coerce_nullable_scalar(time_from_raw),
        "time_to": _coerce_nullable_scalar(time_to_raw),
    }

    if parsed["keywords"] or parsed["date_from"] or parsed["date_to"] or parsed["time_from"] or parsed["time_to"]:
        return parsed
    return None


def parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = strip_json_fences(text)
    if not cleaned:
        return None, "empty-response"

    candidates: List[str] = [cleaned]
    candidates.extend(_extract_fenced_candidates(text))
    candidates.extend(_extract_braced_candidates(cleaned))

    best_obj: Optional[Dict[str, Any]] = None
    best_score = -10_000
    last_error: Optional[str] = None
    seen = set()

    for candidate in candidates:
        token = candidate.strip()
        if not token or token in seen:
            continue
        seen.add(token)

        obj, err = _try_parse_object(token)
        if obj is None:
            if err:
                last_error = err
            continue

        score = _object_quality_score(obj)
        if score > best_score:
            best_score = score
            best_obj = obj

    if best_obj is not None:
        return best_obj, None

    fallback = _parse_labeled_fallback(cleaned)
    if fallback is not None:
        return fallback, None

    return None, last_error or "response-not-object"


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
