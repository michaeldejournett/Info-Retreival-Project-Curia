#!/usr/bin/env python3
"""Search scraped UNL events by keyword, with optional Gemini keyword expansion and date filtering."""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, date, time
from typing import Any, Dict, List, Optional, Tuple

import requests
from google import genai

try:
    from .prompt_templates import build_expand_prompt
except ImportError:
    from prompt_templates import build_expand_prompt

EVENTS_FILE = "scraped/events.json"
DEFAULT_TOP_N = 10
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemma-3-27b-it")

FIELD_WEIGHTS = {
    "title":       4,
    "group":       3,
    "description": 2,
    "location":    1,
    "audience":    1,
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "be", "i", "me", "my", "this",
    "that", "it", "do", "want", "find", "looking", "something", "events",
    "event", "any", "some", "what", "show", "get", "can", "will", "like",
    "go", "going", "around", "near", "about", "up", "out", "next", "this",
    "weekend", "today", "tomorrow", "tonight", "week",
    # time-of-day words — these should become time filters, not keywords
    "morning", "afternoon", "evening", "night", "midnight", "noon",
    "late", "early", "pm", "am", "oclock",
}

def load_events(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["events"]


def base_terms(query: str) -> List[str]:
    """Split query into meaningful words, stripping stop words."""
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def expand_with_gemini(query: str, model: str) -> Tuple[Optional[List[str]], Optional[Tuple[date, date]], Optional[Tuple[Optional[time], Optional[time]]]]:
    """Call Gemini API to extract/expand keywords and resolve date/time references.
    Returns (keywords, date_range, time_range) — any can be None if unavailable."""
    if not GEMINI_API_KEY:
        return None, None, None
    try:
        now = datetime.now()
        prompt = build_expand_prompt(query=query, now=now)
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        # Strip markdown code fences if model wraps response
        text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        keywords = [str(k).lower() for k in (data.get("keywords") or []) if k]

        date_range = None
        df = data.get("date_from")
        dt = data.get("date_to")
        if df and df != "null":
            try:
                start = date.fromisoformat(str(df))
                end = date.fromisoformat(str(dt)) if dt and dt != "null" else start
                date_range = (start, end)
            except ValueError:
                pass

        time_range = None
        tf = data.get("time_from")
        tt = data.get("time_to")
        if (tf and tf != "null") or (tt and tt != "null"):
            try:
                t_start = time.fromisoformat(str(tf)) if tf and tf != "null" else None
                t_end   = time.fromisoformat(str(tt)) if tt and tt != "null" else None
                time_range = (t_start, t_end)
            except ValueError:
                pass

        return keywords, date_range, time_range
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("expand_with_gemini failed: %s", exc)
        return None, None, None


def expand_with_local(
    query: str,
) -> Tuple[Optional[List[str]], Optional[Tuple[date, date]], Optional[Tuple[Optional[time], Optional[time]]]]:
    """Local two-stage expansion: Stage 1 temporal + Stage 2 keyword expansion."""
    try:
        from local_models import is_loaded
        from temporal import extract_temporal
        from expansion import expand_keywords
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Local model modules unavailable: %s", exc)
        return None, None, None

    if not is_loaded():
        return None, None, None

    keywords = expand_keywords(query)
    date_range, time_range = extract_temporal(query)
    return keywords, date_range, time_range


def expand_query(
    query: str, gemini_model: str,
) -> Tuple[Optional[List[str]], Optional[Tuple[date, date]], Optional[Tuple[Optional[time], Optional[time]]], str]:
    """Dispatcher: route to local stack or Gemini based on USE_LOCAL_MODELS.
    Returns (keywords, date_range, time_range, backend_used) where backend_used is
    one of "local", "gemini", or "none"."""
    use_local = os.environ.get("USE_LOCAL_MODELS", "false").lower() in ("1", "true", "yes")
    fallback = os.environ.get("LOCAL_MODEL_FALLBACK_TO_GEMINI", "true").lower() in ("1", "true", "yes")

    if use_local:
        kw, dr, tr = expand_with_local(query)
        if kw is not None or dr is not None or tr is not None:
            return kw, dr, tr, "local"
        if not fallback:
            return None, None, None, "none"

    kw, dr, tr = expand_with_gemini(query, gemini_model)
    if kw is not None or dr is not None or tr is not None:
        return kw, dr, tr, "gemini"
    return None, None, None, "none"


def extract_date_range(query: str) -> Optional[Tuple[date, date]]:
    """Parse a date range from natural language in the query."""
    import dateparser.search

    q = query.lower()
    today = date.today()

    if "tonight" in q:
        return (today, today)
    if "tomorrow" in q:
        d = today + timedelta(days=1)
        return (d, d)
    if "this weekend" in q or "weekend" in q:
        days_to_sat = (5 - today.weekday()) % 7 or 7
        sat = today + timedelta(days=days_to_sat)
        return (sat, sat + timedelta(days=1))
    if "next week" in q:
        mon = today + timedelta(days=(7 - today.weekday()))
        return (mon, mon + timedelta(days=6))

    try:
        results = dateparser.search.search_dates(query, languages=["en"])
        if results:
            found = results[0][1].date()
            return (found, found)
    except Exception:
        pass

    return None


def filter_by_date(
    events: List[Dict[str, Any]],
    date_range: Tuple[date, date],
    time_range: Optional[Tuple[Optional[time], Optional[time]]] = None,
) -> List[Dict[str, Any]]:
    start_d, end_d = date_range
    t_start, t_end = time_range if time_range else (None, None)
    out = []
    for e in events:
        raw = (e.get("start") or "")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw[:25])
            if not (start_d <= dt.date() <= end_d):
                continue
            if t_start is not None and dt.time() < t_start:
                continue
            if t_end is not None and dt.time() > t_end:
                continue
            out.append(e)
        except ValueError:
            continue
    return out


def filter_by_time(
    events: List[Dict[str, Any]],
    time_range: Tuple[Optional[time], Optional[time]],
) -> List[Dict[str, Any]]:
    t_start, t_end = time_range
    out = []
    for e in events:
        raw = (e.get("start") or "")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw[:25])
            if t_start is not None and dt.time() < t_start:
                continue
            if t_end is not None and dt.time() > t_end:
                continue
            out.append(e)
        except ValueError:
            continue
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
    scored = [(s, e) for s, e in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Search UNL events by keyword.")
    parser.add_argument("query", nargs="+", help="Natural-language search query")
    parser.add_argument("--events", default=EVENTS_FILE)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, metavar="N")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model for keyword expansion (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM expansion, use raw keywords only",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "gemini", "local"],
        default="auto",
        help="Expansion backend: 'auto' honors USE_LOCAL_MODELS env, 'gemini' or 'local' force one.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    query = " ".join(args.query)
    terms = base_terms(query)

    llm_date_range = None
    llm_time_range = None
    if not args.no_llm:
        if args.backend == "local":
            from local_models import load_local_models
            load_local_models()
            print("Expanding with local model …", file=sys.stderr)
            llm_keywords, llm_date_range, llm_time_range = expand_with_local(query)
        elif args.backend == "gemini":
            print(f"Expanding with Gemini ({args.model}) …", file=sys.stderr)
            llm_keywords, llm_date_range, llm_time_range = expand_with_gemini(query, args.model)
        else:
            use_local = os.environ.get("USE_LOCAL_MODELS", "false").lower() in ("1", "true", "yes")
            if use_local:
                from local_models import load_local_models
                load_local_models()
            print(f"Expanding ({'local' if use_local else 'gemini'}) …", file=sys.stderr)
            llm_keywords, llm_date_range, llm_time_range, _backend = expand_query(query, args.model)
        if llm_keywords:
            llm_keywords = [k for k in llm_keywords if k not in STOP_WORDS and len(k) > 1]
            print(f"  LLM keywords : {llm_keywords}", file=sys.stderr)
            if llm_date_range:
                print(f"  LLM dates    : {llm_date_range[0]} → {llm_date_range[1]}", file=sys.stderr)
            if llm_time_range:
                print(f"  LLM times    : {llm_time_range[0]} → {llm_time_range[1]}", file=sys.stderr)
            seen = set(terms)
            for kw in llm_keywords:
                if kw not in seen:
                    terms.append(kw)
                    seen.add(kw)
        else:
            print("  LLM unavailable — falling back to raw keywords.", file=sys.stderr)

    if not terms:
        print("No search terms found in query.", file=sys.stderr)
        return 1

    print(f"Terms          : {terms}", file=sys.stderr)

    events = load_events(args.events)
    date_range = llm_date_range or extract_date_range(query)
    time_range = llm_time_range
    if date_range:
        print(f"Date filter    : {date_range[0]} → {date_range[1]}", file=sys.stderr)
        if time_range:
            print(f"Time filter    : {time_range[0]} → {time_range[1]}", file=sys.stderr)
        events = filter_by_date(events, date_range, time_range)
        print(f"Events in range: {len(events)}", file=sys.stderr)

    results = search(events, terms, args.top)

    if not results:
        print("No matching events found.", file=sys.stderr)
        return 0

    if args.as_json:
        print(json.dumps(
            [{"score": s, "url": e["url"], "title": e["title"], "start": e.get("start")}
             for s, e in results],
            indent=2, ensure_ascii=False,
        ))
    else:
        print(f"\nTop {len(results)} results:", file=sys.stderr)
        for score, event in results:
            start = (event.get("start") or "")[:16].replace("T", " ")
            print(f"  [{score:3d}]  {event['url']}")
            print(f"         {event['title']}  —  {start}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
