#!/usr/bin/env python3
"""UNL Events Search — FastAPI microservice with periodic re-scraping."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from scraper import cross_dedupe, enrich_events, scrape_engage, scrape_rss
from search import (
    STOP_WORDS,
    base_terms,
    expand_query,
    extract_date_range,
    filter_by_date,
    filter_by_time,
    load_events,
    search,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

EVENTS_FILE = os.environ.get("EVENTS_FILE", "scraped/events.json")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemma-3-27b-it")
SCRAPE_INTERVAL = int(os.environ.get("SCRAPE_INTERVAL", "3600"))
SCRAPE_WORKERS = int(os.environ.get("SCRAPE_WORKERS", "10"))
USE_LOCAL_MODELS = os.environ.get("USE_LOCAL_MODELS", "false").lower() in ("1", "true", "yes")

_events: List[Dict[str, Any]] = []
_last_scraped: Optional[datetime] = None
_scrape_running = False


def _full_scrape() -> List[Dict[str, Any]]:
    """Blocking full-pipeline scrape. Runs in a thread executor."""
    print("Scraping UNL RSS …")
    events = scrape_rss()
    print(f"  RSS: {len(events)} events — enriching …")
    events = enrich_events(events, workers=SCRAPE_WORKERS)
    print("  Fetching Engage events …")
    try:
        engage = scrape_engage()
        before = len(events)
        events = cross_dedupe(events, engage)
        print(f"  Engage: +{len(events) - before} new events")
    except Exception as exc:
        print(f"  Engage warning: {exc}")
    return [asdict(e) for e in events]


def _save_events(events_list: List[Dict[str, Any]]) -> None:
    output_dir = os.path.dirname(EVENTS_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "count": len(events_list),
        "events": events_list,
    }
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


async def _do_scrape() -> None:
    global _events, _last_scraped, _scrape_running
    if _scrape_running:
        return
    _scrape_running = True
    try:
        loop = asyncio.get_event_loop()
        new_events = await loop.run_in_executor(None, _full_scrape)
        _events = new_events
        _last_scraped = datetime.now(timezone.utc)
        _save_events(new_events)
        print(f"Cache updated: {len(_events)} events at {_last_scraped.isoformat()}")
    except Exception as exc:
        log.exception("Scrape failed: %s", exc)
    finally:
        _scrape_running = False


async def _periodic_scrape() -> None:
    while True:
        await _do_scrape()
        await asyncio.sleep(SCRAPE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Scrape immediately, then repeat every SCRAPE_INTERVAL seconds.
    # /health returns 503 while _events is empty so Railway retries until ready.
    if USE_LOCAL_MODELS:
        from local_models import load_local_models
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(None, load_local_models)
        log.info("Local model load: %s", "ok" if ok else "failed (falling back to Gemini)")
    task = asyncio.create_task(_periodic_scrape())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="UNL Events Search",
    description="Keyword search over scraped UNL + Campus Labs events.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    payload = {
        "status": "ok" if _events else "starting",
        "events_loaded": len(_events),
        "last_scraped": _last_scraped.isoformat() if _last_scraped else None,
        "scrape_running": _scrape_running,
        "scrape_interval_seconds": SCRAPE_INTERVAL,
    }
    if not _events:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/events")
def get_events():
    """Return all cached events in the standard events.json format."""
    return {
        "scraped_at": _last_scraped.isoformat() if _last_scraped else None,
        "count": len(_events),
        "events": _events,
    }


@app.post("/reload")
async def reload_events():
    """Trigger an immediate re-scrape in the background."""
    if _scrape_running:
        return JSONResponse(status_code=409, content={"error": "Scrape already in progress."})
    asyncio.create_task(_do_scrape())
    return {"status": "scrape started"}


@app.get("/search")
def search_events(
    q: str = Query(..., description="Natural-language search query"),
    top: int = Query(10, ge=1, le=100, description="Max results to return"),
    model: str = Query(DEFAULT_MODEL, description="Ollama model for keyword expansion"),
    no_llm: bool = Query(False, description="Skip Ollama expansion"),
):
    base = base_terms(q)
    terms = list(base)
    llm_used = False
    backend_used = "none"

    log.info("SEARCH query=%r  base_terms=%s", q, base)

    llm_date_range = None
    llm_time_range = None
    if not no_llm:
        llm_keywords, llm_date_range, llm_time_range, backend_used = expand_query(q, model)
        if llm_keywords:
            llm_keywords = [k for k in llm_keywords if k not in STOP_WORDS and len(k) > 1]
            seen = set(terms)
            added = []
            for kw in llm_keywords:
                if kw not in seen:
                    terms.append(kw)
                    seen.add(kw)
                    added.append(kw)
            llm_used = True
            log.info("  LLM expansion  backend=%s  added=%s", backend_used, added)
            if llm_date_range:
                log.info("  LLM date_range %s → %s", llm_date_range[0], llm_date_range[1])
            if llm_time_range:
                log.info("  LLM time_range %s → %s", llm_time_range[0], llm_time_range[1])
        else:
            log.warning("  LLM expansion failed or returned no keywords — using base terms only")

    log.info("  final_terms=%s", terms)

    date_range = llm_date_range or extract_date_range(q)
    time_range = llm_time_range

    if not terms and not date_range and not time_range:
        return JSONResponse(status_code=400, content={"error": "No usable search terms in query."})

    pool = _events
    if date_range:
        pool = filter_by_date(pool, date_range, time_range)
    elif time_range:
        pool = filter_by_time(pool, time_range)

    log.info("  date_filter=%s  time_filter=%s  pool=%d events",
             f"{date_range[0]} → {date_range[1]}" if date_range else "none",
             f"{time_range[0]} → {time_range[1]}" if time_range else "none",
             len(pool))

    # If no keyword terms, return entire filtered pool (pure date/time query)
    if not terms:
        log.info("  no terms — returning full filtered pool (%d events)", len(pool))
        result_events = pool[:top]
        return {
            "query": q,
            "terms": [],
            "llm_used": llm_used,
            "backend": backend_used,
            "date_range": (
                {"start": str(date_range[0]), "end": str(date_range[1])}
                if date_range else None
            ),
            "time_range": (
                {"start": str(time_range[0]) if time_range[0] else None,
                 "end":   str(time_range[1]) if time_range[1] else None}
                if time_range else None
            ),
            "total_searched": len(pool),
            "count": len(result_events),
            "results": [
                {
                    "score": 0,
                    "url": e["url"],
                    "title": e["title"],
                    "start": e.get("start"),
                    "location": e.get("location"),
                    "group": e.get("group"),
                    "image_url": e.get("image_url"),
                }
                for e in result_events
            ],
        }

    results = search(pool, terms, top)
    log.info("  results=%d", len(results))

    return {
        "query": q,
        "terms": terms,
        "llm_used": llm_used,
        "backend": backend_used,
        "date_range": (
            {"start": str(date_range[0]), "end": str(date_range[1])}
            if date_range else None
        ),
        "time_range": (
            {"start": str(time_range[0]) if time_range[0] else None,
             "end":   str(time_range[1]) if time_range[1] else None}
            if time_range else None
        ),
        "total_searched": len(pool),
        "count": len(results),
        "results": [
            {
                "score": score,
                "url": e["url"],
                "title": e["title"],
                "start": e.get("start"),
                "location": e.get("location"),
                "group": e.get("group"),
                "image_url": e.get("image_url"),
            }
            for score, e in results
        ],
    }
