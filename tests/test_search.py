import sys
import types
import pytest

# Provide lightweight dummy modules so importing `api.search` doesn't require
# heavyweight external SDKs (requests, google.genai) during tests.
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.genai", types.ModuleType("google.genai"))

from api import search as search_mod


SAMPLE_EVENTS = [
    {
        "title": "Campus Sports Day",
        "group": "Athletics",
        "description": "Join sports activities and soccer tournaments.",
        "location": "Field",
        "audience": "Everyone",
        "start": "2026-04-10T10:00:00",
        "url": "http://example/sports",
    },
    {
        "title": "Jazz Music Night",
        "group": "Music Club",
        "description": "An evening of jazz, choir and band performances.",
        "location": "Auditorium",
        "audience": "Music lovers",
        "start": "2026-04-11T19:00:00",
        "url": "http://example/music",
    },
    {
        "title": "Food Truck Festival",
        "group": "Culinary Society",
        "description": "Taste pizza, tacos, BBQ and sushi.",
        "location": "Quad",
        "audience": "Everyone",
        "start": "2026-04-12T12:00:00",
        "url": "http://example/food",
    },
    {
        "title": "Physics Colloquium",
        "group": "Science Dept",
        "description": "Research talk on quantum mechanics.",
        "location": "Room 101",
        "audience": "Students",
        "start": "2026-04-13T15:00:00",
        "url": "http://example/lecture",
    },
]


def run_search(query, top_n=5):
    terms = search_mod.base_terms(query)
    results = search_mod.search(SAMPLE_EVENTS, terms, top_n)
    return [e for _, e in results]


def run_search_urls(query):
    return [e["url"] for e in run_search(query)]


# Try to load real scraped events; fall back to SAMPLE_EVENTS if unavailable/invalid.
USE_REAL = False
try:
    REAL_EVENTS = search_mod.load_events("scraped/events.json")
    if REAL_EVENTS:
        USE_REAL = True
        EVENTS = REAL_EVENTS
    else:
        EVENTS = SAMPLE_EVENTS
except Exception:
    EVENTS = SAMPLE_EVENTS


def run_search_real(query, top_n=5):
    terms = search_mod.base_terms(query)
    results = search_mod.search(EVENTS, terms, top_n)
    return [e for _, e in results]


def contains_keyword_in_event(e, keyword):
    text = (e.get("title", "") + " " + e.get("description", "")).lower()
    return keyword.lower() in text


def test_music_query_returns_music_event():
    if USE_REAL:
        res = run_search_real("music")
        assert res, "Expected non-empty results for 'music' against real events.json"
        assert any(contains_keyword_in_event(e, "music") for e in res)
    else:
        assert "http://example/music" in run_search_urls("music")


def test_sports_query_returns_sports_event():
    if USE_REAL:
        res = run_search_real("sports")
        assert res, "Expected non-empty results for 'sports' against real events.json"
        assert any(contains_keyword_in_event(e, "sports") for e in res)
    else:
        assert "http://example/sports" in run_search_urls("sports")


def test_food_query_returns_food_event():
    if USE_REAL:
        res = run_search_real("food")
        assert res, "Expected non-empty results for 'food' against real events.json"
        assert any(contains_keyword_in_event(e, "food") or contains_keyword_in_event(e, "pizza") or contains_keyword_in_event(e, "taco") or contains_keyword_in_event(e, "sushi") for e in res)
    else:
        assert "http://example/food" in run_search_urls("food")


def test_sushi_matches_food_event():
    if USE_REAL:
        res = run_search_real("sushi")
        # sushi may or may not appear in real data; if there are no results, skip this assertion
        if not res:
            pytest.skip("no 'sushi' results in real events.json; skipping")
        assert any(contains_keyword_in_event(e, "sushi") for e in res)
    else:
        assert "http://example/food" in run_search_urls("sushi")


def test_case_insensitive_queries_match():
    if USE_REAL:
        r1 = [e.get("url") or e.get("title") for e in run_search_real("Music")]
        r2 = [e.get("url") or e.get("title") for e in run_search_real("music")]
        assert r1 == r2
    else:
        assert run_search_urls("Music") == run_search_urls("music")
