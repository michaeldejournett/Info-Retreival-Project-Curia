import sys
import types
import os
import re
import pytest

# Dummy external modules so importing api.search works in minimal envs
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.genai", types.ModuleType("google.genai"))

# ensure repo root is importable when running tests directly
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from api import search as search_mod


def load_events():
    try:
        return search_mod.load_events("scraped/events.json")
    except Exception:
        return [
            {"title": "Food Truck Festival", "description": "pizza tacos sushi", "url": "http://example/food"},
            {"title": "Campus Dance Night", "description": "salsa hip hop", "url": "http://example/dance"},
            {"title": "Jazz Concert", "description": "jazz band choir", "url": "http://example/music"},
        ]


def parse_category_rules(js_path="backend/db.js"):
    if not os.path.exists(js_path):
        return {}
    text = open(js_path, "r", encoding="utf-8").read()
    pattern = re.compile(
        r"\{\s*keywords\s*:\s*\[([^\]]*)\]\s*,\s*category\s*:\s*['\"]([^'\"]+)['\"][^\}]*\}",
        re.S | re.I,
    )
    rules = {}
    for kws_text, category in pattern.findall(text):
        keywords = re.findall(r"['\"]([^'\"]+)['\"]", kws_text)
        if keywords:
            rules[category.lower()] = sorted(set([k.strip().lower() for k in keywords if k.strip()]))
    return rules


def ground_truth(events, keywords):
    kws = [k.lower() for k in keywords]
    out = set()
    for e in events:
        text = (e.get("title", "") + " " + e.get("description", "")).lower()
        if any(k in text for k in kws):
            out.add(e.get("url") or e.get("title"))
    return out


def run_search_with_terms(events, terms, top_n=10):
    terms = [t.lower() for t in terms]
    scored = search_mod.search(events, terms, top_n)
    return [e for _, e in scored]


# load category rules at module import so pytest can parametrize
RULES = parse_category_rules("backend/db.js")
if not RULES:
    # fallback categories if parsing failed
    RULES = {
        "food": ["food", "pizza", "taco", "sushi"],
        "music": ["music", "concert", "jazz", "band"],
        "sports": ["sport", "soccer", "basketball"],
    }


@pytest.mark.parametrize("category", sorted(RULES.keys()))
def test_single_term_vs_expanded(category):
    events = load_events()
    single = category
    expanded = RULES.get(category, [category])

    gt = ground_truth(events, expanded)
    if not gt:
        pytest.skip(f"No ground-truth events found for category '{category}' in events.json")

    raw_results = run_search_with_terms(events, [single], top_n=10)
    raw_urls = [e.get("url") or e.get("title") for e in raw_results]

    expanded_results = run_search_with_terms(events, expanded, top_n=10)
    exp_urls = [e.get("url") or e.get("title") for e in expanded_results]

    raw_tp = len([u for u in raw_urls if u in gt])
    exp_tp = len([u for u in exp_urls if u in gt])
    raw_recall = raw_tp / len(gt)
    exp_recall = exp_tp / len(gt)

    # Expect expanded recall >= raw recall
    assert exp_recall + 1e-9 >= raw_recall
