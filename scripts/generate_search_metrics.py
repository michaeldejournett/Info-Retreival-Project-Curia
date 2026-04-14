"""Generate search accuracy metrics (precision/recall) for all app categories.

Writes JSON and CSV to metrics/search_metrics.json and metrics/search_metrics.csv.
"""
import json
import os
import re
import sys
import types

# allow importing api.search without external SDKs in minimal envs
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.genai", types.ModuleType("google.genai"))

# ensure repo root is on sys.path so `api` package is importable when running the script
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
    """Extract CATEGORY_RULES from backend/db.js and return {category: [keywords]}."""
    if not os.path.exists(js_path):
        return {}
    text = open(js_path, "r", encoding="utf-8").read()
    # find objects like { keywords: [...], category: 'music' } (allow spacing/newlines)
    pattern = re.compile(
        r"\{\s*keywords\s*:\s*\[([^\]]*)\]\s*,\s*category\s*:\s*['\"]([^'\"]+)['\"][^\}]*\}",
        re.S | re.I,
    )
    rules = {}
    for kws_text, category in pattern.findall(text):
        # extract quoted keywords inside the keywords array
        keywords = re.findall(r"['\"]([^'\"]+)['\"]", kws_text)
        if keywords:
            # normalize keywords (lowercase, strip)
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


def run_search(events, terms, top_n=50):
    terms = [t.lower() for t in terms]
    scored = search_mod.search(events, terms, top_n)
    return [e for _, e in scored]


def metrics_for_category(events, category, expanded_keywords):
    gt = ground_truth(events, expanded_keywords)
    raw_results = run_search(events, [category], top_n=50)
    exp_results = run_search(events, expanded_keywords, top_n=50)

    raw_urls = [e.get("url") or e.get("title") for e in raw_results]
    exp_urls = [e.get("url") or e.get("title") for e in exp_results]

    raw_tp = len([u for u in raw_urls if u in gt])
    exp_tp = len([u for u in exp_urls if u in gt])
    raw_precision = raw_tp / max(1, len(raw_urls))
    exp_precision = exp_tp / max(1, len(exp_urls))
    raw_recall = raw_tp / max(1, len(gt)) if gt else 0.0
    exp_recall = exp_tp / max(1, len(gt)) if gt else 0.0

    return {
        "category": category,
        "gt_count": len(gt),
        "raw_hits": len(raw_urls),
        "exp_hits": len(exp_urls),
        "raw_tp": raw_tp,
        "exp_tp": exp_tp,
        "raw_precision": raw_precision,
        "exp_precision": exp_precision,
        "raw_recall": raw_recall,
        "exp_recall": exp_recall,
        "raw_urls": raw_urls,
        "exp_urls": exp_urls,
    }


def main():
    out_dir = "metrics"
    os.makedirs(out_dir, exist_ok=True)
    events = load_events()

    rules = parse_category_rules("backend/db.js")
    if not rules:
        # fallback to a small set if parsing fails
        rules = {
            "food": ["food", "pizza", "taco", "sushi", "burger"],
            "music": ["music", "concert", "jazz", "band"],
            "sports": ["sport", "soccer", "basketball", "run"],
        }

    results = []
    for cat, expanded in rules.items():
        m = metrics_for_category(events, cat, expanded)
        results.append(m)

    json_path = os.path.join(out_dir, "search_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # also write CSV
    csv_path = os.path.join(out_dir, "search_metrics.csv")
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "gt_count", "raw_hits", "exp_hits", "raw_tp", "exp_tp", "raw_precision", "exp_precision", "raw_recall", "exp_recall"])
        for r in results:
            writer.writerow([
                r["category"], r["gt_count"], r["raw_hits"], r["exp_hits"], r["raw_tp"], r["exp_tp"],
                f"{r['raw_precision']:.4f}", f"{r['exp_precision']:.4f}", f"{r['raw_recall']:.4f}", f"{r['exp_recall']:.4f}",
            ])

    print(f"Wrote metrics to: {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
