"""Generate search accuracy metrics (precision/recall) for all app categories.

Writes JSON and CSV to metrics/search_metrics.json and metrics/search_metrics.csv.
Also writes metrics/search_metrics_extended.json with P@K, PR curves, and score
distributions needed for the additional charts.
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


def run_search_scored(events, terms):
    """Return all (score, event) pairs with score > 0, sorted descending."""
    terms = [t.lower() for t in terms]
    scored = [(search_mod.score_event(e, terms), e) for e in events]
    scored = [(s, e) for s, e in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def precision_at_k(scored, gt, k):
    top = [e.get("url") or e.get("title") for _, e in scored[:k]]
    if not top:
        return 0.0
    return len([u for u in top if u in gt]) / len(top)


def recall_at_k(scored, gt, k):
    if not gt:
        return 0.0
    top = [e.get("url") or e.get("title") for _, e in scored[:k]]
    return len([u for u in top if u in gt]) / len(gt)


def pr_curve(scored, gt, max_k=50):
    """Return list of (recall, precision) pairs as k varies from 1 to max_k."""
    points = []
    for k in range(1, min(max_k, len(scored)) + 1):
        p = precision_at_k(scored, gt, k)
        r = recall_at_k(scored, gt, k)
        points.append({"k": k, "precision": p, "recall": r})
    return points


def metrics_for_category(events, category, expanded_keywords):
    gt = ground_truth(events, expanded_keywords)

    raw_scored = run_search_scored(events, [category])
    exp_scored = run_search_scored(events, expanded_keywords)

    raw_urls_50 = [e.get("url") or e.get("title") for _, e in raw_scored[:50]]
    exp_urls_50 = [e.get("url") or e.get("title") for _, e in exp_scored[:50]]

    raw_tp = len([u for u in raw_urls_50 if u in gt])
    exp_tp = len([u for u in exp_urls_50 if u in gt])
    raw_precision = raw_tp / max(1, len(raw_urls_50))
    exp_precision = exp_tp / max(1, len(exp_urls_50))
    raw_recall = raw_tp / max(1, len(gt)) if gt else 0.0
    exp_recall = exp_tp / max(1, len(gt)) if gt else 0.0

    # P@K
    ks = [1, 3, 5, 10]
    raw_pak = {k: precision_at_k(raw_scored, gt, k) for k in ks}
    exp_pak = {k: precision_at_k(exp_scored, gt, k) for k in ks}

    # PR curves
    raw_pr = pr_curve(raw_scored, gt, max_k=50)
    exp_pr = pr_curve(exp_scored, gt, max_k=50)

    # Score distributions (all non-zero scores)
    raw_scores = [s for s, _ in raw_scored]
    exp_scores = [s for s, _ in exp_scored]

    return {
        "category": category,
        "gt_count": len(gt),
        "raw_hits": len(raw_urls_50),
        "exp_hits": len(exp_urls_50),
        "raw_tp": raw_tp,
        "exp_tp": exp_tp,
        "raw_precision": raw_precision,
        "exp_precision": exp_precision,
        "raw_recall": raw_recall,
        "exp_recall": exp_recall,
        "raw_urls": raw_urls_50,
        "exp_urls": exp_urls_50,
        # extended fields
        "raw_pak": raw_pak,
        "exp_pak": exp_pak,
        "raw_pr_curve": raw_pr,
        "exp_pr_curve": exp_pr,
        "raw_scores": raw_scores,
        "exp_scores": exp_scores,
    }


def main():
    out_dir = "metrics"
    os.makedirs(out_dir, exist_ok=True)
    events = load_events()

    rules = parse_category_rules("backend/db.js")
    if not rules:
        rules = {
            "food": ["food", "pizza", "taco", "sushi", "burger"],
            "music": ["music", "concert", "jazz", "band"],
            "sports": ["sport", "soccer", "basketball", "run"],
        }

    results = []
    for cat, expanded in rules.items():
        m = metrics_for_category(events, cat, expanded)
        results.append(m)

    # core metrics JSON (backward compatible)
    json_path = os.path.join(out_dir, "search_metrics.json")
    core_keys = ["category", "gt_count", "raw_hits", "exp_hits", "raw_tp", "exp_tp",
                 "raw_precision", "exp_precision", "raw_recall", "exp_recall",
                 "raw_urls", "exp_urls"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{k: r[k] for k in core_keys} for r in results], f, indent=2, ensure_ascii=False)

    # extended metrics JSON
    ext_path = os.path.join(out_dir, "search_metrics_extended.json")
    with open(ext_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV (core fields)
    csv_path = os.path.join(out_dir, "search_metrics.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "gt_count", "raw_hits", "exp_hits", "raw_tp", "exp_tp",
                         "raw_precision", "exp_precision", "raw_recall", "exp_recall"])
        for r in results:
            writer.writerow([
                r["category"], r["gt_count"], r["raw_hits"], r["exp_hits"], r["raw_tp"], r["exp_tp"],
                f"{r['raw_precision']:.4f}", f"{r['exp_precision']:.4f}",
                f"{r['raw_recall']:.4f}", f"{r['exp_recall']:.4f}",
            ])

    print(f"Wrote metrics to: {json_path}, {ext_path}, and {csv_path}")


if __name__ == "__main__":
    main()
