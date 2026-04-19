#!/usr/bin/env python3
"""Side-by-side comparison of local pipeline vs Gemini for query expansion.

Prints keyword overlap, date_range agreement, time_range agreement, and latency.
"""
import argparse
import json
import os
import sys
import time as _time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "api"))

from local_models import load_local_models, is_loaded
from retrieval import load_retrieval_model
from temporal import warmup as temporal_warmup
from expansion import warmup as expansion_warmup
from search import expand_with_local, expand_with_gemini, DEFAULT_MODEL

QUERIES = [
    "events two weeks from now",
    "events two weeks from now about data science",
    "morning events this week",
    "pizza on April 25",
    "this weekend concerts",
    "sports games tonight",
    "art exhibits next month",
    "free food thursday",
]


def _jaccard(a, b):
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _fmt_dr(dr):
    if dr is None:
        return "—"
    return f"{dr[0]} → {dr[1]}"


def _fmt_tr(tr):
    if tr is None:
        return "—"
    s = tr[0].isoformat() if tr[0] else "?"
    e = tr[1].isoformat() if tr[1] else "?"
    return f"{s} → {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save results to results/run_TIMESTAMP.json")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: set GEMINI_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    print("Loading local models (may take ~15s first time)...", flush=True)
    t0 = _time.time()
    ok = load_local_models()
    print(f"  loaded={ok} is_loaded={is_loaded()} elapsed={_time.time()-t0:.1f}s", flush=True)
    if not ok:
        sys.exit(1)

    print("Loading retrieval model...", flush=True)
    load_retrieval_model()

    print("Warming prefix KV caches...", flush=True)
    temporal_warmup()
    expansion_warmup()
    print(f"  warmup done  total_elapsed={_time.time()-t0:.1f}s\n", flush=True)

    jaccards = []
    dr_matches = 0
    dr_both = 0
    tr_matches = 0
    tr_both = 0
    local_lat = []
    gem_lat = []
    per_query_records = []

    for q in QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)

        t0 = _time.time()
        l_kw, l_dr, l_tr = expand_with_local(q)
        lt = _time.time() - t0
        local_lat.append(lt)

        t0 = _time.time()
        g_kw, g_dr, g_tr = expand_with_gemini(q, DEFAULT_MODEL)
        gt = _time.time() - t0
        gem_lat.append(gt)

        j = _jaccard(l_kw, g_kw)
        jaccards.append(j)

        print(f"\n  LOCAL   ({lt:.2f}s)")
        print(f"    kw:    {sorted(l_kw) if l_kw else None}")
        print(f"    date:  {_fmt_dr(l_dr)}")
        print(f"    time:  {_fmt_tr(l_tr)}")

        print(f"\n  GEMINI  ({gt:.2f}s)")
        print(f"    kw:    {sorted(g_kw) if g_kw else None}")
        print(f"    date:  {_fmt_dr(g_dr)}")
        print(f"    time:  {_fmt_tr(g_tr)}")

        print(f"\n  AGREEMENT")
        print(f"    kw jaccard:   {j:.2f}")
        if l_kw and g_kw:
            both = set(l_kw) & set(g_kw)
            only_l = set(l_kw) - set(g_kw)
            only_g = set(g_kw) - set(l_kw)
            print(f"    shared:       {sorted(both)}")
            print(f"    local-only:   {sorted(only_l)}")
            print(f"    gemini-only:  {sorted(only_g)}")

        if l_dr is not None and g_dr is not None:
            dr_both += 1
            if l_dr == g_dr:
                dr_matches += 1
            print(f"    date match:   {l_dr == g_dr}")

        if l_tr is not None and g_tr is not None:
            tr_both += 1
            if l_tr == g_tr:
                tr_matches += 1
            print(f"    time match:   {l_tr == g_tr}")
        print()

        per_query_records.append({
            "query": q,
            "jaccard": j,
            "date_match": (l_dr == g_dr) if (l_dr is not None and g_dr is not None) else None,
            "time_match": (l_tr == g_tr) if (l_tr is not None and g_tr is not None) else None,
            "local_latency_s": round(lt, 4),
            "gemini_latency_s": round(gt, 4),
        })

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  queries:                  {len(QUERIES)}")
    print(f"  kw jaccard  mean:         {sum(jaccards)/len(jaccards):.2f}")
    print(f"  date_range agreement:     {dr_matches}/{dr_both} (both non-null)")
    print(f"  time_range agreement:     {tr_matches}/{tr_both} (both non-null)")
    print(f"  local latency  mean:      {sum(local_lat)/len(local_lat):.2f}s")
    print(f"  gemini latency mean:      {sum(gem_lat)/len(gem_lat):.2f}s")

    if args.save:
        ts = datetime.now()
        run_id = f"run_{ts.strftime('%Y%m%d_%H%M%S')}"
        payload = {
            "run_id": run_id,
            "timestamp": ts.isoformat(timespec="seconds"),
            "model_local": os.environ.get("LOCAL_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
            "model_gemini": DEFAULT_MODEL,
            "queries": list(QUERIES),
            "per_query": per_query_records,
            "summary": {
                "mean_jaccard": sum(jaccards) / len(jaccards),
                "date_matches": dr_matches,
                "date_both_non_null": dr_both,
                "time_matches": tr_matches,
                "time_both_non_null": tr_both,
                "mean_local_latency_s": sum(local_lat) / len(local_lat),
                "mean_gemini_latency_s": sum(gem_lat) / len(gem_lat),
            },
        }
        Path("results").mkdir(exist_ok=True)
        out_path = Path("results") / f"{run_id}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n  saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
