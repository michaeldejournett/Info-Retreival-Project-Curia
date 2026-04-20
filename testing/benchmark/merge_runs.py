"""Merge multiple past benchmark runs into a unified visualization.

When models are run separately (e.g., to work around GPU contention), each run
overwrites the canonical figures with only its own model. This script reads
several run directories, combines their summaries and per-query results, and
regenerates the unified figure set.

Each input run must contain `summary.json` and either `per_query.json` (added in
this revision) or only the older `per_query.csv` (which lacks enough fidelity
for visualization). Pass run directories as positional args or with --runs.

Example:
  python -m testing.benchmark.merge_runs \\
    --runs testing/benchmark/reports/run-A testing/benchmark/reports/run-B \\
    --output-dir testing/benchmark/reports/merged-<timestamp>
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from testing.benchmark.reporting import (
    append_figures_sync_to_summary,
    append_visuals_to_summary,
    load_per_query_json,
    load_summary_json,
    write_gate_results_json,
    write_gate_results_markdown,
    write_per_query_csv,
    write_per_query_json,
    write_summary_json,
    write_summary_markdown,
)
from testing.benchmark.retrieval import load_event_corpus
from testing.benchmark.schemas import ModelRunSummary, QueryRunResult
from testing.benchmark.visuals import generate_visual_artifacts, sync_canonical_figures


DEFAULT_FIGURES_DIR = "figures"


def _model_key(summary: ModelRunSummary) -> str:
    return f"{summary.provider}:{summary.model_name}"


def _load_run(run_dir: Path) -> tuple[List[ModelRunSummary], Dict[str, List[QueryRunResult]], Dict[str, object]]:
    summary_path = run_dir / "summary.json"
    per_query_path = run_dir / "per_query.json"

    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    if not per_query_path.is_file():
        raise FileNotFoundError(
            f"Missing per_query.json in {run_dir}. Re-run the benchmark with the "
            "current code to produce a lossless per_query.json (the older "
            "per_query.csv cannot be losslessly reconstructed for visualization)."
        )

    summaries = load_summary_json(summary_path)
    model_results = load_per_query_json(per_query_path)
    run_meta = json.loads(summary_path.read_text(encoding="utf-8")).get("run_meta") or {}
    return summaries, model_results, run_meta


def _validate_compatible_runs(run_metas: List[Dict[str, object]]) -> None:
    datasets = {str(rm.get("dataset") or "") for rm in run_metas if rm.get("dataset")}
    if len(datasets) > 1:
        joined = ", ".join(sorted(datasets))
        raise ValueError(
            f"Cannot merge runs across different datasets: {joined}. "
            "Re-run the models on a common dataset, or pass only runs that used the same one."
        )


def _ensure_gemini_present(summaries: List[ModelRunSummary]) -> None:
    if not any(s.provider.strip().lower() == "gemini" for s in summaries):
        raise ValueError(
            "Merged runs must include at least one Gemini model — visuals.py uses it as "
            "the canonical baseline. Add a Gemini run to the --runs list."
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge past benchmark runs into unified figures")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories to merge (e.g. testing/benchmark/reports/run-...)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Destination for merged outputs (default: testing/benchmark/reports/merged-<timestamp>)",
    )
    parser.add_argument(
        "--figures-dir",
        default=DEFAULT_FIGURES_DIR,
        help="Where to sync canonical figures (default: figures/)",
    )
    parser.add_argument(
        "--skip-figures-sync",
        action="store_true",
        help="Generate figures inside the merged run dir but do not overwrite figures/",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()

    merged_summaries: List[ModelRunSummary] = []
    merged_results: Dict[str, List[QueryRunResult]] = {}
    run_metas: List[Dict[str, object]] = []
    seen_keys: set[str] = set()

    for raw_path in args.runs:
        run_dir = Path(raw_path)
        print(f"Loading {run_dir}", flush=True)
        summaries, model_results, run_meta = _load_run(run_dir)
        run_metas.append(run_meta)

        for summary in summaries:
            key = _model_key(summary)
            if key in seen_keys:
                print(f"  warning: duplicate model {key} found — keeping first occurrence", flush=True)
                continue
            seen_keys.add(key)
            merged_summaries.append(summary)
            if key in model_results:
                merged_results[key] = model_results[key]
            else:
                print(f"  warning: per_query.json has no rows for {key}", flush=True)

    _validate_compatible_runs(run_metas)
    _ensure_gemini_present(merged_summaries)

    events_path = str(run_metas[0].get("events") or "scraped/events.json")
    try:
        events = load_event_corpus(events_path)
    except Exception:
        events = []

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_dir = Path("testing/benchmark/reports") / f"merged-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_meta = {
        "merged_from": [str(Path(r)) for r in args.runs],
        "model_count": len(merged_summaries),
        "case_count": sum(len(rows) for rows in merged_results.values()),
        "datasets": sorted({str(rm.get("dataset") or "") for rm in run_metas}),
    }

    print(f"Writing merged outputs to {out_dir}", flush=True)
    write_per_query_csv(out_dir, merged_results)
    write_per_query_json(out_dir, merged_results)
    summary_md = write_summary_markdown(out_dir, merged_summaries)
    summary_json = write_summary_json(out_dir, merged_summaries, merged_meta)

    # Reuse gate result writers with empty list (gates require live evaluation context).
    write_gate_results_json(out_dir, "merged", [])
    write_gate_results_markdown(out_dir, "merged", [])

    print("Generating visual artifacts...", flush=True)
    visual_output = generate_visual_artifacts(out_dir, merged_summaries, merged_results, events=events)
    visual_artifacts = list(visual_output.get("artifacts") or [])
    visual_warning = str(visual_output.get("warning") or "")
    if visual_warning:
        print(f"Visual warning: {visual_warning}", flush=True)

    if visual_artifacts:
        append_visuals_to_summary(summary_md, visual_artifacts)

    if visual_artifacts and not args.skip_figures_sync:
        print(f"Syncing figures to {args.figures_dir}", flush=True)
        sync_output = sync_canonical_figures(visual_artifacts, Path(args.figures_dir))
        synced = list(sync_output.get("synced") or [])
        sync_warning = str(sync_output.get("warning") or "")
        append_figures_sync_to_summary(summary_md, synced, sync_warning)
        print(f"Synced {len(synced)} canonical figure(s)", flush=True)

    print(f"Done. Summary: {summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
