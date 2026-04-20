from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from testing.benchmark.adapters.base import AdapterConfig, build_adapter
from testing.benchmark.dataset import (
    generate_template_cases,
    load_dataset,
    validate_cases,
    write_template_dataset,
)
from testing.benchmark.evaluator import summarize_model
from testing.benchmark.gates import compute_stability_metrics, evaluate_summary, resolve_gate_profile
from testing.benchmark.model_sets import DEFAULT_MODEL_SET, MODEL_SETS, get_model_set
from testing.benchmark.reporting import (
    append_figures_sync_to_summary,
    append_visuals_to_summary,
    create_run_dir,
    write_gate_results_json,
    write_gate_results_markdown,
    write_per_query_csv,
    write_per_query_json,
    write_summary_json,
    write_summary_markdown,
)
from testing.benchmark.retrieval import load_event_corpus, retrieve_ranked_urls
from testing.benchmark.schemas import ModelInvocationResult, QueryRunResult
from testing.benchmark.visuals import generate_visual_artifacts, sync_canonical_figures


DEFAULT_SMOKE_DATASET = "testing/benchmark/datasets/queries_smoke.json"
DEFAULT_REPORTS_DIR = "testing/benchmark/reports"
DEFAULT_EVENTS_PATH = "scraped/events.json"
DEFAULT_FIGURES_DIR = "figures"


def _parse_models(raw: str) -> List[AdapterConfig]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("No models provided")

    configs: List[AdapterConfig] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid model spec '{item}'. Expected provider:model")
        provider, model_name = item.split(":", 1)
        configs.append(AdapterConfig(provider=provider.strip(), model_name=model_name.strip()))
    return configs


def _resolve_model_specs(args: argparse.Namespace) -> str:
    if args.models and args.models.strip():
        return args.models.strip()
    return get_model_set(args.model_set)


def _load_cases(args: argparse.Namespace):
    if args.dataset:
        return load_dataset(args.dataset)

    if args.profile == "smoke":
        return load_dataset(DEFAULT_SMOKE_DATASET)

    cases = generate_template_cases(size=args.full_size)
    metadata = {
        "name": "generated-template",
        "version": "generated",
        "label_status": "pending-manual",
    }
    return cases, metadata


def _apply_suite_defaults(args: argparse.Namespace) -> None:
    suite = (args.suite or "default").strip().lower()
    if suite == "correctness":
        args.enforce_label_quality = True
        if not args.gate_profile:
            args.gate_profile = "correctness"
        return

    if suite == "latency-stability":
        args.enforce_label_quality = True
        if args.runs_per_query == 1:
            args.runs_per_query = 3
        if not args.gate_profile:
            args.gate_profile = "latency-stability"
        return

    if not args.gate_profile:
        args.gate_profile = "none"


def _ensure_template_written(args: argparse.Namespace) -> None:
    if not args.write_template:
        return
    output = write_template_dataset(args.write_template, size=args.full_size)
    print(f"Template dataset written to: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curia cross-model extraction benchmark")
    parser.add_argument(
        "--suite",
        choices=["default", "correctness", "latency-stability"],
        default="default",
        help="Suite mode with preset validation and gate behavior",
    )
    parser.add_argument(
        "--model-set",
        choices=sorted(MODEL_SETS.keys()),
        default=DEFAULT_MODEL_SET,
        help="Named model set to run when --models is not provided",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated provider:model list (overrides --model-set)",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="smoke",
        help="Use smoke dataset or generated full template",
    )
    parser.add_argument("--dataset", default="", help="Path to benchmark dataset JSON")
    parser.add_argument(
        "--dataset-version",
        default="",
        help="Optional dataset version override to include in run metadata",
    )
    parser.add_argument(
        "--enforce-label-quality",
        action="store_true",
        help="Fail fast when dataset labels are incomplete or malformed",
    )
    parser.add_argument(
        "--expected-label-status",
        default="",
        help="Require dataset metadata.label_status to match this exact value",
    )
    parser.add_argument(
        "--gate-profile",
        choices=["none", "correctness", "latency-stability", "all"],
        default="",
        help="Evaluation gate profile for pass/fail reporting",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip static chart generation in run artifacts",
    )
    parser.add_argument("--events", default=DEFAULT_EVENTS_PATH, help="Path to events corpus JSON")
    parser.add_argument("--output-dir", default=DEFAULT_REPORTS_DIR, help="Directory for run outputs")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N URLs for retrieval metrics")
    parser.add_argument("--runs-per-query", type=int, default=1, help="Repeated runs per query")
    parser.add_argument("--inter-query-delay-ms", type=int, default=0, help="Delay between requests")
    parser.add_argument("--timeout-s", type=float, default=20.0, help="Timeout per model call")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Generation max output tokens")
    parser.add_argument(
        "--huggingface-backend",
        choices=["api", "local"],
        default="",
        help="Set Hugging Face backend mode for this run",
    )
    parser.add_argument(
        "--huggingface-local-device",
        default="",
        help="Optional local device override for Hugging Face local backend (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--huggingface-local-dtype",
        default="",
        help="Optional local dtype override for Hugging Face local backend (auto/float16/bfloat16/float32/none)",
    )
    parser.add_argument("--full-size", type=int, default=100, help="Generated template size for full profile")
    parser.add_argument(
        "--write-template",
        default="",
        help="Optional path to write a generated dataset template and continue",
    )
    parser.add_argument(
        "--max-parallel-models",
        type=int,
        default=4,
        help=(
            "Maximum concurrent-safe models to run in parallel (network-bound adapters). "
            "Models that declare is_concurrent_safe=False (e.g. local HuggingFace) always "
            "run serially after the concurrent batch. Set to 1 for fully sequential behavior."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Emit a per-model progress line every N completed cases",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _apply_suite_defaults(args)
    gate_profile = resolve_gate_profile(args.gate_profile)

    if args.huggingface_backend:
        os.environ["HUGGINGFACE_BACKEND"] = args.huggingface_backend
    if args.huggingface_local_device:
        os.environ["HUGGINGFACE_LOCAL_DEVICE"] = args.huggingface_local_device
    if args.huggingface_local_dtype:
        os.environ["HUGGINGFACE_LOCAL_DTYPE"] = args.huggingface_local_dtype

    resolved_models = _resolve_model_specs(args)
    model_configs = _parse_models(resolved_models)
    _ensure_template_written(args)

    cases, dataset_metadata = _load_cases(args)
    validation_meta = validate_cases(
        cases,
        strict_labels=args.enforce_label_quality,
        metadata=dataset_metadata,
        expected_label_status=args.expected_label_status,
    )

    events = load_event_corpus(args.events)

    run_dir = create_run_dir(args.output_dir)

    model_results: Dict[str, List[QueryRunResult]] = {}
    summaries = []
    gate_results = []

    print_lock = threading.Lock()

    def _emit(message: str) -> None:
        with print_lock:
            print(message, flush=True)

    def _run_model(cfg: AdapterConfig) -> Tuple[str, List[QueryRunResult], Any, Any]:
        cfg.timeout_s = args.timeout_s
        cfg.temperature = args.temperature
        cfg.max_tokens = args.max_tokens

        adapter = build_adapter(cfg)
        model_key = f"{cfg.provider}:{cfg.model_name}"
        rows: List[QueryRunResult] = []

        _emit(f"Running {model_key} on {len(cases)} cases x {args.runs_per_query} run(s) each")

        progress_every = max(1, args.progress_every)
        for case_idx, case in enumerate(cases, start=1):
            for run_idx in range(args.runs_per_query):
                inv: ModelInvocationResult = adapter.extract(case.query)
                inv.metadata["run_index"] = run_idx
                inv.metadata["case_id"] = case.case_id
                predicted_urls = retrieve_ranked_urls(events, inv, top_n=args.top_n)
                rows.append(QueryRunResult(case=case, invocation=inv, predicted_urls=predicted_urls))

                if args.inter_query_delay_ms > 0:
                    time.sleep(args.inter_query_delay_ms / 1000.0)

            if case_idx % progress_every == 0 or case_idx == len(cases):
                _emit(f"  [{model_key}] {case_idx}/{len(cases)} cases done")

        summary = summarize_model(rows)
        if args.runs_per_query > 1:
            summary.details.update(compute_stability_metrics(rows))
        else:
            summary.details.update(
                {
                    "run_count": 1,
                    "parse_success_stddev": None,
                    "keyword_f1_stddev": None,
                    "latency_p95_stddev_ms": None,
                }
            )
        gate_result = evaluate_summary(summary, gate_profile)
        _emit(f"✓ {model_key} complete")
        return model_key, rows, summary, gate_result

    # Partition by concurrency safety (build adapters once to query is_concurrent_safe).
    concurrent_cfgs: List[AdapterConfig] = []
    serial_cfgs: List[AdapterConfig] = []
    for cfg in model_configs:
        probe_adapter = build_adapter(cfg)
        if probe_adapter.is_concurrent_safe:
            concurrent_cfgs.append(cfg)
        else:
            serial_cfgs.append(cfg)

    # Stable ordering for results: concurrent results in submission order, then serial.
    ordered_keys: List[str] = []
    pending_results: Dict[str, Tuple[List[QueryRunResult], Any, Any]] = {}

    if concurrent_cfgs:
        max_workers = max(1, min(args.max_parallel_models, len(concurrent_cfgs)))
        _emit(
            f"Dispatching {len(concurrent_cfgs)} concurrent-safe model(s) "
            f"with max_workers={max_workers}"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_key = {}
            for cfg in concurrent_cfgs:
                key = f"{cfg.provider}:{cfg.model_name}"
                ordered_keys.append(key)
                future_to_key[pool.submit(_run_model, cfg)] = key
            for future in as_completed(future_to_key):
                key, rows, summary, gate_result = future.result()
                pending_results[key] = (rows, summary, gate_result)

    if serial_cfgs:
        _emit(f"Running {len(serial_cfgs)} serial-only model(s) sequentially")
        for cfg in serial_cfgs:
            key, rows, summary, gate_result = _run_model(cfg)
            ordered_keys.append(key)
            pending_results[key] = (rows, summary, gate_result)

    for key in ordered_keys:
        rows, summary, gate_result = pending_results[key]
        model_results[key] = rows
        summaries.append(summary)
        gate_results.append(gate_result)

    dataset_version = args.dataset_version or str(dataset_metadata.get("version") or "")
    gate_passed = sum(1 for g in gate_results if g.get("passed"))

    # Write data files first so results are preserved even if visual generation fails.
    per_query_csv = write_per_query_csv(run_dir, model_results)
    per_query_json = write_per_query_json(run_dir, model_results)
    summary_md = write_summary_markdown(run_dir, summaries)

    visual_artifacts: List[str] = []
    visual_warning = ""
    synced_figures: List[str] = []
    figures_sync_warning = ""
    if not args.skip_visuals:
        visual_output = generate_visual_artifacts(run_dir, summaries, model_results)
        visual_artifacts = list(visual_output.get("artifacts") or [])
        visual_warning = str(visual_output.get("warning") or "")
        if visual_artifacts:
            sync_output = sync_canonical_figures(visual_artifacts, Path(DEFAULT_FIGURES_DIR))
            synced_figures = list(sync_output.get("synced") or [])
            figures_sync_warning = str(sync_output.get("warning") or "")

    run_meta = {
        "suite": args.suite,
        "profile": args.profile,
        "gate_profile": args.gate_profile,
        "model_set": args.model_set,
        "dataset": args.dataset or (DEFAULT_SMOKE_DATASET if args.profile == "smoke" else "generated-template"),
        "dataset_version": dataset_version,
        "dataset_label_status": str(dataset_metadata.get("label_status") or ""),
        "events": args.events,
        "models": resolved_models,
        "top_n": args.top_n,
        "runs_per_query": args.runs_per_query,
        "inter_query_delay_ms": args.inter_query_delay_ms,
        "timeout_s": args.timeout_s,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "huggingface_backend": os.environ.get("HUGGINGFACE_BACKEND", "api"),
        "huggingface_local_device": os.environ.get("HUGGINGFACE_LOCAL_DEVICE", "auto"),
        "huggingface_local_dtype": os.environ.get("HUGGINGFACE_LOCAL_DTYPE", "auto"),
        "enforce_label_quality": args.enforce_label_quality,
        "expected_label_status": args.expected_label_status,
        "dataset_validation": validation_meta,
        "gate_passed_models": gate_passed,
        "gate_total_models": len(gate_results),
        "visual_artifact_count": len(visual_artifacts),
        "visual_warning": visual_warning,
        "figures_sync_dir": DEFAULT_FIGURES_DIR if visual_artifacts else "",
        "figures_synced_count": len(synced_figures),
        "figures_sync_warning": figures_sync_warning,
        "case_count": len(cases),
        "event_count": len(events),
    }

    if visual_artifacts:
        append_visuals_to_summary(summary_md, visual_artifacts)
    if synced_figures or figures_sync_warning:
        append_figures_sync_to_summary(summary_md, synced_figures, figures_sync_warning)

    gate_json = write_gate_results_json(run_dir, args.gate_profile, gate_results)
    gate_md = write_gate_results_markdown(run_dir, args.gate_profile, gate_results)
    summary_json = write_summary_json(run_dir, summaries, run_meta)

    # Save generated template when running full profile without dataset, for manual labeling.
    if args.profile == "full" and not args.dataset:
        template_path = run_dir / "generated_full_template.json"
        template_payload = {
            "metadata": {
                "source": "generated-by-runner",
                "label_status": "pending-manual",
                "size": len(cases),
            },
            "cases": [
                {
                    "id": c.case_id,
                    "query": c.query,
                    "tags": c.tags,
                    "expected": {
                        "keywords": c.expected.keywords,
                        "date_from": c.expected.date_from,
                        "date_to": c.expected.date_to,
                        "time_from": c.expected.time_from,
                        "time_to": c.expected.time_to,
                        "relevant_event_urls": c.expected.relevant_event_urls,
                    },
                    "notes": c.notes,
                }
                for c in cases
            ],
        }
        template_path.write_text(json.dumps(template_payload, indent=2), encoding="utf-8")

    print("Run complete.")
    print(f"Summary JSON: {summary_json}")
    print(f"Per-query CSV: {per_query_csv}")
    print(f"Summary MD : {summary_md}")
    print(f"Gate JSON   : {gate_json}")
    print(f"Gate MD     : {gate_md}")
    if visual_artifacts:
        print(f"Visuals dir : {run_dir / 'visuals'}")
    if synced_figures:
        print(f"Figures sync: {DEFAULT_FIGURES_DIR} ({len(synced_figures)} file(s))")
    if visual_warning:
        print(f"Visual note : {visual_warning}")
    if figures_sync_warning:
        print(f"Sync note   : {figures_sync_warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
