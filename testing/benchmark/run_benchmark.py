from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from testing.benchmark.adapters.base import AdapterConfig, build_adapter
from testing.benchmark.dataset import generate_template_cases, load_cases, write_template_dataset
from testing.benchmark.evaluator import summarize_model
from testing.benchmark.model_sets import DEFAULT_MODEL_SET, MODEL_SETS, get_model_set
from testing.benchmark.reporting import create_run_dir, write_per_query_csv, write_summary_json, write_summary_markdown
from testing.benchmark.retrieval import load_event_corpus, retrieve_ranked_urls
from testing.benchmark.schemas import ModelInvocationResult, QueryRunResult


DEFAULT_SMOKE_DATASET = "testing/benchmark/datasets/queries_smoke.json"
DEFAULT_REPORTS_DIR = "testing/benchmark/reports"
DEFAULT_EVENTS_PATH = "scraped/events.json"


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
        return load_cases(args.dataset)

    if args.profile == "smoke":
        return load_cases(DEFAULT_SMOKE_DATASET)

    return generate_template_cases(size=args.full_size)


def _ensure_template_written(args: argparse.Namespace) -> None:
    if not args.write_template:
        return
    output = write_template_dataset(args.write_template, size=args.full_size)
    print(f"Template dataset written to: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curia cross-model extraction benchmark")
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.huggingface_backend:
        os.environ["HUGGINGFACE_BACKEND"] = args.huggingface_backend
    if args.huggingface_local_device:
        os.environ["HUGGINGFACE_LOCAL_DEVICE"] = args.huggingface_local_device
    if args.huggingface_local_dtype:
        os.environ["HUGGINGFACE_LOCAL_DTYPE"] = args.huggingface_local_dtype

    resolved_models = _resolve_model_specs(args)
    model_configs = _parse_models(resolved_models)
    _ensure_template_written(args)

    cases = _load_cases(args)
    events = load_event_corpus(args.events)

    run_dir = create_run_dir(args.output_dir)

    model_results: Dict[str, List[QueryRunResult]] = {}
    summaries = []

    for cfg in model_configs:
        cfg.timeout_s = args.timeout_s
        cfg.temperature = args.temperature
        cfg.max_tokens = args.max_tokens

        adapter = build_adapter(cfg)
        model_key = f"{cfg.provider}:{cfg.model_name}"
        rows: List[QueryRunResult] = []

        print(f"Running {model_key} on {len(cases)} cases x {args.runs_per_query} run(s) each")

        for case in cases:
            for run_idx in range(args.runs_per_query):
                inv: ModelInvocationResult = adapter.extract(case.query)
                inv.metadata["run_index"] = run_idx
                inv.metadata["case_id"] = case.case_id
                predicted_urls = retrieve_ranked_urls(events, inv, top_n=args.top_n)
                rows.append(QueryRunResult(case=case, invocation=inv, predicted_urls=predicted_urls))

                if args.inter_query_delay_ms > 0:
                    time.sleep(args.inter_query_delay_ms / 1000.0)

        model_results[model_key] = rows
        summaries.append(summarize_model(rows))

    run_meta = {
        "profile": args.profile,
        "model_set": args.model_set,
        "dataset": args.dataset or (DEFAULT_SMOKE_DATASET if args.profile == "smoke" else "generated-template"),
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
        "case_count": len(cases),
        "event_count": len(events),
    }

    summary_json = write_summary_json(run_dir, summaries, run_meta)
    per_query_csv = write_per_query_csv(run_dir, model_results)
    summary_md = write_summary_markdown(run_dir, summaries)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
