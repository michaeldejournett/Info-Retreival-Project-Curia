from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from testing.benchmark.schemas import ModelRunSummary, QueryRunResult


def _plot_dependency_available():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _safe_metric(value: float | None, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    return float(value)


def generate_visual_artifacts(
    run_dir: Path,
    summaries: Sequence[ModelRunSummary],
    model_results: Dict[str, List[QueryRunResult]],
) -> Dict[str, object]:
    plt = _plot_dependency_available()
    if plt is None:
        return {
            "artifacts": [],
            "warning": "matplotlib is not installed; skipping chart generation",
        }

    if not summaries:
        return {"artifacts": [], "warning": "no model summaries available for chart generation"}

    output_dir = run_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[str] = []

    model_keys = [f"{s.provider}:{s.model_name}" for s in summaries]
    x = list(range(len(model_keys)))

    # Parse success per model
    parse_success = [s.parse_success_rate * 100.0 for s in summaries]
    plt.figure(figsize=(10, 5))
    plt.bar(x, parse_success)
    plt.xticks(x, model_keys, rotation=25, ha="right")
    plt.ylabel("Parse Success (%)")
    plt.title("Parse Success by Model")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    out = output_dir / "parse_success_by_model.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    artifacts.append(str(out))

    # Latency chart: p50 vs p95
    latency_p50 = [s.latency_p50_ms for s in summaries]
    latency_p95 = [s.latency_p95_ms for s in summaries]
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], latency_p50, width=width, label="p50")
    plt.bar([i + width / 2 for i in x], latency_p95, width=width, label="p95")
    plt.xticks(x, model_keys, rotation=25, ha="right")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Comparison by Model")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    out = output_dir / "latency_by_model.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    artifacts.append(str(out))

    # Timeout and error rates
    timeout_rates = [s.timeout_rate * 100.0 for s in summaries]
    error_rates = [s.error_rate * 100.0 for s in summaries]
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], timeout_rates, width=width, label="Timeout")
    plt.bar([i + width / 2 for i in x], error_rates, width=width, label="Error")
    plt.xticks(x, model_keys, rotation=25, ha="right")
    plt.ylabel("Rate (%)")
    plt.title("Timeout and Error Rates by Model")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    out = output_dir / "timeout_error_by_model.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    artifacts.append(str(out))

    # Accuracy vs latency scatter
    scatter_y = [
        _safe_metric(s.keyword_f1, fallback=s.parse_success_rate) * 100.0
        for s in summaries
    ]
    scatter_x = [s.latency_p95_ms for s in summaries]
    plt.figure(figsize=(8, 5))
    plt.scatter(scatter_x, scatter_y)
    for i, label in enumerate(model_keys):
        plt.text(scatter_x[i], scatter_y[i], label)
    plt.xlabel("Latency p95 (ms)")
    plt.ylabel("Quality (%)")
    plt.title("Quality vs Latency")
    plt.grid(linestyle="--", alpha=0.4)
    out = output_dir / "quality_vs_latency.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    artifacts.append(str(out))

    # Parse success by case tag
    tags = sorted({tag for rows in model_results.values() for r in rows for tag in r.case.tags})
    if tags:
        parse_by_model_tag: Dict[str, Dict[str, float]] = {}
        for model_key, rows in model_results.items():
            bucket: Dict[str, List[int]] = defaultdict(list)
            for row in rows:
                for tag in row.case.tags:
                    bucket[tag].append(1 if row.invocation.parse_success else 0)
            parse_by_model_tag[model_key] = {
                tag: (sum(values) / len(values) * 100.0) if values else 0.0
                for tag, values in bucket.items()
            }

        width_tag = 0.8 / max(1, len(model_keys))
        tag_x = list(range(len(tags)))
        plt.figure(figsize=(10, 5))
        for idx, model_key in enumerate(model_keys):
            shift = (idx - (len(model_keys) - 1) / 2) * width_tag
            vals = [parse_by_model_tag.get(model_key, {}).get(tag, 0.0) for tag in tags]
            plt.bar([i + shift for i in tag_x], vals, width=width_tag, label=model_key)
        plt.xticks(tag_x, tags, rotation=25, ha="right")
        plt.ylabel("Parse Success (%)")
        plt.title("Parse Success by Query Tag")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        out = output_dir / "parse_success_by_tag.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        artifacts.append(str(out))

    # Stability chart when repeated runs are present
    stability_present = any(s.details.get("run_count", 0) > 1 for s in summaries)
    if stability_present:
        parse_stddev = [
            _safe_metric(s.details.get("parse_success_stddev"), fallback=0.0) * 100.0 for s in summaries
        ]
        p95_stddev = [
            _safe_metric(s.details.get("latency_p95_stddev_ms"), fallback=0.0) for s in summaries
        ]

        plt.figure(figsize=(10, 5))
        plt.bar([i - width / 2 for i in x], parse_stddev, width=width, label="Parse stddev (%)")
        plt.bar([i + width / 2 for i in x], p95_stddev, width=width, label="Latency p95 stddev (ms)")
        plt.xticks(x, model_keys, rotation=25, ha="right")
        plt.ylabel("Stddev")
        plt.title("Stability Across Repeated Runs")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        out = output_dir / "stability_stddev.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        artifacts.append(str(out))

    return {"artifacts": artifacts, "warning": ""}
