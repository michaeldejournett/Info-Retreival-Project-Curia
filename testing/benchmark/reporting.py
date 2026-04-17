from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from testing.benchmark.schemas import ModelRunSummary, QueryRunResult


def create_run_dir(base_dir: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = Path(base_dir) / f"run-{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary_json(path: Path, summaries: Iterable[ModelRunSummary], run_meta: Dict[str, object]) -> Path:
    payload = {
        "run_meta": run_meta,
        "summaries": [asdict(s) for s in summaries],
    }
    out = path / "summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def write_per_query_csv(path: Path, model_results: Dict[str, List[QueryRunResult]]) -> Path:
    out = path / "per_query.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "model_key",
            "provider",
            "model_name",
            "case_id",
            "query",
            "tags",
            "parse_success",
            "timed_out",
            "error",
            "latency_ms",
            "keywords_pred",
            "date_from_pred",
            "date_to_pred",
            "time_from_pred",
            "time_to_pred",
            "predicted_urls",
            "expected_keywords",
            "expected_date_from",
            "expected_date_to",
            "expected_time_from",
            "expected_time_to",
            "expected_relevant_urls",
        ])

        for model_key, rows in model_results.items():
            for item in rows:
                inv = item.invocation
                exp = item.case.expected
                writer.writerow([
                    model_key,
                    inv.provider,
                    inv.model_name,
                    item.case.case_id,
                    item.case.query,
                    "|".join(item.case.tags),
                    int(inv.parse_success),
                    int(inv.timed_out),
                    inv.error or "",
                    f"{inv.latency_ms:.3f}",
                    "|".join(inv.keywords),
                    inv.date_from or "",
                    inv.date_to or "",
                    inv.time_from or "",
                    inv.time_to or "",
                    "|".join(item.predicted_urls),
                    "|".join(exp.keywords),
                    exp.date_from or "",
                    exp.date_to or "",
                    exp.time_from or "",
                    exp.time_to or "",
                    "|".join(exp.relevant_event_urls),
                ])
    return out


def write_summary_markdown(path: Path, summaries: Iterable[ModelRunSummary]) -> Path:
    out = path / "summary.md"
    lines = [
        "# Benchmark Summary",
        "",
        "| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for s in summaries:
        key = f"{s.provider}:{s.model_name}"
        keyword_f1 = "n/a" if s.keyword_f1 is None else f"{s.keyword_f1:.3f}"
        top3 = "n/a" if s.top3_hit_rate is None else f"{s.top3_hit_rate:.3f}"
        lines.append(
            f"| {key} | {s.parse_success_rate:.3f} | {s.latency_p50_ms:.1f} | {s.latency_p95_ms:.1f} | {keyword_f1} | {top3} | {s.timeout_rate:.3f} |"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_gate_results_json(path: Path, gate_profile: str, gate_results: List[Dict[str, object]]) -> Path:
    out = path / "gate_results.json"
    payload = {
        "gate_profile": gate_profile,
        "model_results": gate_results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def write_gate_results_markdown(path: Path, gate_profile: str, gate_results: List[Dict[str, object]]) -> Path:
    out = path / "gate_results.md"
    lines = [
        "# Gate Results",
        "",
        f"Gate profile: `{gate_profile}`",
        "",
        "| Model | Passed | Failed Checks |",
        "|---|---:|---|",
    ]

    for row in gate_results:
        model_key = str(row.get("model_key") or "")
        passed = bool(row.get("passed"))
        checks = row.get("checks") or []
        failed = [
            str(c.get("name"))
            for c in checks
            if isinstance(c, dict) and not bool(c.get("passed"))
        ]
        failed_text = ", ".join(failed) if failed else "-"
        lines.append(f"| {model_key} | {'yes' if passed else 'no'} | {failed_text} |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def append_visuals_to_summary(summary_md_path: Path, visual_artifacts: List[str]) -> None:
    if not visual_artifacts:
        return

    existing = summary_md_path.read_text(encoding="utf-8")
    rel_paths = [Path(p).as_posix() for p in visual_artifacts]
    lines = [
        existing.rstrip(),
        "",
        "## Visual Artifacts",
        "",
    ]
    lines.extend([f"- `{p}`" for p in rel_paths])
    lines.append("")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
