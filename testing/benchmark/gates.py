from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from testing.benchmark.evaluator import summarize_model
from testing.benchmark.schemas import ModelRunSummary, QueryRunResult


@dataclass(frozen=True)
class GateProfile:
    min_parse_success_rate: Optional[float] = None
    min_keyword_f1: Optional[float] = None
    min_date_exact_rate: Optional[float] = None
    min_time_partial_rate: Optional[float] = None
    max_latency_p95_ms: Optional[float] = None
    max_timeout_rate: Optional[float] = None
    max_error_rate: Optional[float] = None
    max_parse_success_stddev: Optional[float] = None
    max_keyword_f1_stddev: Optional[float] = None
    max_latency_p95_stddev_ms: Optional[float] = None


GATE_PROFILES: Dict[str, GateProfile] = {
    "none": GateProfile(),
    "correctness": GateProfile(
        min_parse_success_rate=0.70,
        min_keyword_f1=0.05,
        min_date_exact_rate=0.50,
        min_time_partial_rate=0.50,
    ),
    "latency-stability": GateProfile(
        max_latency_p95_ms=15000.0,
        max_timeout_rate=0.10,
        max_error_rate=0.25,
        max_parse_success_stddev=0.10,
        max_keyword_f1_stddev=0.10,
        max_latency_p95_stddev_ms=4000.0,
    ),
    "all": GateProfile(
        min_parse_success_rate=0.70,
        min_keyword_f1=0.05,
        min_date_exact_rate=0.50,
        min_time_partial_rate=0.50,
        max_latency_p95_ms=15000.0,
        max_timeout_rate=0.10,
        max_error_rate=0.25,
        max_parse_success_stddev=0.10,
        max_keyword_f1_stddev=0.10,
        max_latency_p95_stddev_ms=4000.0,
    ),
}


def _stddev(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)


def compute_stability_metrics(rows: Sequence[QueryRunResult]) -> Dict[str, Any]:
    by_run: Dict[int, List[QueryRunResult]] = defaultdict(list)
    for row in rows:
        run_index_raw = row.invocation.metadata.get("run_index", 0)
        try:
            run_index = int(run_index_raw)
        except (TypeError, ValueError):
            run_index = 0
        by_run[run_index].append(row)

    run_summaries: List[ModelRunSummary] = []
    for _, run_rows in sorted(by_run.items()):
        run_summaries.append(summarize_model(run_rows))

    parse_rates = [s.parse_success_rate for s in run_summaries]
    keyword_f1_values = [s.keyword_f1 for s in run_summaries if s.keyword_f1 is not None]
    latency_p95_values = [s.latency_p95_ms for s in run_summaries]

    return {
        "run_count": len(run_summaries),
        "parse_success_stddev": _stddev(parse_rates),
        "keyword_f1_stddev": _stddev(keyword_f1_values),
        "latency_p95_stddev_ms": _stddev(latency_p95_values),
    }


def resolve_gate_profile(name: str) -> GateProfile:
    key = (name or "none").strip().lower()
    if key not in GATE_PROFILES:
        options = ", ".join(sorted(GATE_PROFILES.keys()))
        raise ValueError(f"Unknown gate profile '{name}'. Available profiles: {options}")
    return GATE_PROFILES[key]


def _make_check(
    name: str,
    actual: Optional[float],
    expected: float,
    mode: str,
) -> Dict[str, Any]:
    passed = False
    if actual is not None:
        if mode == "min":
            passed = actual >= expected
        else:
            passed = actual <= expected

    comparator = ">=" if mode == "min" else "<="
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "comparator": comparator,
        "passed": passed,
    }


def evaluate_summary(summary: ModelRunSummary, profile: GateProfile) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    if profile.min_parse_success_rate is not None:
        checks.append(
            _make_check(
                "parse_success_rate",
                summary.parse_success_rate,
                profile.min_parse_success_rate,
                "min",
            )
        )
    if profile.min_keyword_f1 is not None:
        checks.append(_make_check("keyword_f1", summary.keyword_f1, profile.min_keyword_f1, "min"))
    if profile.min_date_exact_rate is not None:
        checks.append(
            _make_check("date_exact_rate", summary.date_exact_rate, profile.min_date_exact_rate, "min")
        )
    if profile.min_time_partial_rate is not None:
        checks.append(
            _make_check(
                "time_partial_rate",
                summary.time_partial_rate,
                profile.min_time_partial_rate,
                "min",
            )
        )
    if profile.max_latency_p95_ms is not None:
        checks.append(
            _make_check("latency_p95_ms", summary.latency_p95_ms, profile.max_latency_p95_ms, "max")
        )
    if profile.max_timeout_rate is not None:
        checks.append(_make_check("timeout_rate", summary.timeout_rate, profile.max_timeout_rate, "max"))
    if profile.max_error_rate is not None:
        checks.append(_make_check("error_rate", summary.error_rate, profile.max_error_rate, "max"))

    parse_stddev = summary.details.get("parse_success_stddev")
    keyword_stddev = summary.details.get("keyword_f1_stddev")
    latency_stddev = summary.details.get("latency_p95_stddev_ms")

    if profile.max_parse_success_stddev is not None:
        checks.append(
            _make_check(
                "parse_success_stddev",
                parse_stddev,
                profile.max_parse_success_stddev,
                "max",
            )
        )
    if profile.max_keyword_f1_stddev is not None:
        checks.append(
            _make_check(
                "keyword_f1_stddev",
                keyword_stddev,
                profile.max_keyword_f1_stddev,
                "max",
            )
        )
    if profile.max_latency_p95_stddev_ms is not None:
        checks.append(
            _make_check(
                "latency_p95_stddev_ms",
                latency_stddev,
                profile.max_latency_p95_stddev_ms,
                "max",
            )
        )

    return {
        "model_key": f"{summary.provider}:{summary.model_name}",
        "passed": all(c["passed"] for c in checks) if checks else True,
        "checks": checks,
    }
