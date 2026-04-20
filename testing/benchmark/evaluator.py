from __future__ import annotations

import math
import statistics
from datetime import date
from typing import List, Optional, Sequence, Tuple

from testing.benchmark.schemas import ModelRunSummary, QueryRunResult


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(s[lo])
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _keyword_metrics(results: Sequence[QueryRunResult]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for r in results:
        exp = set(r.case.expected.keywords)
        if not exp:
            continue
        pred = set(r.invocation.keywords)
        tp = len(pred.intersection(exp))
        p = tp / len(pred) if pred else 0.0
        rc = tp / len(exp)
        f1 = (2 * p * rc / (p + rc)) if (p + rc) else 0.0
        precisions.append(p)
        recalls.append(rc)
        f1s.append(f1)

    return _safe_mean(precisions), _safe_mean(recalls), _safe_mean(f1s), len(precisions)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _range_overlap_ratio(
    expected: Tuple[date, date],
    predicted: Tuple[date, date],
) -> float:
    e0, e1 = expected
    p0, p1 = predicted
    left = max(e0, p0)
    right = min(e1, p1)
    if right < left:
        return 0.0
    overlap = (right - left).days + 1
    union = (max(e1, p1) - min(e0, p0)).days + 1
    return overlap / union if union > 0 else 0.0


def _date_metrics(results: Sequence[QueryRunResult]) -> Tuple[Optional[float], Optional[float], int]:
    exact_values: List[float] = []
    overlap_values: List[float] = []

    for r in results:
        exp_start = _parse_date(r.case.expected.date_from)
        exp_end = _parse_date(r.case.expected.date_to) or exp_start
        if not exp_start or not exp_end:
            continue

        pred_start = _parse_date(r.invocation.date_from)
        pred_end = _parse_date(r.invocation.date_to) or pred_start

        if not pred_start or not pred_end:
            exact_values.append(0.0)
            overlap_values.append(0.0)
            continue

        exact_values.append(1.0 if (exp_start == pred_start and exp_end == pred_end) else 0.0)
        overlap_values.append(_range_overlap_ratio((exp_start, exp_end), (pred_start, pred_end)))

    return _safe_mean(exact_values), _safe_mean(overlap_values), len(exact_values)


def _to_minutes(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
    except ValueError:
        return None
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return hh * 60 + mm


def _time_metrics(results: Sequence[QueryRunResult]) -> Tuple[Optional[float], Optional[float], int]:
    exact_values: List[float] = []
    partial_values: List[float] = []

    for r in results:
        exp_from = _to_minutes(r.case.expected.time_from)
        exp_to = _to_minutes(r.case.expected.time_to)
        if exp_from is None and exp_to is None:
            continue

        pred_from = _to_minutes(r.invocation.time_from)
        pred_to = _to_minutes(r.invocation.time_to)

        if pred_from is None and pred_to is None:
            exact_values.append(0.0)
            partial_values.append(0.0)
            continue

        exact_values.append(1.0 if (exp_from == pred_from and exp_to == pred_to) else 0.0)

        # Partial match is overlap of windows when both sides specify at least one boundary.
        e0 = exp_from if exp_from is not None else 0
        e1 = exp_to if exp_to is not None else 24 * 60
        p0 = pred_from if pred_from is not None else 0
        p1 = pred_to if pred_to is not None else 24 * 60
        overlap = max(0, min(e1, p1) - max(e0, p0))
        partial_values.append(1.0 if overlap > 0 else 0.0)

    return _safe_mean(exact_values), _safe_mean(partial_values), len(exact_values)


def _retrieval_metrics(results: Sequence[QueryRunResult]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int]:
    hits1 = 0
    hits3 = 0
    hits5 = 0
    reciprocal_ranks: List[float] = []
    labeled = 0

    for r in results:
        relevant = set(r.case.expected.relevant_event_urls)
        if not relevant:
            continue
        labeled += 1

        ranked = r.predicted_urls
        if any(u in relevant for u in ranked[:1]):
            hits1 += 1
        if any(u in relevant for u in ranked[:3]):
            hits3 += 1
        if any(u in relevant for u in ranked[:5]):
            hits5 += 1

        rr = 0.0
        for idx, url in enumerate(ranked, start=1):
            if url in relevant:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)

    if labeled == 0:
        return None, None, None, None, 0

    return (
        hits1 / labeled,
        hits3 / labeled,
        hits5 / labeled,
        _safe_mean(reciprocal_ranks),
        labeled,
    )


def summarize_model(results: Sequence[QueryRunResult]) -> ModelRunSummary:
    if not results:
        raise ValueError("Cannot summarize empty results")

    provider = results[0].invocation.provider
    model_name = results[0].invocation.model_name
    latencies = [r.invocation.latency_ms for r in results]

    parse_success_count = sum(1 for r in results if r.invocation.parse_success)
    timeout_count = sum(1 for r in results if r.invocation.timed_out)
    error_count = sum(1 for r in results if r.invocation.error)

    k_p, k_r, k_f1, keyword_labeled = _keyword_metrics(results)
    d_exact, d_overlap, date_labeled = _date_metrics(results)
    t_exact, t_partial, time_labeled = _time_metrics(results)
    top1, top3, top5, mrr, retrieval_labeled = _retrieval_metrics(results)

    return ModelRunSummary(
        provider=provider,
        model_name=model_name,
        total_queries=len(results),
        parse_success_rate=parse_success_count / len(results),
        timeout_rate=timeout_count / len(results),
        error_rate=error_count / len(results),
        latency_mean_ms=statistics.mean(latencies) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        latency_max_ms=max(latencies) if latencies else 0.0,
        keyword_precision=k_p,
        keyword_recall=k_r,
        keyword_f1=k_f1,
        date_exact_rate=d_exact,
        date_overlap_avg=d_overlap,
        time_exact_rate=t_exact,
        time_partial_rate=t_partial,
        top1_hit_rate=top1,
        top3_hit_rate=top3,
        top5_hit_rate=top5,
        mrr=mrr,
        details={
            "keyword_labeled_cases": keyword_labeled,
            "date_labeled_cases": date_labeled,
            "time_labeled_cases": time_labeled,
            "retrieval_labeled_cases": retrieval_labeled,
        },
    )
