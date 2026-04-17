from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExpectedLabels:
    keywords: List[str] = field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    relevant_event_urls: List[str] = field(default_factory=list)


@dataclass
class BenchmarkCase:
    case_id: str
    query: str
    tags: List[str] = field(default_factory=list)
    expected: ExpectedLabels = field(default_factory=ExpectedLabels)
    notes: str = ""


@dataclass
class ModelInvocationResult:
    provider: str
    model_name: str
    keywords: List[str] = field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    raw_text: str = ""
    parse_success: bool = False
    latency_ms: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRunResult:
    case: BenchmarkCase
    invocation: ModelInvocationResult
    predicted_urls: List[str] = field(default_factory=list)


@dataclass
class ModelRunSummary:
    provider: str
    model_name: str
    total_queries: int
    parse_success_rate: float
    timeout_rate: float
    error_rate: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    keyword_precision: Optional[float]
    keyword_recall: Optional[float]
    keyword_f1: Optional[float]
    date_exact_rate: Optional[float]
    date_overlap_avg: Optional[float]
    time_exact_rate: Optional[float]
    time_partial_rate: Optional[float]
    top1_hit_rate: Optional[float]
    top3_hit_rate: Optional[float]
    top5_hit_rate: Optional[float]
    mrr: Optional[float]
    details: Dict[str, Any] = field(default_factory=dict)
