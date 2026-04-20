from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from testing.benchmark.schemas import BenchmarkCase, ExpectedLabels


def _coerce_expected(value: Dict[str, Any]) -> ExpectedLabels:
    return ExpectedLabels(
        keywords=[str(x).strip().lower() for x in (value.get("keywords") or []) if str(x).strip()],
        date_from=value.get("date_from"),
        date_to=value.get("date_to"),
        time_from=value.get("time_from"),
        time_to=value.get("time_to"),
        relevant_event_urls=[
            str(x).strip() for x in (value.get("relevant_event_urls") or []) if str(x).strip()
        ],
    )


def _coerce_case(value: Dict[str, Any]) -> BenchmarkCase:
    cid = str(value.get("id") or value.get("case_id") or "").strip()
    if not cid:
        raise ValueError("Benchmark case missing id")
    query = str(value.get("query") or "").strip()
    if not query:
        raise ValueError(f"Benchmark case {cid} missing query")

    expected_raw = value.get("expected") or {}
    if not isinstance(expected_raw, dict):
        raise ValueError(f"Benchmark case {cid} has non-object expected field")

    tags = [str(t).strip().lower() for t in (value.get("tags") or []) if str(t).strip()]

    return BenchmarkCase(
        case_id=cid,
        query=query,
        tags=tags,
        expected=_coerce_expected(expected_raw),
        notes=str(value.get("notes") or "").strip(),
    )


def load_dataset(path: str) -> Tuple[List[BenchmarkCase], Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases: Iterable[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    if isinstance(payload, dict):
        maybe_metadata = payload.get("metadata")
        if isinstance(maybe_metadata, dict):
            metadata = maybe_metadata
        raw_cases = payload.get("cases") or []
    elif isinstance(payload, list):
        raw_cases = payload
    else:
        raise ValueError("Dataset root must be an object or array")

    cases = [_coerce_case(c) for c in raw_cases]
    if not cases:
        raise ValueError("Dataset contains no cases")
    return cases, metadata


def load_cases(path: str) -> List[BenchmarkCase]:
    cases, _ = load_dataset(path)
    return cases


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def _is_iso_date(value: str) -> bool:
    if not _DATE_RE.match(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _is_hhmm(value: str) -> bool:
    if not _TIME_RE.match(value):
        return False
    try:
        hh, mm = value.split(":", 1)
        hhi = int(hh)
        mmi = int(mm)
    except ValueError:
        return False
    return 0 <= hhi <= 23 and 0 <= mmi <= 59


def validate_cases(
    cases: List[BenchmarkCase],
    strict_labels: bool = False,
    metadata: Dict[str, Any] | None = None,
    expected_label_status: str = "",
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("Dataset contains no cases")

    errors: List[str] = []
    seen: set[str] = set()

    label_status = str((metadata or {}).get("label_status") or "").strip().lower()
    expected_status = expected_label_status.strip().lower()
    if expected_status and label_status != expected_status:
        errors.append(
            f"metadata.label_status is '{label_status or 'missing'}' but expected '{expected_status}'"
        )
    if strict_labels and not expected_status and label_status == "pending-manual":
        errors.append("metadata.label_status is 'pending-manual'; strict mode requires reviewed labels")

    for case in cases:
        if case.case_id in seen:
            errors.append(f"duplicate case id: {case.case_id}")
        seen.add(case.case_id)

        exp = case.expected

        if exp.date_from and not _is_iso_date(exp.date_from):
            errors.append(f"{case.case_id}: expected.date_from must be YYYY-MM-DD")
        if exp.date_to and not _is_iso_date(exp.date_to):
            errors.append(f"{case.case_id}: expected.date_to must be YYYY-MM-DD")
        if exp.date_from and exp.date_to and exp.date_from > exp.date_to:
            errors.append(f"{case.case_id}: expected.date_from is after expected.date_to")

        if exp.time_from and not _is_hhmm(exp.time_from):
            errors.append(f"{case.case_id}: expected.time_from must be HH:MM")
        if exp.time_to and not _is_hhmm(exp.time_to):
            errors.append(f"{case.case_id}: expected.time_to must be HH:MM")
        if exp.time_from and exp.time_to and exp.time_from > exp.time_to:
            errors.append(f"{case.case_id}: expected.time_from is after expected.time_to")

        for url in exp.relevant_event_urls:
            if not (url.startswith("http://") or url.startswith("https://")):
                errors.append(f"{case.case_id}: expected.relevant_event_urls must use http(s)")
                break

        if strict_labels:
            has_signal = any(
                [
                    bool(exp.keywords),
                    bool(exp.date_from),
                    bool(exp.date_to),
                    bool(exp.time_from),
                    bool(exp.time_to),
                    bool(exp.relevant_event_urls),
                ]
            )
            if not has_signal:
                errors.append(f"{case.case_id}: strict mode requires at least one expected label signal")

    if errors:
        preview = errors[:25]
        suffix = "" if len(errors) <= 25 else f"\n... and {len(errors) - 25} more"
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(preview) + suffix)

    return {
        "case_count": len(cases),
        "label_status": label_status,
        "version": str((metadata or {}).get("version") or "").strip(),
        "strict_labels": strict_labels,
    }


def _template_topics() -> List[str]:
    return [
        "food",
        "music",
        "volunteer",
        "sports",
        "career",
        "research",
        "art",
        "dance",
        "coding",
        "startup",
        "wellness",
        "mental health",
        "international",
        "religion",
        "climate",
        "sustainability",
        "engineering",
        "medicine",
        "business",
        "finance",
        "community",
        "social",
        "networking",
        "leadership",
        "study",
    ]


def generate_template_cases(size: int = 100) -> List[BenchmarkCase]:
    if size <= 0:
        raise ValueError("Template size must be positive")

    topics = _template_topics()
    date_phrases = ["today", "tomorrow", "this weekend", "next week", "in april"]
    time_phrases = ["in the morning", "in the afternoon", "in the evening", "after 6pm", "before noon"]

    cases: List[BenchmarkCase] = []

    for i, topic in enumerate(topics, start=1):
        cases.append(
            BenchmarkCase(
                case_id=f"TOPIC-{i:03d}",
                query=f"{topic} events",
                tags=["topic"],
                notes="template: label expected fields manually",
            )
        )

    for i in range(25):
        topic = topics[i % len(topics)]
        phrase = date_phrases[i % len(date_phrases)]
        cases.append(
            BenchmarkCase(
                case_id=f"DATE-{i + 1:03d}",
                query=f"{topic} events {phrase}",
                tags=["topic", "date"],
                notes="template: label expected fields manually",
            )
        )

    for i in range(25):
        topic = topics[i % len(topics)]
        phrase = time_phrases[i % len(time_phrases)]
        cases.append(
            BenchmarkCase(
                case_id=f"TIME-{i + 1:03d}",
                query=f"{topic} events {phrase}",
                tags=["topic", "time"],
                notes="template: label expected fields manually",
            )
        )

    for i in range(25):
        topic = topics[i % len(topics)]
        d_phrase = date_phrases[i % len(date_phrases)]
        t_phrase = time_phrases[(i + 2) % len(time_phrases)]
        cases.append(
            BenchmarkCase(
                case_id=f"COMBO-{i + 1:03d}",
                query=f"{topic} events {d_phrase} {t_phrase}",
                tags=["topic", "date", "time"],
                notes="template: label expected fields manually",
            )
        )

    if len(cases) < size:
        idx = 1
        while len(cases) < size:
            topic = topics[idx % len(topics)]
            cases.append(
                BenchmarkCase(
                    case_id=f"EXTRA-{idx:03d}",
                    query=f"{topic} events near city campus",
                    tags=["topic"],
                    notes="template: label expected fields manually",
                )
            )
            idx += 1

    return cases[:size]


def write_template_dataset(path: str, size: int = 100) -> str:
    cases = generate_template_cases(size=size)
    payload = {
        "metadata": {
            "name": "curia-benchmark-template",
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "size": len(cases),
            "label_status": "pending-manual",
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

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_path)
