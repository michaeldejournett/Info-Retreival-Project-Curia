from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from testing.benchmark.env_loader import load_project_env_once
from testing.benchmark.normalize import normalize_from_model_json, normalize_keywords, parse_json_object
from testing.benchmark.prompts import build_keywords_prompt, build_temporal_prompt
from testing.benchmark.schemas import ModelInvocationResult

from .base import BaseModelAdapter


def _parse_keyword_array(text: str) -> Optional[List[str]]:
    cleaned = (text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        items = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    return normalize_keywords(items) or None


class GeminiAdapter(BaseModelAdapter):
    # Gemini's genai client has resource cleanup issues in concurrent threads;
    # creating many clients in parallel exhausts connection pools and causes hangs.
    # Running serially prevents resource contention and client lifecycle problems.
    is_concurrent_safe = False

    def extract(self, query: str) -> ModelInvocationResult:
        started = time.perf_counter()
        load_project_env_once()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                error="missing GEMINI_API_KEY/GOOGLE_API_KEY",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency/environment specific
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                error=f"google-genai import failed: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        raw_temporal = ""
        raw_expansion = ""
        try:
            with genai.Client(api_key=api_key) as client:
                # Stage 1: temporal extraction (date/time)
                r1 = client.models.generate_content(
                    model=self.model_name,
                    contents=build_temporal_prompt(query),
                )
                raw_temporal = (getattr(r1, "text", "") or "").strip()

                # Stage 2: keyword expansion
                r2 = client.models.generate_content(
                    model=self.model_name,
                    contents=build_keywords_prompt(query),
                )
                raw_expansion = (getattr(r2, "text", "") or "").strip()

            raw_text = f"[temporal] {raw_temporal}\n[expansion] {raw_expansion}"

            temporal_payload, temporal_error = parse_json_object(raw_temporal)
            if temporal_payload is not None:
                norm: Dict[str, Any] = normalize_from_model_json(temporal_payload)
                date_from = norm["date_from"]
                date_to = norm["date_to"]
                time_from = norm["time_from"]
                time_to = norm["time_to"]
            else:
                date_from = date_to = time_from = time_to = None

            keywords = _parse_keyword_array(raw_expansion)

            if temporal_payload is None and keywords is None:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    raw_text=raw_text,
                    error=temporal_error or "keywords-parse-failed",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                keywords=keywords or [],
                date_from=date_from,
                date_to=date_to,
                time_from=time_from,
                time_to=time_to,
                raw_text=raw_text,
                parse_success=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except TimeoutError:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                raw_text=f"[temporal] {raw_temporal}\n[expansion] {raw_expansion}",
                timed_out=True,
                error="timeout",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                raw_text=f"[temporal] {raw_temporal}\n[expansion] {raw_expansion}",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
