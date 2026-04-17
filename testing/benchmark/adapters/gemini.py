from __future__ import annotations

import os
import time
from typing import Any, Dict

from testing.benchmark.env_loader import load_project_env_once
from testing.benchmark.normalize import normalize_from_model_json, parse_json_object
from testing.benchmark.schemas import ModelInvocationResult

from .base import BaseModelAdapter


class GeminiAdapter(BaseModelAdapter):
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

        raw_text = ""
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=self.build_prompt(query),
            )
            raw_text = (getattr(response, "text", "") or "").strip()
            payload, parse_error = parse_json_object(raw_text)
            if payload is None:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    raw_text=raw_text,
                    error=parse_error,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            normalized: Dict[str, Any] = normalize_from_model_json(payload)
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                keywords=normalized["keywords"],
                date_from=normalized["date_from"],
                date_to=normalized["date_to"],
                time_from=normalized["time_from"],
                time_to=normalized["time_to"],
                raw_text=raw_text,
                parse_success=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except TimeoutError:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                raw_text=raw_text,
                timed_out=True,
                error="timeout",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                raw_text=raw_text,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
