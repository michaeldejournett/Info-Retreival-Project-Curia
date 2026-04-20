from __future__ import annotations

import os
import time
from typing import Any, Dict

try:
    import requests
except Exception:  # pragma: no cover - dependency/environment specific
    requests = None

from testing.benchmark.normalize import normalize_from_model_json, parse_json_object
from testing.benchmark.schemas import ModelInvocationResult

from .base import BaseModelAdapter


class OllamaAdapter(BaseModelAdapter):
    def extract(self, query: str) -> ModelInvocationResult:
        started = time.perf_counter()
        if requests is None:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                error="requests package is not installed",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base_url.rstrip('/')}/api/generate"

        body: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": self.build_prompt(query),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        raw_text = ""
        try:
            response = requests.post(url, json=body, timeout=self.config.timeout_s)
            if response.status_code >= 400:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    error=f"http {response.status_code}: {response.text[:500]}",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            payload = response.json()
            raw_text = str(payload.get("response") or "").strip()
            parsed, parse_error = parse_json_object(raw_text)
            if parsed is None:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    raw_text=raw_text,
                    error=parse_error,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            normalized: Dict[str, Any] = normalize_from_model_json(parsed)
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
        except requests.Timeout:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                timed_out=True,
                raw_text=raw_text,
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
