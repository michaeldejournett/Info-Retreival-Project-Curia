from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover - dependency/environment specific
    requests = None

from testing.benchmark.env_loader import load_project_env_once
from testing.benchmark.normalize import normalize_from_model_json, parse_json_object
from testing.benchmark.schemas import ModelInvocationResult

from .base import BaseModelAdapter


@dataclass
class _LocalRuntime:
    tokenizer: Any
    model: Any
    torch_module: Any
    is_encoder_decoder: bool
    device: str


_HF_LEGACY_MODELS_URL = "https://api-inference.huggingface.co/models"
_HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


def _extract_generated_text(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("generated_text", "summary_text", "translation_text", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    if isinstance(payload, list):
        chunks = []
        for item in payload:
            if isinstance(item, dict):
                for key in ("generated_text", "summary_text", "translation_text", "text"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        chunks.append(value.strip())
                        break
            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())

        if chunks:
            return "\n".join(chunks)

    return ""


def _extract_error(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        message = payload.get("error")
        if isinstance(message, str) and message.strip():
            return message.strip()

        if isinstance(message, dict):
            nested = message.get("message") or message.get("error") or message.get("detail")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

        fallback = payload.get("message") or payload.get("detail")
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    return None


def _extract_chat_completion_text(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

                text = choice.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    return _extract_generated_text(payload)


def _is_chat_completions_url(base_url: str) -> bool:
    return base_url.rstrip("/").lower().endswith("/v1/chat/completions")


def _is_legacy_models_url(base_url: str) -> bool:
    return base_url.rstrip("/").lower() == _HF_LEGACY_MODELS_URL


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_backend() -> str:
    value = (os.environ.get("HUGGINGFACE_BACKEND", "api") or "api").strip().lower()
    return value or "api"


def _is_local_base_url(base_url: str) -> bool:
    try:
        hostname = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _resolve_local_device(torch_module: Any) -> str:
    requested = (os.environ.get("HUGGINGFACE_LOCAL_DEVICE", "auto") or "auto").strip().lower()
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(torch_module.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("HUGGINGFACE_LOCAL_DEVICE=cuda but CUDA is not available")
        return "cuda"

    if requested == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("HUGGINGFACE_LOCAL_DEVICE=mps but MPS is not available")
        return "mps"

    if requested == "cpu":
        return "cpu"

    raise RuntimeError("Unsupported HUGGINGFACE_LOCAL_DEVICE value. Use auto, cpu, cuda, or mps")


def _resolve_local_dtype(torch_module: Any, device: str):
    requested = (os.environ.get("HUGGINGFACE_LOCAL_DTYPE", "auto") or "auto").strip().lower()
    if requested == "auto":
        if device in {"cuda", "mps"}:
            return torch_module.float16
        return None

    if requested in {"none", "default"}:
        return None
    if requested in {"float16", "fp16"}:
        return torch_module.float16
    if requested in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if requested in {"float32", "fp32"}:
        return torch_module.float32

    raise RuntimeError(
        "Unsupported HUGGINGFACE_LOCAL_DTYPE value. Use auto, float16, bfloat16, float32, or none"
    )


class HuggingFaceAdapter(BaseModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._local_runtime: Optional[_LocalRuntime] = None
        self._local_runtime_error: Optional[str] = None

    def _load_local_runtime(self) -> _LocalRuntime:
        if self._local_runtime is not None:
            return self._local_runtime
        if self._local_runtime_error:
            raise RuntimeError(self._local_runtime_error)

        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            message = (
                "local backend requires transformers and torch packages; "
                "install with: pip install transformers accelerate sentencepiece safetensors torch"
            )
            self._local_runtime_error = message
            raise RuntimeError(message) from exc

        trust_remote_code = _as_bool(os.environ.get("HUGGINGFACE_LOCAL_TRUST_REMOTE_CODE"), default=False)

        try:
            device = _resolve_local_device(torch)
            dtype = _resolve_local_dtype(torch, device)

            model_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
            model_cls = (
                AutoModelForSeq2SeqLM
                if bool(getattr(model_config, "is_encoder_decoder", False))
                else AutoModelForCausalLM
            )

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
            }
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype

            model = model_cls.from_pretrained(self.model_name, **model_kwargs)
            model = model.to(device)
            model.eval()

            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            runtime = _LocalRuntime(
                tokenizer=tokenizer,
                model=model,
                torch_module=torch,
                is_encoder_decoder=bool(getattr(model_config, "is_encoder_decoder", False)),
                device=device,
            )
            self._local_runtime = runtime
            return runtime
        except Exception as exc:
            self._local_runtime_error = str(exc)
            raise

    def _extract_local(self, query: str, started: float) -> ModelInvocationResult:
        try:
            runtime = self._load_local_runtime()
        except Exception as exc:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        prompt = self.build_prompt(query)
        raw_text = ""

        try:
            tokenizer = runtime.tokenizer
            model = runtime.model
            torch = runtime.torch_module

            encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
            encoded = {k: v.to(runtime.device) for k, v in encoded.items()}

            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": max(32, min(self.config.max_tokens, 1024)),
                "do_sample": self.config.temperature > 0.0,
            }
            if self.config.temperature > 0.0:
                generate_kwargs["temperature"] = max(0.01, self.config.temperature)
            if tokenizer.pad_token_id is not None:
                generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

            with torch.no_grad():
                output_ids = model.generate(**encoded, **generate_kwargs)

            if runtime.is_encoder_decoder:
                raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            else:
                input_len = int(encoded["input_ids"].shape[-1])
                completion_ids = output_ids[0][input_len:]
                raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                if not raw_text:
                    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                    raw_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text

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
        except Exception as exc:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                raw_text=raw_text,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    def _extract_api_legacy(self, query: str, started: float, base_url: str, token: Optional[str]) -> ModelInvocationResult:
        url = f"{base_url.rstrip('/')}/{self.model_name}"
        headers = {
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        body: Dict[str, Any] = {
            "inputs": self.build_prompt(query),
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": max(32, min(self.config.max_tokens, 1024)),
                "return_full_text": False,
            },
            "options": {
                "wait_for_model": True,
            },
        }

        raw_text = ""
        deadline = started + max(1.0, self.config.timeout_s)
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    timed_out=True,
                    raw_text=raw_text,
                    error="timeout",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            req_timeout = max(1.0, min(remaining, 60.0))
            try:
                response = requests.post(url, headers=headers, json=body, timeout=req_timeout)
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

            payload: Any
            try:
                payload = response.json()
            except ValueError:
                payload = None

            if response.status_code >= 400:
                message = _extract_error(payload)
                if response.status_code in {503, 429} and message and "loading" in message.lower():
                    time.sleep(min(1.5, max(0.25, remaining / 2)))
                    continue

                suffix = f": {message}" if message else ""
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    raw_text=raw_text,
                    error=f"http {response.status_code}{suffix}",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            if payload is None:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    raw_text=raw_text,
                    error="invalid-json-response",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            raw_text = _extract_generated_text(payload)
            if not raw_text:
                raw_text = json.dumps(payload)

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

    def _extract_api_chat(self, query: str, started: float, base_url: str, token: Optional[str]) -> ModelInvocationResult:
        headers = {
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": self.build_prompt(query),
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": max(32, min(self.config.max_tokens, 1024)),
        }

        raw_text = ""
        deadline = started + max(1.0, self.config.timeout_s)
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    timed_out=True,
                    raw_text=raw_text,
                    error="timeout",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            req_timeout = max(1.0, min(remaining, 60.0))
            try:
                response = requests.post(base_url, headers=headers, json=body, timeout=req_timeout)
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

            payload: Any
            try:
                payload = response.json()
            except ValueError:
                payload = None

            if response.status_code >= 400:
                message = _extract_error(payload)
                if response.status_code in {503, 429} and message and "loading" in message.lower():
                    time.sleep(min(1.5, max(0.25, remaining / 2)))
                    continue

                if (
                    response.status_code == 400
                    and message
                    and "not supported by any provider you have enabled" in message.lower()
                ):
                    message = (
                        f"{message}. Choose supported models from "
                        "https://router.huggingface.co/v1/models"
                    )

                suffix = f": {message}" if message else ""
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    raw_text=raw_text,
                    error=f"http {response.status_code}{suffix}",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            if payload is None:
                return ModelInvocationResult(
                    provider=self.provider,
                    model_name=self.model_name,
                    parse_success=False,
                    raw_text=raw_text,
                    error="invalid-json-response",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            raw_text = _extract_chat_completion_text(payload)
            if not raw_text:
                raw_text = json.dumps(payload)

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

    def _extract_api(self, query: str, started: float) -> ModelInvocationResult:
        if requests is None:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                error="requests package is not installed",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        base_url = (os.environ.get("HUGGINGFACE_BASE_URL") or _HF_ROUTER_CHAT_URL).strip()
        if not base_url:
            base_url = _HF_ROUTER_CHAT_URL

        needs_auth = not _is_local_base_url(base_url)
        token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")
        if needs_auth and not token:
            return ModelInvocationResult(
                provider=self.provider,
                model_name=self.model_name,
                parse_success=False,
                error="missing HUGGINGFACE_API_TOKEN/HF_TOKEN",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if _is_chat_completions_url(base_url):
            return self._extract_api_chat(query, started, base_url, token)

        legacy_result = self._extract_api_legacy(query, started, base_url, token)
        if _is_legacy_models_url(base_url) and legacy_result.error and legacy_result.error.startswith("http 404"):
            return self._extract_api_chat(query, started, _HF_ROUTER_CHAT_URL, token)
        return legacy_result

    def extract(self, query: str) -> ModelInvocationResult:
        started = time.perf_counter()
        load_project_env_once()

        backend = _resolve_backend()
        if backend == "local":
            return self._extract_local(query, started)
        if backend == "api":
            return self._extract_api(query, started)

        return ModelInvocationResult(
            provider=self.provider,
            model_name=self.model_name,
            parse_success=False,
            error="Unsupported HUGGINGFACE_BACKEND value. Use 'api' or 'local'.",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
