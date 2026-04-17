from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from testing.benchmark.prompts import build_expand_prompt


@dataclass
class AdapterConfig:
    provider: str
    model_name: str
    timeout_s: float = 20.0
    temperature: float = 0.0
    max_tokens: int = 1200
    metadata: Optional[Dict[str, str]] = None


class BaseModelAdapter(ABC):
    def __init__(self, config: AdapterConfig):
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def build_prompt(self, query: str) -> str:
        return build_expand_prompt(query)

    @abstractmethod
    def extract(self, query: str):
        raise NotImplementedError


def build_adapter(config: AdapterConfig) -> BaseModelAdapter:
    provider = config.provider.lower()
    if provider == "gemini":
        from .gemini import GeminiAdapter

        return GeminiAdapter(config)
    if provider in {"huggingface", "hf"}:
        from .huggingface import HuggingFaceAdapter

        return HuggingFaceAdapter(config)
    if provider == "ollama":
        from .ollama import OllamaAdapter

        return OllamaAdapter(config)

    raise ValueError(f"Unsupported provider: {config.provider}")
