from __future__ import annotations

from typing import Dict


CORE_SEVEN_MODELS = ",".join(
    [
        "gemini:gemma-3-27b-it",
        "ollama:llama3:latest",
        "huggingface:Qwen/Qwen2.5-1.5B-Instruct",
        "huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "huggingface:Qwen/Qwen2.5-0.5B-Instruct",
        "huggingface:google/flan-t5-base",
        "huggingface:MBZUAI/LaMini-Flan-T5-248M",
    ]
)


MODEL_SETS: Dict[str, str] = {
    "core-seven": CORE_SEVEN_MODELS,
}

DEFAULT_MODEL_SET = "core-seven"


def get_model_set(name: str) -> str:
    try:
        return MODEL_SETS[name]
    except KeyError as exc:
        options = ", ".join(sorted(MODEL_SETS.keys()))
        raise ValueError(f"Unknown model set '{name}'. Available sets: {options}") from exc
