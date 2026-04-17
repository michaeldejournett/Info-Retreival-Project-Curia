from __future__ import annotations

from typing import Dict


MODEL_SETS: Dict[str, str] = {
    "baseline": "gemini:gemma-3-27b-it,ollama:llama3.1",
    "hf-local-lite": ",".join(
        [
            "huggingface:MBZUAI/LaMini-Flan-T5-248M",
            "huggingface:google/flan-t5-base",
            "huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ]
    ),
    "hf-first-pass": ",".join(
        [
            "huggingface:MBZUAI/LaMini-Flan-T5-248M",
            "huggingface:google/flan-t5-base",
            "huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "huggingface:google/gemma-2-2b-it",
            "huggingface:Qwen/Qwen2.5-3B-Instruct",
            "huggingface:microsoft/Phi-3-mini-4k-instruct",
            "huggingface:mistralai/Mistral-7B-Instruct-v0.2",
        ]
    ),
    "all": ",".join(
        [
            "gemini:gemma-3-27b-it",
            "huggingface:MBZUAI/LaMini-Flan-T5-248M",
            "huggingface:google/flan-t5-base",
            "huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "huggingface:google/gemma-2-2b-it",
            "huggingface:Qwen/Qwen2.5-3B-Instruct",
            "huggingface:microsoft/Phi-3-mini-4k-instruct",
            "huggingface:mistralai/Mistral-7B-Instruct-v0.2",
        ]
    )
}

DEFAULT_MODEL_SET = "baseline"


def get_model_set(name: str) -> str:
    try:
        return MODEL_SETS[name]
    except KeyError as exc:
        options = ", ".join(sorted(MODEL_SETS.keys()))
        raise ValueError(f"Unknown model set '{name}'. Available sets: {options}") from exc
