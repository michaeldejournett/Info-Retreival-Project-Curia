"""Loaders for two local models: temporal tagger (Stage 1) and expansion LM (Stage 2)."""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

TEMPORAL_MODEL_ID = os.getenv("TEMPORAL_MODEL_ID", "satyaalmasian/temporal_tagger_roberta2roberta")
EXPANSION_MODEL_ID = os.getenv("EXPANSION_MODEL_ID", "google/flan-t5-base")

_TEMPORAL_MODEL = None
_TEMPORAL_TOKENIZER = None
_EXPANSION_MODEL = None
_EXPANSION_TOKENIZER = None
_TEMPORAL_LOCK = threading.Lock()
_EXPANSION_LOCK = threading.Lock()


def _load_temporal() -> bool:
    global _TEMPORAL_MODEL, _TEMPORAL_TOKENIZER
    if _TEMPORAL_MODEL is not None:
        return True
    try:
        import torch
        from transformers import AutoTokenizer, EncoderDecoderModel

        logger.info("Loading temporal tagger %s", TEMPORAL_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(TEMPORAL_MODEL_ID)
        model = EncoderDecoderModel.from_pretrained(TEMPORAL_MODEL_ID, torch_dtype=torch.float32)
        model.eval()
        _TEMPORAL_TOKENIZER = tokenizer
        _TEMPORAL_MODEL = model
        logger.info("Temporal tagger ready")
        return True
    except Exception as exc:
        logger.warning("Failed to load temporal tagger: %s", exc)
        return False


def _load_expansion() -> bool:
    global _EXPANSION_MODEL, _EXPANSION_TOKENIZER
    if _EXPANSION_MODEL is not None:
        return True
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading expansion model %s", EXPANSION_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(EXPANSION_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(EXPANSION_MODEL_ID, torch_dtype=torch.float32)
        model.eval()
        _EXPANSION_TOKENIZER = tokenizer
        _EXPANSION_MODEL = model
        logger.info("Expansion model ready")
        return True
    except Exception as exc:
        logger.warning("Failed to load expansion model: %s", exc)
        return False


def load_local_models() -> bool:
    """Load both models. Returns True only if both succeed."""
    ok_t = _load_temporal()
    ok_e = _load_expansion()
    return ok_t and ok_e


def temporal_loaded() -> bool:
    return _TEMPORAL_MODEL is not None and _TEMPORAL_TOKENIZER is not None


def expansion_loaded() -> bool:
    return _EXPANSION_MODEL is not None and _EXPANSION_TOKENIZER is not None


def is_loaded() -> bool:
    return temporal_loaded() and expansion_loaded()


def generate_temporal(query: str, dct: str, max_new_tokens: int = 256) -> Optional[str]:
    """Run the temporal tagger. `dct` is the document creation date (YYYY-MM-DD)."""
    if not temporal_loaded():
        return None
    import torch

    input_text = f"date: {dct} text: {query}"
    with _TEMPORAL_LOCK:
        try:
            inputs = _TEMPORAL_TOKENIZER(
                input_text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.inference_mode():
                output_ids = _TEMPORAL_MODEL.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            return _TEMPORAL_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
        except Exception as exc:
            logger.warning("Temporal tagger generate failed: %s", exc)
            return None


def generate_expansion(prompt: str, max_new_tokens: int = 512) -> Optional[str]:
    if not expansion_loaded():
        return None
    import torch

    with _EXPANSION_LOCK:
        try:
            inputs = _EXPANSION_TOKENIZER(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            with torch.inference_mode():
                output_ids = _EXPANSION_MODEL.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            return _EXPANSION_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
        except Exception as exc:
            logger.warning("Expansion generate failed: %s", exc)
            return None
