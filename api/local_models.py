"""Single-model loader for an instruction-tuned local LM (default: Qwen2.5-0.5B-Instruct)."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

_MODEL = None
_TOKENIZER = None
_LOCK = threading.Lock()

# Per-stage prefix KV caches: label -> (past_key_values, n_prefix_tokens)
_PREFIX_CACHE: Dict[str, Tuple[Any, int]] = {}


def load_local_models() -> bool:
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return True
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local model %s", LOCAL_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        try:
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, dtype=torch.bfloat16)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, dtype=torch.float32)
        torch.set_num_threads(os.cpu_count() or 4)
        model.eval()
        _TOKENIZER = tokenizer
        _MODEL = model
        logger.info("Local model ready")
        return True
    except Exception as exc:
        logger.warning("Failed to load local model: %s", exc)
        return False


def is_loaded() -> bool:
    return _MODEL is not None and _TOKENIZER is not None


def warmup_prefix_cache(label: str, prefix_messages: List[Dict[str, str]]) -> None:
    """Pre-compute KV states for the static system+exemplar prefix. Call once after load."""
    if not is_loaded():
        return
    import torch
    prompt = _TOKENIZER.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=False
    )
    inputs = _TOKENIZER(prompt, return_tensors="pt")
    n_tokens = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = _MODEL(**inputs, use_cache=True)
    _PREFIX_CACHE[label] = (out.past_key_values, n_tokens)
    logger.info("Prefix KV cache warmed for '%s' (%d tokens)", label, n_tokens)


def _qwen_user_tail(content: str) -> str:
    """Format a single user turn + generation prompt in Qwen's chat format."""
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def generate_chat(messages: List[Dict[str, str]], max_new_tokens: int = 256) -> Optional[str]:
    """Run full conversation through the model (no prefix cache). Fallback path."""
    if not is_loaded():
        return None
    import torch
    with _LOCK:
        try:
            prompt = _TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = _TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=4096)
            input_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                output_ids = _MODEL.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=_TOKENIZER.eos_token_id,
                )
            return _TOKENIZER.decode(output_ids[0][input_len:], skip_special_tokens=True)
        except Exception as exc:
            logger.warning("generate_chat failed: %s", exc)
            return None


def generate_with_prefix(
    label: str, user_content: str, max_new_tokens: int = 256
) -> Optional[str]:
    """Generate using a pre-warmed KV prefix cache + a new user message. Fast path."""
    if not is_loaded() or label not in _PREFIX_CACHE:
        return None
    import torch
    with _LOCK:
        try:
            prefix_kv, _ = _PREFIX_CACHE[label]
            tail = _qwen_user_tail(user_content)
            tail_ids = _TOKENIZER(
                tail, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            with torch.inference_mode():
                output_ids = _MODEL.generate(
                    tail_ids,
                    past_key_values=prefix_kv,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=_TOKENIZER.eos_token_id,
                )
            new_tokens = output_ids[0][tail_ids.shape[1]:]
            return _TOKENIZER.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:
            logger.warning("generate_with_prefix failed for '%s': %s", label, exc)
            return None
