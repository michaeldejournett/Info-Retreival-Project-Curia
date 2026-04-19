"""Stage 2: keyword expansion — chat few-shot + prefix KV cache + retrieval hybrid."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from local_models import generate_chat, generate_with_prefix, is_loaded, warmup_prefix_cache

logger = logging.getLogger(__name__)

LABEL = "expansion"

SYSTEM = (
    "You expand search queries into related keywords for a university event database. "
    "Respond with ONE JSON array of 20-30 lowercase strings and nothing else. "
    "Include the original topic, synonyms, specific examples, subcategories, and related roles. "
    "Do not include time words (tonight, morning, weekend) or generic stop words."
)

FEWSHOT: List[Tuple[str, str]] = [
    (
        "pizza party",
        '["food","pizza","pasta","italian","spaghetti","lasagna","calzone","dining","buffet",'
        '"snacks","potluck","dinner","lunch","meal","restaurant","cuisine","eating","social",'
        '"gathering","party","celebration","refreshments","free food","catering"]',
    ),
    (
        "data science talks",
        '["data science","machine learning","ai","analytics","statistics","python","r",'
        '"visualization","deep learning","neural network","nlp","big data","research","seminar",'
        '"lecture","workshop","stem","technology","algorithms","modeling","prediction",'
        '"conference","presentation","speaker","career","internship"]',
    ),
    (
        "weekend concerts",
        '["music","concert","band","jazz","rock","pop","indie","folk","blues","classical",'
        '"orchestra","choir","live music","performance","show","gig","recital","singer",'
        '"musician","acoustic","electronic","hip hop","r&b","open mic","venue","entertainment"]',
    ),
]


def _prefix_messages() -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        msgs.append({"role": "user", "content": f"Query: {q}"})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def warmup() -> None:
    warmup_prefix_cache(LABEL, _prefix_messages())


def _parse_array(text: str) -> Optional[List[str]]:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    out: List[str] = []
    seen: set = set()
    for item in items:
        s = re.sub(r"\s+", " ", str(item).strip().lower())
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out or None


def expand_keywords(query: str) -> Optional[List[str]]:
    """LLM expansion (prefix-cached fast path) + retrieval hybrid."""
    llm_terms: List[str] = []

    if is_loaded():
        raw = generate_with_prefix(LABEL, f"Query: {query}", max_new_tokens=160)
        if raw is None:
            msgs = _prefix_messages()
            msgs.append({"role": "user", "content": f"Query: {query}"})
            raw = generate_chat(msgs, max_new_tokens=160)
        parsed = _parse_array(raw) if raw else None
        if parsed:
            llm_terms = parsed
        else:
            logger.info("Expansion LLM non-JSON output: %s", (raw or "")[:150])

    # Hybrid: augment with retrieval over keywords.js vocabulary
    try:
        from retrieval import retrieve_related, retrieval_loaded
        if retrieval_loaded():
            seeds = llm_terms if llm_terms else [w for w in query.lower().split() if len(w) > 2]
            retrieved = retrieve_related(seeds, top_k=10)
            seen = set(llm_terms)
            for t in retrieved:
                if t not in seen:
                    llm_terms.append(t)
                    seen.add(t)
    except Exception as exc:
        logger.debug("Retrieval augmentation skipped: %s", exc)

    return llm_terms if llm_terms else None
