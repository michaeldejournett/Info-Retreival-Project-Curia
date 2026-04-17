"""Stage 2: semantic keyword expansion via a local seq2seq model."""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from local_models import expansion_loaded, generate_expansion

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """Expand the topics in the search query into related keywords for a university event database.

Rules:
- Do NOT include time words ("tonight", "weekend", "morning", "6pm").
- Do NOT include stop words ("looking", "events", "find").
- Include the original topic plus specific examples, synonyms, and subcategories.
  - "food" -> ["food", "pizza", "pasta", "tacos", "burger", "salad", "bbq", "sushi"]
  - "music" -> ["music", "concert", "jazz", "rock", "band", "choir", "orchestra"]
  - "volunteer" -> ["volunteer", "service", "community", "outreach", "nonprofit"]
- Lowercase, single words or short phrases.
- Aim for 30-60 keywords for general queries, fewer for specific queries.

Return ONLY a JSON array of strings.

Query: "{query}"
JSON:"""


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
    out = []
    seen = set()
    for item in items:
        s = str(item).strip().lower()
        s = re.sub(r"\s+", " ", s)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out or None


def expand_keywords(query: str) -> Optional[List[str]]:
    if not expansion_loaded():
        return None
    prompt = EXPANSION_PROMPT.format(query=query)
    raw = generate_expansion(prompt, max_new_tokens=512)
    if not raw:
        return None
    result = _parse_array(raw)
    if result is None:
        logger.info("Expansion stage produced non-JSON output: %s", raw[:200])
    return result
