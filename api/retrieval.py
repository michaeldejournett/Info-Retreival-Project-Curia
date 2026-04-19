"""Retrieval-based keyword expansion using sentence-transformers over the keywords.js corpus."""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from keyword_corpus import CATEGORIES, SPECIFIC_TERMS, TERM_TO_PARENTS

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDER = None
_CORPUS_EMBS: Optional[np.ndarray] = None  # (len(SPECIFIC_TERMS), 384)
_LOAD_LOCK = threading.Lock()


def load_retrieval_model() -> bool:
    global _EMBEDDER, _CORPUS_EMBS
    with _LOAD_LOCK:
        if _EMBEDDER is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading retrieval model %s", _MODEL_NAME)
            model = SentenceTransformer(_MODEL_NAME)
            embs = model.encode(SPECIFIC_TERMS, convert_to_numpy=True, normalize_embeddings=True)
            _EMBEDDER = model
            _CORPUS_EMBS = embs
            logger.info("Retrieval model ready — %d terms indexed", len(SPECIFIC_TERMS))
            return True
        except Exception as exc:
            logger.warning("Failed to load retrieval model: %s", exc)
            return False


def retrieval_loaded() -> bool:
    return _EMBEDDER is not None and _CORPUS_EMBS is not None


def retrieve_related(seeds: List[str], top_k: int = 8) -> List[str]:
    """Given seed keywords, retrieve top-K related specific terms + their parent categories."""
    if not retrieval_loaded() or not seeds:
        return []
    try:
        seed_embs = _EMBEDDER.encode(seeds, convert_to_numpy=True, normalize_embeddings=True)
        # Mean pool seed embeddings into a single query vector
        query = seed_embs.mean(axis=0)
        query /= np.linalg.norm(query) + 1e-9
        scores = _CORPUS_EMBS @ query  # cosine similarity (already normalized)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: List[str] = []
        seen = set(s.lower() for s in seeds)
        parent_cats: set = set()
        for i in top_idx:
            term = SPECIFIC_TERMS[i]
            if term not in seen:
                results.append(term)
                seen.add(term)
            for cat in TERM_TO_PARENTS.get(term, []):
                parent_cats.add(cat)
        # Add parent categories not already in seeds
        for cat in sorted(parent_cats):
            if cat not in seen:
                results.append(cat)
        return results
    except Exception as exc:
        logger.warning("Retrieval failed: %s", exc)
        return []
