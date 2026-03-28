"""FAISS-backed memory store with a bag-of-words hash embedding."""

from __future__ import annotations
import numpy as np
from typing import Any

from .base import MemoryBackend

_DIM = 256  # embedding dimension


def _embed(text: str) -> np.ndarray:
    """Deterministic, dependency-free embedding using word + trigram hashing."""
    vec = np.zeros(_DIM, dtype=np.float32)
    for word in text.lower().split():
        h = abs(hash(word)) % _DIM
        vec[h] += 1.0
    chars = text.lower()
    for i in range(len(chars) - 2):
        h = abs(hash(chars[i : i + 3])) % _DIM
        vec[h] += 0.3
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


class FAISSBackend(MemoryBackend):
    """Flat-L2 FAISS index storing arbitrary text memories."""

    name = "faiss"

    def __init__(self) -> None:
        import faiss  # lazy import

        self._index = faiss.IndexFlatL2(_DIM)
        self._texts: list[dict[str, Any]] = []
        self._token_usage: int = 0

    # ── MemoryBackend interface ───────────────────────────────────────────────

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        vec = _embed(content).reshape(1, -1)
        self._index.add(vec)
        self._texts.append({"content": content, "metadata": metadata or {}})

    def query(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if len(self._texts) == 0:
            return []
        vec = _embed(query).reshape(1, -1)
        k = min(top_k, len(self._texts))
        distances, indices = self._index.search(vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # L2 distance to similarity
            score = max(0.0, 1.0 - float(dist) / 2.0)
            results.append({"content": self._texts[idx]["content"], "score": score})
        return results

    def clear(self) -> None:
        import faiss

        self._index = faiss.IndexFlatL2(_DIM)
        self._texts = []
        self._token_usage = 0
