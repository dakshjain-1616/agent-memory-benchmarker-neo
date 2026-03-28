"""Mem0-backed memory store.

In real mode (MEM0_API_KEY or OPENAI_API_KEY set), uses mem0ai with a local
ChromaDB + OpenAI embeddings config.
In mock mode (no keys), falls back to an in-memory dict with TF-IDF retrieval.
"""

from __future__ import annotations
import os
import math
from collections import defaultdict
from typing import Any

from .base import MemoryBackend


# ── TF-IDF helpers for mock mode ─────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    total = len(tokens) or 1
    return {k: v / total for k, v in counts.items()}


def _cosine_tfidf(query: str, docs: list[str]) -> list[float]:
    """Return cosine similarity between *query* and each doc using TF-IDF."""
    if not docs:
        return []
    corpus = [query] + docs
    tokenized = [_tokenize(d) for d in corpus]
    # IDF over corpus
    N = len(corpus)
    df: dict[str, int] = defaultdict(int)
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
    # Build TF-IDF vectors
    def vec(toks: list[str]) -> dict[str, float]:
        tf = _tf(toks)
        return {t: tf[t] * idf.get(t, 1) for t in tf}
    q_vec = vec(tokenized[0])
    scores = []
    for toks in tokenized[1:]:
        d_vec = vec(toks)
        # Cosine
        common = set(q_vec) & set(d_vec)
        dot = sum(q_vec[t] * d_vec[t] for t in common)
        norm_q = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1.0
        norm_d = math.sqrt(sum(v ** 2 for v in d_vec.values())) or 1.0
        scores.append(dot / (norm_q * norm_d))
    return scores


# ── Mem0 Backend ──────────────────────────────────────────────────────────────

class Mem0Backend(MemoryBackend):
    """Mem0 memory backend with graceful mock fallback."""

    name = "mem0"

    def __init__(self) -> None:
        self._token_usage: int = 0
        self._mem0 = None
        self._user_id = os.getenv("MEM0_USER_ID", "benchmark_user")
        # Try to initialize real Mem0
        if self._has_keys():
            self._mem0 = self._init_mem0()
        # Always maintain a fallback store
        self._mock_store: list[str] = []

    def _has_keys(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("MEM0_API_KEY"))

    def _init_mem0(self):
        """Return a configured mem0 Memory instance, or None on failure."""
        try:
            from mem0 import Memory  # type: ignore

            config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": os.getenv("MEM0_LLM_MODEL", "openai/gpt-5.4-mini"),
                        "api_key": os.getenv("OPENAI_API_KEY", ""),
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": os.getenv("MEM0_EMBED_MODEL", "text-embedding-ada-002"),
                        "api_key": os.getenv("OPENAI_API_KEY", ""),
                    },
                },
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": os.getenv("MEM0_COLLECTION_NAME", "mem0_benchmark"),
                        "path": os.getenv("MEM0_CHROMA_PATH", "/tmp/mem0_chroma"),
                    },
                },
            }
            return Memory.from_config(config)
        except Exception:
            return None

    # ── MemoryBackend interface ───────────────────────────────────────────────

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._mock_store.append(content)
        if self._mem0 is not None:
            try:
                self._mem0.add(content, user_id=self._user_id, metadata=metadata or {})
            except Exception:
                pass  # degrade to mock

    def query(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        # Try real Mem0 first
        if self._mem0 is not None:
            try:
                results = self._mem0.search(query, user_id=self._user_id, limit=top_k)
                memories = results if isinstance(results, list) else results.get("results", [])
                output = []
                for m in memories[:top_k]:
                    text = m.get("memory", m.get("content", str(m)))
                    score = float(m.get("score", 0.5))
                    output.append({"content": text, "score": score})
                if output:
                    return output
            except Exception:
                pass

        # Mock fallback: TF-IDF retrieval
        if not self._mock_store:
            return []
        scores = _cosine_tfidf(query, self._mock_store)
        ranked = sorted(zip(scores, self._mock_store), reverse=True)
        return [
            {"content": text, "score": max(0.0, score)}
            for score, text in ranked[:top_k]
        ]

    def clear(self) -> None:
        self._mock_store = []
        if self._mem0 is not None:
            try:
                self._mem0.delete_all(user_id=self._user_id)
            except Exception:
                pass
        self._token_usage = 0
