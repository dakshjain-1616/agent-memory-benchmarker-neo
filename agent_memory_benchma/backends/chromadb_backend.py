"""ChromaDB-backed memory store using a deterministic hash embedding function."""

from __future__ import annotations
import uuid
import numpy as np
from typing import Any

from .base import MemoryBackend


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """Fast, deterministic bag-of-words-style embedding (no model download)."""
    vec = np.zeros(dim, dtype=np.float32)
    for word in text.lower().split():
        h = abs(hash(word)) % dim
        vec[h] += 1.0
    # also use character trigrams for better recall
    chars = text.lower()
    for i in range(len(chars) - 2):
        trigram = chars[i : i + 3]
        h = abs(hash(trigram)) % dim
        vec[h] += 0.3
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


class _HashEmbeddingFunction:
    """chromadb-compatible embedding callable."""

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [_hash_embed(t) for t in input]


class ChromaDBBackend(MemoryBackend):
    """In-process ChromaDB collection with a custom hash-based embedding."""

    name = "chromadb"

    def __init__(self) -> None:
        import chromadb  # lazy import — keeps startup fast

        # EphemeralClient (0.4.15+) or Client (older) — both create an in-memory store
        if hasattr(chromadb, "EphemeralClient"):
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.Client()
        self._ef = _HashEmbeddingFunction()
        self._collection = self._client.create_collection(
            name=f"benchmark_{uuid.uuid4().hex[:8]}",
            embedding_function=self._ef,
        )
        self._token_usage: int = 0

    # ── MemoryBackend interface ───────────────────────────────────────────────

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        # ChromaDB requires non-empty metadata dict
        meta = metadata or {}
        if not meta:
            meta = {"_added": True}
        self._collection.add(
            documents=[content],
            ids=[str(uuid.uuid4())],
            metadatas=[meta],
        )

    def query(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        count = self._collection.count()
        if count == 0:
            return []
        k = min(top_k, count)
        results = self._collection.query(query_texts=[query], n_results=k)
        docs = results["documents"][0]
        distances = results["distances"][0]
        # ChromaDB L2 distance: 0 = identical. Convert to similarity (1 = best).
        return [
            {"content": doc, "score": max(0.0, 1.0 - dist / 2.0)}
            for doc, dist in zip(docs, distances)
        ]

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name, embedding_function=self._ef
        )
        self._token_usage = 0
