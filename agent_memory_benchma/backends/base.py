"""Abstract base class for all memory backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class MemoryBackend(ABC):
    """Common interface every memory backend must implement."""

    name: str = "base"

    # ── Write ────────────────────────────────────────────────────────────────

    @abstractmethod
    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Persist *content* (with optional *metadata*) to the memory store."""

    # ── Read ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def query(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return up to *top_k* memories most relevant to *query*.

        Each entry must have at least::

            {"content": str, "score": float}
        """

    # ── Maintenance ──────────────────────────────────────────────────────────

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored memories (used between benchmark tasks)."""

    # ── Telemetry ────────────────────────────────────────────────────────────

    def get_token_usage(self) -> int:
        """Return cumulative token count consumed (0 if not tracked)."""
        return getattr(self, "_token_usage", 0)

    def reset_token_usage(self) -> None:
        """Reset the token counter to 0."""
        self._token_usage = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
