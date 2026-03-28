"""Base classes for benchmark task suites."""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Memory:
    """A single piece of information to store in the memory backend."""

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Query:
    """A single retrieval query with expected answer keywords."""

    text: str
    # The query is considered answered correctly if *any* of these
    # keywords appear in the concatenated retrieved memory text.
    expected_keywords: list[str]
    # Optional exact phrase that must appear verbatim (case-insensitive).
    expected_phrase: str = ""


class TaskSuite(ABC):
    """Abstract base for all 5 benchmark task suites."""

    name: str = "base"
    description: str = ""

    @property
    @abstractmethod
    def memories(self) -> list[Memory]:
        """Ordered list of memories to inject into the backend."""

    @property
    @abstractmethod
    def queries(self) -> list[Query]:
        """Queries to issue after memories are loaded."""
