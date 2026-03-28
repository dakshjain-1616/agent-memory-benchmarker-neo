"""Benchmark preset profiles — named configurations for common run scenarios."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class BenchmarkProfile:
    """A named preset specifying which backends/tasks/settings to use.

    ``backends`` and ``tasks`` may be *None* to mean "all registered".
    """

    name: str
    description: str
    backends: Optional[List[str]] = None   # None → all registered backends
    tasks: Optional[List[str]] = None       # None → all registered tasks
    top_k: int = 3
    mock_mode: bool = False

    def resolve_backends(self, registry: dict) -> list:
        """Instantiate backend objects for the backends listed in this profile."""
        names = self.backends if self.backends is not None else list(registry.keys())
        result = []
        for n in names:
            cls = registry.get(n)
            if cls is None:
                continue
            try:
                result.append(cls())
            except Exception:
                pass
        return result

    def resolve_tasks(self, all_tasks: list) -> list:
        """Filter *all_tasks* (list of TaskSuite classes) to those in this profile."""
        if self.tasks is None:
            return list(all_tasks)
        name_set = set(self.tasks)
        return [t for t in all_tasks if t.name in name_set]


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, BenchmarkProfile] = {
    "quick": BenchmarkProfile(
        name="quick",
        description="Two fast backends × two tasks — ideal for smoke-testing",
        backends=["faiss", "sqlite"],
        tasks=["factual_recall", "temporal_ordering"],
        top_k=3,
    ),
    "vector": BenchmarkProfile(
        name="vector",
        description="Vector-only backends (ChromaDB + FAISS) × all tasks",
        backends=["chromadb", "faiss"],
        tasks=None,
        top_k=5,
    ),
    "standard": BenchmarkProfile(
        name="standard",
        description="All backends × four core tasks — balanced benchmark",
        backends=None,
        tasks=[
            "factual_recall",
            "temporal_ordering",
            "entity_tracking",
            "contradiction_detection",
        ],
        top_k=3,
    ),
    "full": BenchmarkProfile(
        name="full",
        description="All backends × all 7 tasks — comprehensive benchmark",
        backends=None,
        tasks=None,
        top_k=5,
    ),
}

DEFAULT_PROFILE_NAME: str = os.getenv("BENCHMARK_PROFILE", "standard")


def get_profile(name: str) -> BenchmarkProfile:
    """Return the named profile, falling back to *standard* if not found."""
    return PROFILES.get(name, PROFILES["standard"])


def list_profiles() -> List[str]:
    """Return sorted list of available profile names."""
    return sorted(PROFILES.keys())
