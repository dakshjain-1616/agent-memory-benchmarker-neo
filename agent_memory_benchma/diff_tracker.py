"""Memory diff tracker — records content changes between benchmark sessions.

Use this to measure *memory volatility*: how much the contents of a backend
change from one task run to the next (churn rate).  The tracker is purely
additive and does not modify any backend.

Typical usage::

    tracker = MemoryDiffTracker()

    # Before a task run — snapshot what the backend currently holds
    tracker.snapshot("chromadb", [m.content for m in task.memories])

    # … run the task, then re-populate the backend …

    # After re-population — compare new contents against the snapshot
    diff = tracker.compute_diff("chromadb", [m.content for m in new_task.memories])
    print(diff["churn_rate"])
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryDiffTracker:
    """Track content changes (adds / removes / retains) across benchmark sessions.

    Each call to :meth:`compute_diff` records a diff entry and updates the
    internal snapshot so subsequent calls diff against the most-recent state.
    """

    def __init__(self) -> None:
        # backend_name → (frozenset of memory strings, snapshot timestamp)
        self._snapshots: Dict[str, Tuple[frozenset, float]] = {}
        # backend_name → list of diff records
        self._history: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def snapshot(self, backend_name: str, memories: List[str]) -> None:
        """Record the current memory contents for *backend_name*.

        Call this *before* clearing and re-populating a backend so the next
        :meth:`compute_diff` can measure what changed.
        """
        self._snapshots[backend_name] = (frozenset(memories), time.time())

    def has_snapshot(self, backend_name: str) -> bool:
        """Return *True* if a snapshot exists for *backend_name*."""
        return backend_name in self._snapshots

    # ------------------------------------------------------------------
    # Diff computation
    # ------------------------------------------------------------------

    def compute_diff(
        self, backend_name: str, new_memories: List[str]
    ) -> Dict:
        """Compare *new_memories* against the stored snapshot for *backend_name*.

        Returns a diff record::

            {
                "backend":     str,
                "added":       int,   # memories present in new but not old
                "retained":    int,   # memories present in both
                "removed":     int,   # memories present in old but not new
                "total_old":   int,
                "total_new":   int,
                "churn_rate":  float, # (added + removed) / |union|; 0=identical, 1=fully changed
                "timestamp":   float,
            }

        The internal snapshot is automatically updated to *new_memories* after
        each call, so successive calls always diff against the previous state.
        """
        old_set, _ = self._snapshots.get(backend_name, (frozenset(), 0.0))
        new_set = frozenset(new_memories)

        added = len(new_set - old_set)
        removed = len(old_set - new_set)
        retained = len(old_set & new_set)
        total_old = len(old_set)
        total_new = len(new_set)
        union = len(old_set | new_set)
        churn_rate = (added + removed) / union if union > 0 else 0.0

        diff: Dict = {
            "backend": backend_name,
            "added": added,
            "retained": retained,
            "removed": removed,
            "total_old": total_old,
            "total_new": total_new,
            "churn_rate": round(churn_rate, 4),
            "timestamp": time.time(),
        }
        self._history.setdefault(backend_name, []).append(diff)
        # Update snapshot to current state
        self._snapshots[backend_name] = (new_set, diff["timestamp"])
        return diff

    # ------------------------------------------------------------------
    # Aggregated reporting
    # ------------------------------------------------------------------

    def get_history(self, backend_name: str) -> List[Dict]:
        """Return all diff records for *backend_name* in chronological order."""
        return list(self._history.get(backend_name, []))

    def get_volatility_report(self) -> Dict[str, Dict]:
        """Return mean/max churn rate aggregated per backend.

        Structure::

            {
                backend_name: {
                    "sessions_tracked": int,
                    "mean_churn_rate":  float,
                    "max_churn_rate":   float,
                    "total_added":      int,
                    "total_removed":    int,
                }
            }
        """
        report: Dict[str, Dict] = {}
        for backend, diffs in self._history.items():
            if not diffs:
                continue
            churn_rates = [d["churn_rate"] for d in diffs]
            report[backend] = {
                "sessions_tracked": len(diffs),
                "mean_churn_rate": round(sum(churn_rates) / len(churn_rates), 4),
                "max_churn_rate": round(max(churn_rates), 4),
                "total_added": sum(d["added"] for d in diffs),
                "total_removed": sum(d["removed"] for d in diffs),
            }
        return report

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self, backend_name: Optional[str] = None) -> None:
        """Clear snapshots and history for *backend_name* (or all if *None*)."""
        if backend_name is not None:
            self._snapshots.pop(backend_name, None)
            self._history.pop(backend_name, None)
        else:
            self._snapshots.clear()
            self._history.clear()
