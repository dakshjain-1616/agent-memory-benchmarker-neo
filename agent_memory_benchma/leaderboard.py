"""Leaderboard — per-backend rankings and trend analysis from the results DB."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Leaderboard:
    """Compute historical rankings and accuracy trends from a ResultCollector database.

    Pass a :class:`~agent_memory_benchma.collector.ResultCollector` instance
    (already open) **or** a raw SQLite ``db_path`` string.  When a collector is
    supplied, it is used directly (no extra connection is opened).
    """

    def __init__(self, collector_or_path=None, db_path: str = "outputs/results.db"):
        # Accept either a ResultCollector instance or a plain db path string.
        if collector_or_path is None:
            self._collector = None
            self._db_path = db_path
        elif isinstance(collector_or_path, str):
            self._collector = None
            self._db_path = collector_or_path
        else:
            self._collector = collector_or_path
            self._db_path = getattr(collector_or_path, "db_path", db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_rows(self) -> List[Dict[str, Any]]:
        if self._collector is not None:
            return self._collector.get_all_results()
        # Fallback: open a fresh connection to the DB path
        try:
            import sqlite3
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM results")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            conn.close()
            return rows
        except Exception as exc:
            logger.warning("Leaderboard: could not read DB (%s)", exc)
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rankings(self) -> List[Dict[str, Any]]:
        """Return backends ranked by mean accuracy (descending).

        Each entry::

            {
                "rank": int,
                "backend": str,
                "mean_accuracy": float,
                "best_accuracy": float,
                "worst_accuracy": float,
                "mean_latency_ms": float,
                "mean_confidence": float,
                "run_count": int,
            }
        """
        rows = self._all_rows()
        if not rows:
            return []

        acc: Dict[str, List[float]] = {}
        lat: Dict[str, List[float]] = {}
        conf: Dict[str, List[float]] = {}

        for row in rows:
            b = row.get("backend", "unknown")
            acc.setdefault(b, []).append(float(row.get("accuracy", 0.0)))
            lat.setdefault(b, []).append(float(row.get("latency_ms", 0.0)))
            conf.setdefault(b, []).append(float(row.get("confidence", 0.0)))

        rankings: List[Dict[str, Any]] = []
        for backend in acc:
            a = acc[backend]
            rankings.append({
                "backend": backend,
                "mean_accuracy": round(sum(a) / len(a), 4),
                "best_accuracy": round(max(a), 4),
                "worst_accuracy": round(min(a), 4),
                "mean_latency_ms": round(
                    sum(lat.get(backend, [0])) / max(len(lat.get(backend, [1])), 1), 2
                ),
                "mean_confidence": round(
                    sum(conf.get(backend, [0])) / max(len(conf.get(backend, [1])), 1), 4
                ),
                "run_count": len(a),
            })

        rankings.sort(key=lambda x: x["mean_accuracy"], reverse=True)
        for i, entry in enumerate(rankings):
            entry["rank"] = i + 1
        return rankings

    def get_run_trend(self, backend: str) -> List[Dict[str, Any]]:
        """Return per-run accuracy trend for *backend* sorted by timestamp.

        Each entry::

            {"run_id": str, "timestamp": str, "mean_accuracy": float}
        """
        rows = self._all_rows()
        by_run: Dict[str, List[float]] = {}
        run_ts: Dict[str, str] = {}
        for row in rows:
            if row.get("backend") != backend:
                continue
            rid = row.get("run_id", "default")
            by_run.setdefault(rid, []).append(float(row.get("accuracy", 0.0)))
            # Use the earliest (first-seen) timestamp for the run
            if rid not in run_ts:
                run_ts[rid] = row.get("timestamp", "")

        trend = [
            {
                "run_id": rid,
                "timestamp": run_ts.get(rid, ""),
                "mean_accuracy": round(sum(a) / len(a), 4),
            }
            for rid, a in by_run.items()
        ]
        trend.sort(key=lambda x: x["timestamp"])
        return trend

    def get_best_backend(self) -> Optional[str]:
        """Return the name of the best-ranked backend, or *None* if no data."""
        rankings = self.get_rankings()
        return rankings[0]["backend"] if rankings else None

    def get_task_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Return mean accuracy per backend per task.

        Structure::

            {backend: {task: mean_accuracy}}
        """
        rows = self._all_rows()
        data: Dict[str, Dict[str, List[float]]] = {}
        for row in rows:
            b = row.get("backend", "unknown")
            t = row.get("task", "unknown")
            data.setdefault(b, {}).setdefault(t, []).append(float(row.get("accuracy", 0.0)))

        return {
            b: {t: round(sum(a) / len(a), 4) for t, a in task_data.items()}
            for b, task_data in data.items()
        }
