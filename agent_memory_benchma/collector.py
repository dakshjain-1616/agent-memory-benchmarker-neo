"""SQLite-backed result persistence."""

import os
import uuid
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ResultCollector:
    """Persist benchmark results to a SQLite database."""

    def __init__(self, db_path: str = "outputs/results.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        self._current_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    task TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    expected TEXT,
                    accuracy REAL,
                    latency_ms REAL,
                    tokens INTEGER,
                    confidence REAL,
                    staleness REAL,
                    timestamp TEXT
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT,
                    completed_at TEXT,
                    model TEXT,
                    notes TEXT
                )
            """)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def record_run_metadata(self, model: str = "", notes: str = "") -> str:
        run_id = str(uuid.uuid4())[:8]
        self._current_run_id = run_id
        with self._conn:
            self._conn.execute(
                "INSERT INTO run_metadata (run_id, started_at, model, notes) VALUES (?, ?, ?, ?)",
                (run_id, datetime.utcnow().isoformat(), model, notes),
            )
        return run_id

    def complete_run(self, run_id: Optional[str] = None) -> None:
        rid = run_id or self._current_run_id
        if rid:
            with self._conn:
                self._conn.execute(
                    "UPDATE run_metadata SET completed_at = ? WHERE run_id = ?",
                    (datetime.utcnow().isoformat(), rid),
                )

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_result(
        self,
        backend: str,
        task: str,
        query: str,
        response: str,
        expected: str,
        accuracy: float,
        latency_ms: float,
        tokens: int,
        confidence: float = 0.0,
        staleness: float = 0.0,
        run_id: Optional[str] = None,
    ) -> None:
        rid = run_id or self._current_run_id or "default"
        with self._conn:
            self._conn.execute(
                """INSERT INTO results
                   (run_id, backend, task, query, response, expected,
                    accuracy, latency_ms, tokens, confidence, staleness, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rid, backend, task, query, response, expected,
                 accuracy, latency_ms, tokens, confidence, staleness,
                 datetime.utcnow().isoformat()),
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_all_results(self, run_id: Optional[str] = None) -> List[Dict]:
        cur = self._conn.cursor()
        if run_id:
            cur.execute("SELECT * FROM results WHERE run_id = ?", (run_id,))
        else:
            cur.execute("SELECT * FROM results")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_summary_df(self):
        """Return a pandas DataFrame with per-backend/task aggregate stats."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed; cannot return DataFrame")
            return None
        rows = self.get_all_results()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return (
            df.groupby(["backend", "task"])
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                latency_mean=("latency_ms", "mean"),
                tokens_mean=("tokens", "mean"),
                confidence_mean=("confidence", "mean"),
                staleness_mean=("staleness", "mean"),
                count=("accuracy", "count"),
            )
            .reset_index()
        )

    def get_run_history(self) -> List[Dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM run_metadata ORDER BY started_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def export_csv(self, path: str = "outputs/comparison.csv") -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = self.get_summary_df()
        if df is not None and not df.empty:
            df.to_csv(path, index=False)
        return path

    def close(self) -> None:
        self._conn.close()
