"""SQLite-backed summary memory store using FTS5 full-text search."""

from __future__ import annotations
import sqlite3
import tempfile
import os
from typing import Any

from .base import MemoryBackend

class SQLiteBackend(MemoryBackend):
    """Persistent full-text search using SQLite FTS5.

    Each memory is stored as a row; queries use SQLite's built-in BM25 ranking.
    Falls back to simple LIKE matching if FTS5 is unavailable.
    """

    name = "sqlite"

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            db_path = self._tmpfile.name
        else:
            self._tmpfile = None
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._token_usage: int = 0
        self._use_fts = self._setup_tables()

    def _setup_tables(self) -> bool:
        cur = self._conn.cursor()
        # Try FTS5 first
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                "USING fts5(content, metadata)"
            )
            # Also create a regular table for storage
            cur.execute(
                "CREATE TABLE IF NOT EXISTS memories "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, metadata TEXT)"
            )
            self._conn.commit()
            return True
        except sqlite3.OperationalError:
            # FTS5 not compiled — fall back to regular table
            cur.execute(
                "CREATE TABLE IF NOT EXISTS memories "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, metadata TEXT)"
            )
            self._conn.commit()
            return False

    # ── MemoryBackend interface ───────────────────────────────────────────────

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        import json

        meta_str = json.dumps(metadata or {})
        cur = self._conn.cursor()
        cur.execute("INSERT INTO memories(content, metadata) VALUES (?, ?)", (content, meta_str))
        if self._use_fts:
            cur.execute("INSERT INTO memories_fts(content, metadata) VALUES (?, ?)", (content, meta_str))
        
        self._conn.commit()

    def query(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        results: list[dict[str, Any]] = []

        if self._use_fts:
            # Escape FTS special characters
            safe_q = query.replace('"', '""').replace("*", "").replace("'", "''")
            try:
                cur.execute(
                    "SELECT content, bm25(memories_fts) as rank FROM memories_fts WHERE content MATCH ? ORDER BY rank LIMIT ?",
                    (safe_q, top_k),
                )
                for row in cur.fetchall():
                    content = row[0]
                    # FTS5 rank is negative BM25 — convert to 0-1 similarity
                    rank = float(row[1]) if row[1] is not None else -1.0
                    score = max(0.0, min(1.0, 1.0 / (1.0 + abs(rank))))
                    results.append({"content": content, "score": score})
                if results:
                    return results
            except sqlite3.OperationalError:
                pass  # fall through to LIKE

        # LIKE fallback
        words = query.lower().split()[:5]
        if words:
            conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in words])
            params = [f"%{w}%" for w in words]
            cur.execute(
                f"SELECT content FROM memories WHERE {conditions} LIMIT ?",
                params + [top_k],
            )
            for row in cur.fetchall():
                results.append({"content": row[0], "score": 0.5})

        # If still empty, return top rows
        if not results:
            cur.execute("SELECT content FROM memories LIMIT ?", (top_k,))
            for row in cur.fetchall():
                results.append({"content": row[0], "score": 0.1})

        return results

    def clear(self) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM memories")
        if self._use_fts:
            cur.execute("DELETE FROM memories_fts")
        self._conn.commit()
        self._token_usage = 0

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        if self._tmpfile is not None:
            try:
                os.unlink(self._db_path)
            except Exception:
                pass
