"""Tree Ring Memory CLI backend."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import MemoryBackend

_STOPWORDS = {
    "about",
    "after",
    "all",
    "and",
    "are",
    "been",
    "before",
    "being",
    "can",
    "current",
    "currently",
    "did",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "list",
    "made",
    "now",
    "our",
    "the",
    "this",
    "used",
    "uses",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _resolve_binary() -> str | None:
    configured = os.getenv("TREE_RING_BIN")
    if configured:
        return configured if Path(configured).exists() else None
    return shutil.which("tree-ring")


class TreeRingBackend(MemoryBackend):
    """Local-first Tree Ring Memory backend using the ``tree-ring`` CLI.

    Set ``TREE_RING_BIN`` when the CLI is not on ``PATH``. By default the
    backend creates an isolated temporary Tree Ring root for benchmark data.
    ``TREE_RING_ROOT`` can point at an explicit root; in that mode ``clear()``
    only deletes memories created by this backend instance.
    """

    name = "tree_ring"

    @classmethod
    def available(cls) -> bool:
        """Return True when the tree-ring CLI can be launched."""
        return _resolve_binary() is not None

    def __init__(self, root: str | None = None) -> None:
        binary = _resolve_binary()
        if not binary:
            raise RuntimeError(
                "Tree Ring backend requires the tree-ring CLI. "
                "Install it or set TREE_RING_BIN=/path/to/tree-ring."
            )

        configured_root = root or os.getenv("TREE_RING_ROOT")
        if configured_root:
            self._root = Path(configured_root)
            self._owns_root = False
        else:
            self._root = Path(tempfile.mkdtemp(prefix="amb-tree-ring-"))
            self._owns_root = True

        self._bin = binary
        self._project = os.getenv("TREE_RING_PROJECT", "agent-memory-benchmarker")
        self._stored_ids: list[str] = []
        self._token_usage = 0
        self._init_store()

    # ── MemoryBackend interface ───────────────────────────────────────────────

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        tags = ["benchmark"]
        for key, value in (metadata or {}).items():
            if isinstance(value, (str, int, float, bool)):
                tags.append(f"{key}:{value}")

        args = [
            "remember",
            "--event-type",
            "lesson",
            "--scope",
            "project",
            "--project",
            self._project,
            *[part for tag in tags for part in ("--tag", tag)],
            content,
        ]
        memory = self._run_json(args)
        memory_id = memory.get("id")
        if isinstance(memory_id, str):
            self._stored_ids.append(memory_id)

    def query(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        for recall_query in [query, *self._fallback_queries(query)]:
            for item in self._recall(recall_query, top_k):
                memory = item.get("memory", {})
                memory_id = memory.get("id", memory.get("summary", ""))
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                results.append(
                    {
                        "content": memory.get("summary", ""),
                        "score": float(item.get("score", 0.0)),
                    }
                )
                if len(results) >= top_k:
                    return results

        return results

    def clear(self) -> None:
        if self._owns_root:
            shutil.rmtree(self._root, ignore_errors=True)
            self._root.mkdir(parents=True, exist_ok=True)
            self._stored_ids.clear()
            self._init_store()
        else:
            for memory_id in list(self._stored_ids):
                self._forget(memory_id)
            self._stored_ids.clear()
        self._token_usage = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_store(self) -> None:
        self._run(["init"])

    def _recall(self, query: str, top_k: int) -> list[dict[str, Any]]:
        result = self._run_json(
            [
                "recall",
                "--project",
                self._project,
                "--limit",
                str(top_k),
                query,
            ]
        )
        return result if isinstance(result, list) else []

    def _forget(self, memory_id: str) -> None:
        try:
            self._run_json(
                [
                    "forget",
                    "--mode",
                    "delete",
                    "--reason",
                    "benchmark cleanup",
                    memory_id,
                ]
            )
        except RuntimeError:
            pass

    def _fallback_queries(self, query: str) -> list[str]:
        terms = [
            term
            for term in re.findall(r"[a-z0-9][a-z0-9./-]*", query.lower())
            if len(term) >= 3 and term not in _STOPWORDS
        ]
        expanded: list[str] = []
        for term in terms:
            expanded.append(term)
            if term.startswith("authent"):
                expanded.append("auth")
            elif term.endswith("ing") and len(term) > 5:
                expanded.append(term[:-3])
            elif term.endswith("s") and len(term) > 4:
                expanded.append(term[:-1])
        return list(dict.fromkeys(expanded))

    def _run_json(self, args: list[str]) -> Any:
        return json.loads(self._run(args, emit_json=True))

    def _run(self, args: list[str], emit_json: bool = False) -> str:
        cli_args = ["--root", str(self._root), *(["--json"] if emit_json else []), *args]
        try:
            completed = subprocess.run(
                [self._bin, *cli_args],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            raise RuntimeError(f"tree-ring {args[0]} failed: {stderr}") from exc
        return completed.stdout.strip()

    def __del__(self) -> None:
        if getattr(self, "_owns_root", False):
            shutil.rmtree(getattr(self, "_root", ""), ignore_errors=True)
