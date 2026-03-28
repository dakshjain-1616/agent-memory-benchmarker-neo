"""Memory backend adapters."""

from .base import MemoryBackend
from .chromadb_backend import ChromaDBBackend
from .faiss_backend import FAISSBackend
from .mem0_backend import Mem0Backend
from .sqlite_backend import SQLiteBackend

__all__ = [
    "MemoryBackend",
    "ChromaDBBackend",
    "FAISSBackend",
    "Mem0Backend",
    "SQLiteBackend",
]

BACKEND_REGISTRY: dict[str, type] = {
    "chromadb": ChromaDBBackend,
    "faiss": FAISSBackend,
    "mem0": Mem0Backend,
    "sqlite": SQLiteBackend,
}
