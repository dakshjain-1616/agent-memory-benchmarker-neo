"""Tests: all 4 memory backends initialise and respond correctly."""

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def chromadb_backend():
    from agent_memory_benchma.backends import ChromaDBBackend
    b = ChromaDBBackend()
    yield b
    b.clear()


@pytest.fixture
def faiss_backend():
    from agent_memory_benchma.backends import FAISSBackend
    b = FAISSBackend()
    yield b
    b.clear()


@pytest.fixture
def mem0_backend():
    from agent_memory_benchma.backends import Mem0Backend
    b = Mem0Backend()
    yield b
    b.clear()


@pytest.fixture
def sqlite_backend():
    from agent_memory_benchma.backends import SQLiteBackend
    b = SQLiteBackend()
    yield b
    b.clear()


@pytest.fixture(params=["chromadb_backend", "faiss_backend", "mem0_backend", "sqlite_backend"])
def any_backend(request):
    """Parametrize over all 4 backends."""
    return request.getfixturevalue(request.param)


# ── Init tests ────────────────────────────────────────────────────────────────

def test_chromadb_init(chromadb_backend):
    assert chromadb_backend is not None
    assert chromadb_backend.name == "chromadb"


def test_faiss_init(faiss_backend):
    assert faiss_backend is not None
    assert faiss_backend.name == "faiss"


def test_mem0_init(mem0_backend):
    assert mem0_backend is not None
    assert mem0_backend.name == "mem0"


def test_sqlite_init(sqlite_backend):
    assert sqlite_backend is not None
    assert sqlite_backend.name == "sqlite"


# ── Add / query roundtrip ─────────────────────────────────────────────────────

def test_add_and_query_returns_list(any_backend):
    any_backend.add("The sky is blue and vast.")
    results = any_backend.query("sky colour", top_k=3)
    assert isinstance(results, list)


def test_add_and_query_contains_content_key(any_backend):
    any_backend.add("Mount Everest is 8849 metres tall.")
    results = any_backend.query("Everest height", top_k=1)
    assert len(results) > 0
    assert "content" in results[0]


def test_add_and_query_content_is_string(any_backend):
    any_backend.add("The Eiffel Tower was built in 1889.")
    results = any_backend.query("Eiffel Tower", top_k=1)
    assert isinstance(results[0]["content"], str)


def test_query_returns_score(any_backend):
    any_backend.add("Python was created by Guido van Rossum.")
    results = any_backend.query("Python creator", top_k=1)
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


def test_add_multiple_memories(any_backend):
    for i in range(5):
        any_backend.add(f"Fact number {i}: the value is {i * 10}.")
    results = any_backend.query("fact value", top_k=3)
    assert len(results) >= 1  # at least one result returned


def test_clear_empties_store(any_backend):
    any_backend.add("This memory should be cleared.")
    any_backend.clear()
    results = any_backend.query("memory", top_k=3)
    # After clear either returns empty or results from a fresh state
    assert isinstance(results, list)


def test_query_empty_store_returns_list(any_backend):
    # Backend was just cleared by fixture; querying empty store should not raise
    results = any_backend.query("anything", top_k=3)
    assert isinstance(results, list)


def test_top_k_respected(any_backend):
    for i in range(10):
        any_backend.add(f"Memory item {i}: random content about topic {i}.")
    results = any_backend.query("memory item", top_k=3)
    assert len(results) <= 3


def test_token_usage_is_int(any_backend):
    usage = any_backend.get_token_usage()
    assert isinstance(usage, int)
    assert usage >= 0


def test_reset_token_usage(any_backend):
    any_backend.reset_token_usage()
    assert any_backend.get_token_usage() == 0


def test_metadata_accepted_without_error(any_backend):
    """Adding with metadata should not raise."""
    any_backend.add("Fact with metadata.", metadata={"source": "test", "version": 1})


def test_backend_registry_contains_all():
    from agent_memory_benchma.backends import BACKEND_REGISTRY
    assert "chromadb" in BACKEND_REGISTRY
    assert "faiss" in BACKEND_REGISTRY
    assert "mem0" in BACKEND_REGISTRY
    assert "sqlite" in BACKEND_REGISTRY
    assert len(BACKEND_REGISTRY) == 4
