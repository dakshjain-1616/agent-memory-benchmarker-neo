"""Test suite for agent_memory_benchma package."""

import math
import pytest

from agent_memory_benchma.backends import (
    ChromaDBBackend,
    FAISSBackend,
    Mem0Backend,
    SQLiteBackend,
    BACKEND_REGISTRY,
)
from agent_memory_benchma.tasks import (
    ALL_TASKS,
    FactualRecallTask,
    TemporalOrderingTask,
    EntityTrackingTask,
    ContradictionDetectionTask,
    LongRangeDependencyTask,
    MultiSessionTask,
    PreferenceEvolutionTask,
    Memory,
    Query,
)
from agent_memory_benchma.scorer import Scorer
from agent_memory_benchma.collector import ResultCollector
from agent_memory_benchma.staleness_tracker import StalenessTracker, compute_staleness
from agent_memory_benchma.retry import with_retry, retry_call
from agent_memory_benchma.benchmark_runner import BenchmarkRunner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def chroma():
    b = ChromaDBBackend()
    yield b
    b.clear()


@pytest.fixture
def faiss():
    b = FAISSBackend()
    yield b
    b.clear()


@pytest.fixture
def mem0():
    b = Mem0Backend()
    yield b
    b.clear()


@pytest.fixture
def sqlite():
    b = SQLiteBackend()
    yield b
    b.clear()


@pytest.fixture(params=["chromadb", "faiss", "mem0", "sqlite"])
def any_backend(request):
    cls = BACKEND_REGISTRY[request.param]
    b = cls()
    yield b
    b.clear()


@pytest.fixture
def collector(tmp_path):
    c = ResultCollector(db_path=str(tmp_path / "test.db"))
    yield c
    c.close()


# ── Backend registry ──────────────────────────────────────────────────────────

class TestBackendRegistry:
    def test_all_four_backends_registered(self):
        assert set(BACKEND_REGISTRY.keys()) == {"chromadb", "faiss", "mem0", "sqlite"}

    def test_registry_values_are_classes(self):
        for name, cls in BACKEND_REGISTRY.items():
            assert isinstance(cls, type), f"{name} should be a class"


# ── Backend initialization ────────────────────────────────────────────────────

class TestBackendInit:
    def test_chromadb_initializes(self, chroma):
        assert chroma.name == "chromadb"

    def test_faiss_initializes(self, faiss):
        assert faiss.name == "faiss"

    def test_mem0_initializes(self, mem0):
        assert mem0.name == "mem0"

    def test_sqlite_initializes(self, sqlite):
        assert sqlite.name == "sqlite"


# ── Backend add / query ───────────────────────────────────────────────────────

class TestBackendAddQuery:
    def test_add_and_query_returns_results(self, any_backend):
        any_backend.add("The capital of France is Paris.")
        results = any_backend.query("What is the capital of France?", top_k=1)
        assert len(results) >= 1
        assert "content" in results[0]
        assert "score" in results[0]

    def test_query_empty_backend_returns_empty_list(self, any_backend):
        results = any_backend.query("anything", top_k=3)
        assert results == []

    def test_clear_empties_backend(self, any_backend):
        any_backend.add("Some memory.")
        any_backend.clear()
        results = any_backend.query("Some memory", top_k=3)
        assert results == []

    def test_multiple_memories_stored(self, any_backend):
        for i in range(5):
            any_backend.add(f"Fact number {i}: the value is {i * 10}.")
        results = any_backend.query("fact value", top_k=5)
        assert len(results) >= 1

    def test_score_in_zero_one_range(self, any_backend):
        any_backend.add("Python was created by Guido van Rossum.")
        results = any_backend.query("Who created Python?", top_k=1)
        if results:
            assert 0.0 <= results[0]["score"] <= 1.0


# ── Token tracking ────────────────────────────────────────────────────────────

class TestTokenTracking:
    def test_token_usage_starts_at_zero(self, any_backend):
        assert any_backend.get_token_usage() == 0

    def test_reset_token_usage(self, any_backend):
        any_backend._token_usage = 42
        any_backend.reset_token_usage()
        assert any_backend.get_token_usage() == 0

    def test_clear_resets_token_usage(self, any_backend):
        any_backend._token_usage = 99
        any_backend.clear()
        assert any_backend.get_token_usage() == 0


# ── Task suites ───────────────────────────────────────────────────────────────

class TestTaskSuites:
    def test_all_tasks_count(self):
        assert len(ALL_TASKS) == 7

    @pytest.mark.parametrize("task_cls", ALL_TASKS)
    def test_task_has_memories(self, task_cls):
        task = task_cls()
        assert len(task.memories) > 0

    @pytest.mark.parametrize("task_cls", ALL_TASKS)
    def test_task_has_queries(self, task_cls):
        task = task_cls()
        assert len(task.queries) > 0

    @pytest.mark.parametrize("task_cls", ALL_TASKS)
    def test_task_memories_are_memory_objects(self, task_cls):
        task = task_cls()
        for mem in task.memories:
            assert isinstance(mem, Memory)
            assert isinstance(mem.content, str)

    @pytest.mark.parametrize("task_cls", ALL_TASKS)
    def test_task_queries_are_query_objects(self, task_cls):
        task = task_cls()
        for q in task.queries:
            assert isinstance(q, Query)
            assert isinstance(q.text, str)

    def test_factual_recall_name(self):
        assert FactualRecallTask.name == "factual_recall"

    def test_temporal_ordering_name(self):
        assert TemporalOrderingTask.name == "temporal_ordering"

    def test_multi_session_name(self):
        assert MultiSessionTask.name == "multi_session"

    def test_preference_evolution_name(self):
        assert PreferenceEvolutionTask.name == "preference_evolution"


# ── Scorer ────────────────────────────────────────────────────────────────────

class TestScorer:
    @pytest.fixture
    def scorer(self):
        return Scorer(mock_mode=True)

    def test_exact_match_identical(self, scorer):
        score = scorer.exact_match("cerulean blue", "cerulean blue")
        assert score >= 0.9

    def test_exact_match_no_overlap(self, scorer):
        score = scorer.exact_match("xyz abc", "pqr def")
        assert score <= 0.3

    def test_semantic_similarity_mock(self, scorer):
        score = scorer.semantic_similarity("cat", "feline")
        assert 0.0 <= score <= 1.0

    def test_score_response_returns_float(self, scorer):
        result = scorer.score_response("Paris is the capital.", "Paris")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_response_perfect_match(self, scorer):
        result = scorer.score_response("cerulean blue", "cerulean blue")
        assert result >= 0.8

    def test_score_response_zero_for_empty_expected(self, scorer):
        # Empty expected should still return a score, not crash
        result = scorer.score_response("some text", "")
        assert isinstance(result, float)

    def test_weights_sum_to_one(self, scorer):
        assert abs(scorer.exact_weight + scorer.semantic_weight - 1.0) < 1e-9


# ── ResultCollector ───────────────────────────────────────────────────────────

class TestResultCollector:
    def test_record_and_retrieve(self, collector):
        collector.record_result(
            backend="sqlite", task="factual_recall",
            query="q", response="r", expected="e",
            accuracy=0.9, latency_ms=5.0, tokens=10,
        )
        rows = collector.get_all_results()
        assert len(rows) == 1
        assert rows[0]["backend"] == "sqlite"

    def test_run_metadata_round_trip(self, collector):
        run_id = collector.record_run_metadata(model="test", notes="unit test")
        collector.complete_run(run_id)
        history = collector.get_run_history()
        ids = [h["run_id"] for h in history]
        assert run_id in ids

    def test_export_csv(self, collector, tmp_path):
        collector.record_result(
            backend="faiss", task="temporal_ordering",
            query="q", response="r", expected="e",
            accuracy=0.5, latency_ms=2.0, tokens=5,
        )
        path = collector.export_csv(str(tmp_path / "out.csv"))
        import os
        assert os.path.exists(path)

    def test_multiple_records(self, collector):
        for i in range(5):
            collector.record_result(
                backend="chromadb", task="entity_tracking",
                query=f"q{i}", response=f"r{i}", expected=f"e{i}",
                accuracy=i * 0.2, latency_ms=1.0, tokens=i,
            )
        rows = collector.get_all_results()
        assert len(rows) == 5


# ── StalenessTracker ──────────────────────────────────────────────────────────

class TestStalenessTracker:
    def test_fresh_memory_low_staleness(self):
        tracker = StalenessTracker(halflife=3600)
        tracker.record_addition("chromadb")
        report = tracker.get_staleness_report("chromadb")
        assert report["avg_staleness"] < 0.01

    def test_aged_memory_high_staleness(self):
        tracker = StalenessTracker(halflife=3600)
        tracker.record_addition("faiss")
        tracker.simulate_aging("faiss", age_seconds=3600)
        report = tracker.get_staleness_report("faiss")
        assert abs(report["avg_staleness"] - 0.5) < 0.05

    def test_empty_backend_default_report(self):
        tracker = StalenessTracker()
        report = tracker.get_staleness_report("nonexistent")
        assert report["avg_staleness"] == 0.0

    def test_get_all_reports_returns_dict(self):
        tracker = StalenessTracker()
        tracker.record_addition("chromadb")
        tracker.record_addition("faiss")
        reports = tracker.get_all_reports()
        assert set(reports.keys()) == {"chromadb", "faiss"}

    def test_clear_single_backend(self):
        tracker = StalenessTracker()
        tracker.record_addition("chromadb")
        tracker.record_addition("faiss")
        tracker.clear("chromadb")
        reports = tracker.get_all_reports()
        assert "chromadb" not in reports
        assert "faiss" in reports

    def test_compute_staleness_function(self):
        s = compute_staleness(0, halflife=3600)
        assert s == pytest.approx(0.0, abs=1e-9)
        s = compute_staleness(3600, halflife=3600)
        assert s == pytest.approx(0.5, abs=0.01)


# ── Retry ─────────────────────────────────────────────────────────────────────

class TestRetry:
    def test_succeeds_on_first_attempt(self):
        calls = []

        @with_retry(attempts=3, base_delay=0)
        def fn():
            calls.append(1)
            return "ok"

        assert fn() == "ok"
        assert len(calls) == 1

    def test_retries_on_failure(self):
        calls = []

        @with_retry(attempts=3, base_delay=0)
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "ok"

        assert fn() == "ok"
        assert len(calls) == 3

    def test_raises_after_max_attempts(self):
        @with_retry(attempts=2, base_delay=0)
        def fn():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError):
            fn()

    def test_retry_call_functional(self):
        count = {"n": 0}

        def fn():
            count["n"] += 1
            if count["n"] < 2:
                raise ValueError
            return "done"

        result = retry_call(fn, attempts=3, base_delay=0)
        assert result == "done"
        assert count["n"] == 2


# ── BenchmarkRunner ───────────────────────────────────────────────────────────

class TestBenchmarkRunner:
    @pytest.fixture
    def runner(self, tmp_path):
        collector = ResultCollector(db_path=str(tmp_path / "r.db"))
        backends = [SQLiteBackend(), FAISSBackend()]
        tasks = [FactualRecallTask, TemporalOrderingTask]
        runner = BenchmarkRunner(backends=backends, tasks=tasks, collector=collector)
        yield runner
        collector.close()

    def test_run_returns_nested_dict(self, runner):
        results = runner.run()
        assert "sqlite" in results
        assert "faiss" in results
        assert "factual_recall" in results["sqlite"]
        assert "temporal_ordering" in results["faiss"]

    def test_metrics_keys_present(self, runner):
        results = runner.run()
        for backend_data in results.values():
            for metrics in backend_data.values():
                assert "accuracy" in metrics
                assert "latency_ms" in metrics
                assert "tokens" in metrics
                assert "confidence" in metrics
                assert "staleness" in metrics

    def test_accuracy_in_range(self, runner):
        results = runner.run()
        for backend_data in results.values():
            for metrics in backend_data.values():
                assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_latency_positive(self, runner):
        results = runner.run()
        for backend_data in results.values():
            for metrics in backend_data.values():
                assert metrics["latency_ms"] >= 0.0
