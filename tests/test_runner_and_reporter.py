"""Tests: BenchmarkRunner end-to-end + Reporter output files."""

import os
import pytest
import json


# ── BenchmarkRunner integration ───────────────────────────────────────────────

@pytest.fixture
def all_backends():
    from agent_memory_benchma.backends import ChromaDBBackend, FAISSBackend, Mem0Backend, SQLiteBackend
    return [ChromaDBBackend(), FAISSBackend(), Mem0Backend(), SQLiteBackend()]


@pytest.fixture
def two_backends():
    from agent_memory_benchma.backends import ChromaDBBackend, FAISSBackend
    return [ChromaDBBackend(), FAISSBackend()]


@pytest.fixture
def two_tasks():
    from agent_memory_benchma.tasks import FactualRecallTask, TemporalOrderingTask
    return [FactualRecallTask, TemporalOrderingTask]


@pytest.fixture
def collector(tmp_path):
    from agent_memory_benchma.collector import ResultCollector
    db = str(tmp_path / "run.db")
    c = ResultCollector(db_path=db)
    yield c
    c.close()


def test_runner_returns_nested_dict(two_backends, two_tasks, collector):
    from agent_memory_benchma.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        backends=two_backends,
        tasks=two_tasks,
        collector=collector,
        mock_mode=True,
    )
    results = runner.run()
    assert isinstance(results, dict)
    assert "chromadb" in results
    assert "faiss" in results


def test_runner_task_keys_present(two_backends, two_tasks, collector):
    from agent_memory_benchma.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        backends=two_backends,
        tasks=two_tasks,
        collector=collector,
        mock_mode=True,
    )
    results = runner.run()
    for backend_name, task_data in results.items():
        assert "factual_recall" in task_data
        assert "temporal_ordering" in task_data


def test_runner_accuracy_in_range(two_backends, two_tasks, collector):
    from agent_memory_benchma.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        backends=two_backends,
        tasks=two_tasks,
        collector=collector,
        mock_mode=True,
    )
    results = runner.run()
    for backend_name, task_data in results.items():
        for task_name, m in task_data.items():
            assert 0.0 <= m["accuracy"] <= 1.0, f"{backend_name}/{task_name} accuracy out of range"


def test_runner_latency_measured(two_backends, two_tasks, collector):
    from agent_memory_benchma.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        backends=two_backends,
        tasks=two_tasks,
        collector=collector,
        mock_mode=True,
    )
    results = runner.run()
    for backend_name, task_data in results.items():
        for task_name, m in task_data.items():
            assert m["latency_ms"] >= 0.0, f"Negative latency for {backend_name}/{task_name}"


def test_runner_token_usage_tracked(two_backends, two_tasks, collector):
    from agent_memory_benchma.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        backends=two_backends,
        tasks=two_tasks,
        collector=collector,
        mock_mode=True,
    )
    results = runner.run()
    for task_data in results.values():
        for m in task_data.values():
            assert isinstance(m["tokens"], int)
            assert m["tokens"] >= 0


def test_runner_all_backends_run(all_backends, two_tasks, collector):
    """Verify all 4 backends complete without error."""
    from agent_memory_benchma.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        backends=all_backends,
        tasks=two_tasks,
        collector=collector,
        mock_mode=True,
    )
    results = runner.run()
    assert len(results) == 4


# ── Reporter ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_results():
    """Minimal plausible results dict for reporter tests."""
    tasks = ["factual_recall", "temporal_ordering", "entity_tracking"]
    backends = ["chromadb", "faiss", "mem0", "sqlite"]
    return {
        b: {
            t: {"accuracy": 0.75, "latency_ms": 12.0, "tokens": 0, "confidence": 0.8, "staleness": 0.1}
            for t in tasks
        }
        for b in backends
    }


@pytest.fixture
def reporter(tmp_path):
    from agent_memory_benchma.reporter import Reporter
    return Reporter(output_dir=str(tmp_path))


def test_radar_chart_saved(reporter, sample_results, tmp_path):
    path = reporter.generate_radar_chart(sample_results, filename="radar.png")
    assert path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_latency_chart_saved(reporter, sample_results, tmp_path):
    path = reporter.generate_latency_boxplot(sample_results, filename="latency.png")
    assert path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_comparison_csv_saved(reporter, sample_results, tmp_path):
    path = reporter.generate_comparison_csv(sample_results, filename="comparison.csv")
    assert path
    assert os.path.exists(path)
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 4  # at least 4 backend rows


def test_pdf_report_saved(reporter, sample_results, tmp_path):
    path = reporter.generate_pdf_report(sample_results, filename="report.pdf")
    assert path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 100  # non-empty PDF
    # Check PDF magic bytes
    with open(path, "rb") as f:
        assert f.read(4) == b"%PDF"
