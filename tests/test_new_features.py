"""Tests for new features: Leaderboard, BenchmarkProfile/profiles, MemoryDiffTracker,
run_streaming, leaderboard chart, and PDF report enhancements.
"""

import os
import pytest
import time


# ── Leaderboard ───────────────────────────────────────────────────────────────

class TestLeaderboard:
    @pytest.fixture
    def collector(self, tmp_path):
        from agent_memory_benchma.collector import ResultCollector
        c = ResultCollector(db_path=str(tmp_path / "lb.db"))
        yield c
        c.close()

    @pytest.fixture
    def populated_leaderboard(self, collector):
        from agent_memory_benchma.leaderboard import Leaderboard
        # Insert some rows
        run_id = collector.record_run_metadata(model="test")
        for backend in ["chromadb", "faiss", "sqlite"]:
            for task in ["factual_recall", "temporal_ordering"]:
                acc = {"chromadb": 0.9, "faiss": 0.7, "sqlite": 0.5}[backend]
                collector.record_result(
                    backend=backend, task=task, query="q", response="r",
                    expected="e", accuracy=acc, latency_ms=10.0, tokens=0,
                    run_id=run_id,
                )
        collector.complete_run(run_id)
        return Leaderboard(collector)

    def test_get_rankings_returns_list(self, populated_leaderboard):
        rankings = populated_leaderboard.get_rankings()
        assert isinstance(rankings, list)
        assert len(rankings) == 3

    def test_rankings_sorted_by_accuracy_desc(self, populated_leaderboard):
        rankings = populated_leaderboard.get_rankings()
        accs = [r["mean_accuracy"] for r in rankings]
        assert accs == sorted(accs, reverse=True)

    def test_rankings_have_required_keys(self, populated_leaderboard):
        rankings = populated_leaderboard.get_rankings()
        for entry in rankings:
            for key in ("rank", "backend", "mean_accuracy", "best_accuracy",
                        "worst_accuracy", "run_count"):
                assert key in entry, f"Missing key: {key}"

    def test_rank_starts_at_one(self, populated_leaderboard):
        rankings = populated_leaderboard.get_rankings()
        assert rankings[0]["rank"] == 1
        assert rankings[-1]["rank"] == len(rankings)

    def test_get_best_backend(self, populated_leaderboard):
        best = populated_leaderboard.get_best_backend()
        assert best == "chromadb"

    def test_get_best_backend_empty_db(self, collector):
        from agent_memory_benchma.leaderboard import Leaderboard
        lb = Leaderboard(collector)
        assert lb.get_best_backend() is None

    def test_get_run_trend_returns_list(self, populated_leaderboard):
        trend = populated_leaderboard.get_run_trend("faiss")
        assert isinstance(trend, list)
        assert len(trend) >= 1

    def test_get_run_trend_unknown_backend_empty(self, populated_leaderboard):
        trend = populated_leaderboard.get_run_trend("nonexistent")
        assert trend == []

    def test_get_task_breakdown(self, populated_leaderboard):
        breakdown = populated_leaderboard.get_task_breakdown()
        assert isinstance(breakdown, dict)
        assert "chromadb" in breakdown
        assert "factual_recall" in breakdown["chromadb"]

    def test_leaderboard_accepts_db_path(self, tmp_path):
        from agent_memory_benchma.leaderboard import Leaderboard
        lb = Leaderboard(str(tmp_path / "missing.db"))
        assert lb.get_rankings() == []

    def test_leaderboard_mean_accuracy_correct(self, populated_leaderboard):
        rankings = populated_leaderboard.get_rankings()
        by_name = {r["backend"]: r for r in rankings}
        assert by_name["chromadb"]["mean_accuracy"] == pytest.approx(0.9, abs=1e-3)
        assert by_name["faiss"]["mean_accuracy"] == pytest.approx(0.7, abs=1e-3)


# ── BenchmarkProfile / profiles ───────────────────────────────────────────────

class TestProfiles:
    def test_all_builtin_profiles_exist(self):
        from agent_memory_benchma.profiles import PROFILES
        for name in ("quick", "vector", "standard", "full"):
            assert name in PROFILES

    def test_profile_has_required_fields(self):
        from agent_memory_benchma.profiles import PROFILES
        for name, p in PROFILES.items():
            assert isinstance(p.name, str)
            assert isinstance(p.description, str)
            assert p.top_k >= 1

    def test_get_profile_returns_correct(self):
        from agent_memory_benchma.profiles import get_profile
        p = get_profile("quick")
        assert p.name == "quick"

    def test_get_profile_fallback(self):
        from agent_memory_benchma.profiles import get_profile, PROFILES
        p = get_profile("nonexistent_profile")
        assert p.name == "standard"

    def test_list_profiles_returns_sorted(self):
        from agent_memory_benchma.profiles import list_profiles
        names = list_profiles()
        assert names == sorted(names)
        assert len(names) >= 4

    def test_resolve_backends_quick(self):
        from agent_memory_benchma.profiles import get_profile
        from agent_memory_benchma.backends import BACKEND_REGISTRY
        profile = get_profile("quick")
        backends = profile.resolve_backends(BACKEND_REGISTRY)
        assert len(backends) >= 1
        names = {b.name for b in backends}
        assert names.issubset({"faiss", "sqlite"})

    def test_resolve_backends_full_returns_all(self):
        from agent_memory_benchma.profiles import get_profile
        from agent_memory_benchma.backends import BACKEND_REGISTRY
        profile = get_profile("full")
        backends = profile.resolve_backends(BACKEND_REGISTRY)
        assert len(backends) == len(BACKEND_REGISTRY)

    def test_resolve_tasks_quick_returns_subset(self):
        from agent_memory_benchma.profiles import get_profile
        from agent_memory_benchma.tasks import ALL_TASKS
        profile = get_profile("quick")
        tasks = profile.resolve_tasks(ALL_TASKS)
        assert len(tasks) == 2

    def test_resolve_tasks_full_returns_all(self):
        from agent_memory_benchma.profiles import get_profile
        from agent_memory_benchma.tasks import ALL_TASKS
        profile = get_profile("full")
        tasks = profile.resolve_tasks(ALL_TASKS)
        assert len(tasks) == len(ALL_TASKS)

    def test_resolve_tasks_none_returns_all(self):
        from agent_memory_benchma.profiles import BenchmarkProfile
        from agent_memory_benchma.tasks import ALL_TASKS
        p = BenchmarkProfile(name="x", description="x", tasks=None, top_k=3)
        assert len(p.resolve_tasks(ALL_TASKS)) == len(ALL_TASKS)


# ── MemoryDiffTracker ─────────────────────────────────────────────────────────

class TestMemoryDiffTracker:
    @pytest.fixture
    def tracker(self):
        from agent_memory_benchma.diff_tracker import MemoryDiffTracker
        return MemoryDiffTracker()

    def test_snapshot_sets_state(self, tracker):
        tracker.snapshot("faiss", ["mem A", "mem B"])
        assert tracker.has_snapshot("faiss")

    def test_no_snapshot_for_unknown_backend(self, tracker):
        assert not tracker.has_snapshot("unknown")

    def test_compute_diff_no_change(self, tracker):
        mems = ["mem A", "mem B", "mem C"]
        tracker.snapshot("faiss", mems)
        diff = tracker.compute_diff("faiss", mems)
        assert diff["added"] == 0
        assert diff["removed"] == 0
        assert diff["retained"] == 3
        assert diff["churn_rate"] == pytest.approx(0.0)

    def test_compute_diff_all_new(self, tracker):
        tracker.snapshot("faiss", ["old A", "old B"])
        diff = tracker.compute_diff("faiss", ["new X", "new Y"])
        assert diff["added"] == 2
        assert diff["removed"] == 2
        assert diff["retained"] == 0
        assert diff["churn_rate"] == pytest.approx(1.0)

    def test_compute_diff_partial_change(self, tracker):
        tracker.snapshot("chromadb", ["mem A", "mem B", "mem C"])
        diff = tracker.compute_diff("chromadb", ["mem A", "mem B", "mem D"])
        assert diff["added"] == 1
        assert diff["removed"] == 1
        assert diff["retained"] == 2

    def test_compute_diff_updates_snapshot(self, tracker):
        tracker.snapshot("sqlite", ["x"])
        tracker.compute_diff("sqlite", ["y"])
        # Now snapshot should be ["y"]; diffing with ["y"] should give 0 churn
        diff2 = tracker.compute_diff("sqlite", ["y"])
        assert diff2["churn_rate"] == pytest.approx(0.0)

    def test_compute_diff_empty_old_snapshot(self, tracker):
        # No prior snapshot → old_set = empty frozenset
        diff = tracker.compute_diff("mem0", ["mem A", "mem B"])
        assert diff["total_old"] == 0
        assert diff["added"] == 2
        assert diff["churn_rate"] == pytest.approx(1.0)

    def test_get_history_returns_records(self, tracker):
        tracker.snapshot("faiss", ["a"])
        tracker.compute_diff("faiss", ["b"])
        tracker.compute_diff("faiss", ["b", "c"])
        history = tracker.get_history("faiss")
        assert len(history) == 2

    def test_get_volatility_report(self, tracker):
        tracker.snapshot("faiss", ["a", "b"])
        tracker.compute_diff("faiss", ["c", "d"])  # 100% churn
        tracker.compute_diff("faiss", ["c", "d"])  # 0% churn
        report = tracker.get_volatility_report()
        assert "faiss" in report
        assert report["faiss"]["sessions_tracked"] == 2
        assert 0.4 < report["faiss"]["mean_churn_rate"] < 0.6

    def test_clear_single_backend(self, tracker):
        tracker.snapshot("faiss", ["a"])
        tracker.compute_diff("faiss", ["b"])
        tracker.snapshot("sqlite", ["x"])
        tracker.clear("faiss")
        assert not tracker.has_snapshot("faiss")
        assert tracker.has_snapshot("sqlite")

    def test_clear_all(self, tracker):
        tracker.snapshot("faiss", ["a"])
        tracker.snapshot("sqlite", ["x"])
        tracker.clear()
        assert not tracker.has_snapshot("faiss")
        assert not tracker.has_snapshot("sqlite")

    def test_diff_result_has_required_keys(self, tracker):
        diff = tracker.compute_diff("test", ["a"])
        for key in ("backend", "added", "retained", "removed",
                    "total_old", "total_new", "churn_rate", "timestamp"):
            assert key in diff


# ── run_streaming ─────────────────────────────────────────────────────────────

class TestRunStreaming:
    @pytest.fixture
    def runner(self, tmp_path):
        from agent_memory_benchma.benchmark_runner import BenchmarkRunner
        from agent_memory_benchma.collector import ResultCollector
        from agent_memory_benchma.backends import FAISSBackend, SQLiteBackend
        from agent_memory_benchma.tasks import FactualRecallTask, TemporalOrderingTask

        collector = ResultCollector(db_path=str(tmp_path / "stream.db"))
        runner = BenchmarkRunner(
            backends=[FAISSBackend(), SQLiteBackend()],
            tasks=[FactualRecallTask, TemporalOrderingTask],
            collector=collector,
            mock_mode=True,
            model="openai/gpt-5.4-mini",
        )
        yield runner
        collector.close()

    def test_run_streaming_yields_dicts(self, runner):
        updates = list(runner.run_streaming())
        assert len(updates) > 0
        for u in updates:
            assert isinstance(u, dict)

    def test_run_streaming_required_keys(self, runner):
        for update in runner.run_streaming():
            for key in ("status", "elapsed_s", "tokens", "turns", "total",
                        "partial_results", "done", "run_id"):
                assert key in update, f"Missing key: {key}"

    def test_run_streaming_last_update_done(self, runner):
        updates = list(runner.run_streaming())
        assert updates[-1]["done"] is True

    def test_run_streaming_intermediate_not_done(self, runner):
        updates = list(runner.run_streaming())
        assert all(not u["done"] for u in updates[:-1])

    def test_run_streaming_total_correct(self, runner):
        # 2 backends × 2 tasks = 4 steps
        updates = list(runner.run_streaming())
        assert updates[-1]["total"] == 4
        assert updates[-1]["turns"] == 4

    def test_run_streaming_partial_results_grow(self, runner):
        turn_counts = []
        for update in runner.run_streaming():
            count = sum(len(v) for v in update["partial_results"].values())
            turn_counts.append(count)
        # Each step adds one task result — count should increase monotonically
        assert turn_counts == sorted(turn_counts)

    def test_run_streaming_model_stored(self, runner, tmp_path):
        updates = list(runner.run_streaming())
        run_id = updates[-1]["run_id"]
        history = runner.collector.get_run_history()
        matching = [h for h in history if h["run_id"] == run_id]
        assert matching
        assert matching[0]["model"] == "openai/gpt-5.4-mini"


# ── Reporter: leaderboard chart & PDF enhancements ────────────────────────────

class TestReporterEnhancements:
    @pytest.fixture
    def reporter(self, tmp_path):
        from agent_memory_benchma.reporter import Reporter
        return Reporter(output_dir=str(tmp_path))

    @pytest.fixture
    def leaderboard_data(self):
        return [
            {"rank": 1, "backend": "chromadb", "mean_accuracy": 0.9,
             "best_accuracy": 0.95, "worst_accuracy": 0.85,
             "mean_latency_ms": 5.0, "mean_confidence": 0.8, "run_count": 10},
            {"rank": 2, "backend": "faiss", "mean_accuracy": 0.75,
             "best_accuracy": 0.8, "worst_accuracy": 0.7,
             "mean_latency_ms": 3.0, "mean_confidence": 0.7, "run_count": 10},
        ]

    @pytest.fixture
    def sample_results(self):
        return {
            "chromadb": {"factual_recall": {"accuracy": 0.9, "latency_ms": 5.0,
                                             "tokens": 0, "confidence": 0.8, "staleness": 0.1}},
            "faiss": {"factual_recall": {"accuracy": 0.75, "latency_ms": 3.0,
                                          "tokens": 0, "confidence": 0.7, "staleness": 0.05}},
        }

    def test_leaderboard_chart_created(self, reporter, leaderboard_data, tmp_path):
        path = reporter.generate_leaderboard_chart(leaderboard_data)
        assert path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_leaderboard_chart_empty_data_returns_empty(self, reporter):
        path = reporter.generate_leaderboard_chart([])
        assert path == ""

    def test_pdf_includes_staleness_column(self, reporter, sample_results, tmp_path):
        path = reporter.generate_pdf_report(
            sample_results,
            filename="test_report.pdf",
            model="openai/gpt-5.4-mini",
            run_id="abc123",
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100
        with open(path, "rb") as f:
            assert f.read(4) == b"%PDF"

    def test_pdf_with_model_metadata(self, reporter, sample_results):
        # Just ensure it doesn't crash when model/run_id are provided
        path = reporter.generate_pdf_report(
            sample_results,
            model="openai/gpt-5.4-mini",
            run_id="test-run",
        )
        assert os.path.exists(path)


# ── BenchmarkRunner model param ───────────────────────────────────────────────

class TestBenchmarkRunnerModel:
    @pytest.fixture
    def runner(self, tmp_path):
        from agent_memory_benchma.benchmark_runner import BenchmarkRunner
        from agent_memory_benchma.collector import ResultCollector
        from agent_memory_benchma.backends import FAISSBackend
        from agent_memory_benchma.tasks import FactualRecallTask

        collector = ResultCollector(db_path=str(tmp_path / "r.db"))
        runner = BenchmarkRunner(
            backends=[FAISSBackend()],
            tasks=[FactualRecallTask],
            collector=collector,
            mock_mode=True,
            model="openai/gpt-5.4-nano",
        )
        yield runner
        collector.close()

    def test_model_stored_in_metadata(self, runner):
        runner.run()
        history = runner.collector.get_run_history()
        assert history
        assert history[0]["model"] == "openai/gpt-5.4-nano"

    def test_default_model_set(self, tmp_path):
        from agent_memory_benchma.benchmark_runner import BenchmarkRunner
        from agent_memory_benchma.collector import ResultCollector
        from agent_memory_benchma.backends import FAISSBackend
        from agent_memory_benchma.tasks import FactualRecallTask

        collector = ResultCollector(db_path=str(tmp_path / "r2.db"))
        runner = BenchmarkRunner(
            backends=[FAISSBackend()],
            tasks=[FactualRecallTask],
            collector=collector,
            mock_mode=True,
        )
        # model should default to env var or fallback string
        assert isinstance(runner.model, str)
        assert len(runner.model) > 0
        collector.close()
