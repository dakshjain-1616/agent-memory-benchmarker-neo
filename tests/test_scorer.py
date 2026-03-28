"""Tests: Scorer, staleness tracker, and result collector."""

import pytest
import time
from agent_memory_benchma.scorer import Scorer
from agent_memory_benchma.staleness_tracker import StalenessTracker, compute_staleness
from agent_memory_benchma.collector import ResultCollector


# ── Scorer ────────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    return Scorer(mock_mode=True)


def test_scorer_exact_match_identical(scorer):
    score = scorer.exact_match("hello world", "hello world")
    assert score >= 0.9


def test_scorer_exact_match_substring_bonus(scorer):
    score = scorer.exact_match("the deadline is April 1st", "April 1st")
    assert score > 0.0


def test_scorer_exact_match_no_overlap(scorer):
    score = scorer.exact_match("quantum physics research", "budget allocation")
    assert score < 0.5


def test_scorer_semantic_similarity_range(scorer):
    score = scorer.semantic_similarity("the cat sat on the mat", "cat sitting on mat")
    assert 0.0 <= score <= 1.0


def test_scorer_semantic_empty_string(scorer):
    score = scorer.semantic_similarity("", "something")
    assert 0.0 <= score <= 1.0


def test_scorer_score_response_range(scorer):
    score = scorer.score_response("Alice is 30 years old", "Alice")
    assert 0.0 <= score <= 1.0


def test_scorer_score_response_relevant_higher(scorer):
    high = scorer.score_response("Python was created by Guido van Rossum in 1991", "Guido van Rossum")
    low = scorer.score_response("The weather in Paris is mild in spring", "Guido van Rossum")
    assert high > low


# ── Staleness Tracker ─────────────────────────────────────────────────────────

def test_staleness_fresh_memory():
    s = compute_staleness(age_seconds=0)
    assert s == pytest.approx(0.0, abs=0.01)


def test_staleness_old_memory():
    s = compute_staleness(age_seconds=3600)  # 1 halflife
    assert 0.4 < s < 0.6  # ~0.5 after one half-life


def test_staleness_very_old():
    s = compute_staleness(age_seconds=36000)
    assert s > 0.9


def test_staleness_tracker_record():
    tracker = StalenessTracker(halflife=3600)
    tracker.record_addition("chromadb")
    report = tracker.get_staleness_report("chromadb")
    assert report["count"] == 1
    assert report["avg_freshness"] > 0.99  # just added, very fresh


def test_staleness_tracker_aging():
    tracker = StalenessTracker(halflife=3600)
    tracker.record_addition("faiss")
    tracker.simulate_aging("faiss", age_seconds=3600)
    report = tracker.get_staleness_report("faiss")
    assert report["avg_staleness"] > 0.4


def test_staleness_tracker_clear():
    tracker = StalenessTracker()
    tracker.record_addition("mem0")
    tracker.clear("mem0")
    report = tracker.get_staleness_report("mem0")
    assert report["count"] == 0


# ── ResultCollector ──────────────────────────────────────────────────────────

@pytest.fixture
def collector(tmp_path):
    db = str(tmp_path / "test_results.db")
    c = ResultCollector(db_path=db)
    yield c
    c.close()


def test_collector_record_and_retrieve(collector):
    run_id = collector.record_run_metadata(model="test", notes="pytest")
    collector.record_result(
        backend="chromadb",
        task="factual_recall",
        query="What is Alice's favourite colour?",
        response="Alice's favourite colour is cerulean blue.",
        expected="cerulean blue",
        accuracy=1.0,
        latency_ms=5.2,
        tokens=0,
        run_id=run_id,
    )
    rows = collector.get_all_results(run_id=run_id)
    assert len(rows) == 1
    assert rows[0]["backend"] == "chromadb"
    assert rows[0]["accuracy"] == pytest.approx(1.0)


def test_collector_summary_df(collector):
    run_id = collector.record_run_metadata()
    for b in ["faiss", "sqlite"]:
        for t in ["factual_recall", "temporal_ordering"]:
            collector.record_result(
                backend=b, task=t, query="q", response="r",
                expected="e", accuracy=0.8, latency_ms=10.0, tokens=0, run_id=run_id,
            )
    df = collector.get_summary_df()
    assert df is not None
    assert len(df) >= 2


def test_collector_export_csv(collector, tmp_path):
    run_id = collector.record_run_metadata()
    collector.record_result(
        backend="mem0", task="entity_tracking", query="q", response="r",
        expected="e", accuracy=0.7, latency_ms=8.0, tokens=0, run_id=run_id,
    )
    csv_path = str(tmp_path / "out.csv")
    collector.export_csv(csv_path)
    import os
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 0
