"""Tests: all task suites return valid memories and queries."""

import pytest
from agent_memory_benchma.tasks import (
    ALL_TASKS,
    FactualRecallTask,
    TemporalOrderingTask,
    EntityTrackingTask,
    ContradictionDetectionTask,
    LongRangeDependencyTask,
    MultiSessionTask,
    PreferenceEvolutionTask,
)
from agent_memory_benchma.tasks.base import Memory, Query


# ── ALL_TASKS completeness ─────────────────────────────────────────────────────

def test_all_tasks_nonempty():
    assert len(ALL_TASKS) >= 5


def test_all_tasks_have_name():
    for task_cls in ALL_TASKS:
        t = task_cls()
        assert isinstance(t.name, str) and len(t.name) > 0


def test_all_tasks_have_description():
    for task_cls in ALL_TASKS:
        t = task_cls()
        assert isinstance(t.description, str) and len(t.description) > 0


# ── Memory structure ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_cls", ALL_TASKS)
def test_memories_nonempty(task_cls):
    t = task_cls()
    assert len(t.memories) > 0


@pytest.mark.parametrize("task_cls", ALL_TASKS)
def test_memories_are_Memory_instances(task_cls):
    t = task_cls()
    for mem in t.memories:
        assert isinstance(mem, Memory)
        assert isinstance(mem.content, str) and len(mem.content) > 0


@pytest.mark.parametrize("task_cls", ALL_TASKS)
def test_queries_nonempty(task_cls):
    t = task_cls()
    assert len(t.queries) > 0


@pytest.mark.parametrize("task_cls", ALL_TASKS)
def test_queries_are_Query_instances(task_cls):
    t = task_cls()
    for q in t.queries:
        assert isinstance(q, Query)
        assert isinstance(q.text, str) and len(q.text) > 0
        assert isinstance(q.expected_keywords, list) and len(q.expected_keywords) > 0


# ── Task-specific checks ──────────────────────────────────────────────────────

def test_factual_recall_has_at_least_5_queries():
    t = FactualRecallTask()
    assert len(t.queries) >= 5


def test_temporal_ordering_memories_have_dates():
    t = TemporalOrderingTask()
    # at least some memories mention year numbers
    years_found = any(
        any(str(y) in m.content for y in range(1940, 2030))
        for m in t.memories
    )
    assert years_found


def test_entity_tracking_mentions_multiple_entities():
    t = EntityTrackingTask()
    # Should mention at least 3 different named entities
    names = {"Bob", "Carol", "David", "Eve"}
    found = sum(1 for n in names if any(n in m.content for m in t.memories))
    assert found >= 3


def test_contradiction_detection_has_corrections():
    t = ContradictionDetectionTask()
    # At least one memory should mention CORRECTION or UPDATE
    has_correction = any(
        ("CORRECTION" in m.content or "UPDATE" in m.content)
        for m in t.memories
    )
    assert has_correction


def test_long_range_dependency_queries_have_keywords():
    t = LongRangeDependencyTask()
    for q in t.queries:
        assert len(q.expected_keywords) >= 1


def test_multi_session_memories_span_sessions():
    t = MultiSessionTask()
    sessions = {m.metadata.get("session") for m in t.memories if m.metadata}
    assert len(sessions) >= 2


def test_preference_evolution_has_updates():
    t = PreferenceEvolutionTask()
    updates = [m for m in t.memories if "UPDATE" in m.content]
    assert len(updates) >= 2
