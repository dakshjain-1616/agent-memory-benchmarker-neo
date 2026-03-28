"""Benchmark task suites."""

from .base import TaskSuite, Memory, Query
from .factual_recall import FactualRecallTask
from .temporal_ordering import TemporalOrderingTask
from .entity_tracking import EntityTrackingTask
from .contradiction_detection import ContradictionDetectionTask
from .long_range_dependency import LongRangeDependencyTask
from .multi_session import MultiSessionTask
from .preference_evolution import PreferenceEvolutionTask

__all__ = [
    "TaskSuite",
    "Memory",
    "Query",
    "FactualRecallTask",
    "TemporalOrderingTask",
    "EntityTrackingTask",
    "ContradictionDetectionTask",
    "LongRangeDependencyTask",
    "MultiSessionTask",
    "PreferenceEvolutionTask",
    "ALL_TASKS",
]

ALL_TASKS: list[type] = [
    FactualRecallTask,
    TemporalOrderingTask,
    EntityTrackingTask,
    ContradictionDetectionTask,
    LongRangeDependencyTask,
    MultiSessionTask,
    PreferenceEvolutionTask,
]
