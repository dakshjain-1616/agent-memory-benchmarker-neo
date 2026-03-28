"""Agent Memory Benchmarker — public API."""

__version__ = "1.1.0"

from .benchmark_runner import BenchmarkRunner
from .scorer import Scorer
from .collector import ResultCollector
from .reporter import Reporter
from .staleness_tracker import StalenessTracker, compute_staleness
from .retry import with_retry, retry_call
from .leaderboard import Leaderboard
from .profiles import BenchmarkProfile, PROFILES, get_profile, list_profiles
from .diff_tracker import MemoryDiffTracker
from .backends import (
    MemoryBackend,
    ChromaDBBackend,
    FAISSBackend,
    Mem0Backend,
    SQLiteBackend,
    BACKEND_REGISTRY,
)
from .tasks import (
    TaskSuite,
    Memory,
    Query,
    ALL_TASKS,
    FactualRecallTask,
    TemporalOrderingTask,
    EntityTrackingTask,
    ContradictionDetectionTask,
    LongRangeDependencyTask,
    MultiSessionTask,
    PreferenceEvolutionTask,
)

__all__ = [
    # Runner
    "BenchmarkRunner",
    # Scoring / collection / reporting
    "Scorer",
    "ResultCollector",
    "Reporter",
    "StalenessTracker",
    "compute_staleness",
    # Retry
    "with_retry",
    "retry_call",
    # Leaderboard & profiles
    "Leaderboard",
    "BenchmarkProfile",
    "PROFILES",
    "get_profile",
    "list_profiles",
    # Diff tracking
    "MemoryDiffTracker",
    # Backends
    "MemoryBackend",
    "ChromaDBBackend",
    "FAISSBackend",
    "Mem0Backend",
    "SQLiteBackend",
    "BACKEND_REGISTRY",
    # Tasks
    "TaskSuite",
    "Memory",
    "Query",
    "ALL_TASKS",
    "FactualRecallTask",
    "TemporalOrderingTask",
    "EntityTrackingTask",
    "ContradictionDetectionTask",
    "LongRangeDependencyTask",
    "MultiSessionTask",
    "PreferenceEvolutionTask",
]
