"""03 — Custom Configuration

Shows how to customise behaviour via environment variables and
constructor parameters — scorer weights, staleness half-life, retry
settings, and output directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Override settings via environment *before* importing the package
os.environ.setdefault("SCORER_EXACT_WEIGHT", "0.6")    # weight exact-match higher
os.environ.setdefault("SCORER_SEMANTIC_WEIGHT", "0.4") # reduce semantic weight
os.environ.setdefault("MEMORY_DECAY_HALFLIFE", "7200")  # 2-hour half-life
os.environ.setdefault("BENCHMARK_TOP_K", "5")
os.environ.setdefault("BENCHMARK_OUTPUT_DIR", "outputs/custom")

from agent_memory_benchma import (
    BenchmarkRunner,
    ResultCollector,
    StalenessTracker,
    Scorer,
)
from agent_memory_benchma.backends import FAISSBackend, ChromaDBBackend
from agent_memory_benchma.tasks import (
    ContradictionDetectionTask,
    PreferenceEvolutionTask,
)

# 1. Build components with custom settings
scorer = Scorer(
    exact_weight=float(os.environ["SCORER_EXACT_WEIGHT"]),
    semantic_weight=float(os.environ["SCORER_SEMANTIC_WEIGHT"]),
    mock_mode=True,
)
print(f"Scorer weights — exact: {scorer.exact_weight}  semantic: {scorer.semantic_weight}")

staleness_tracker = StalenessTracker(
    halflife=float(os.environ["MEMORY_DECAY_HALFLIFE"])
)
print(f"Staleness half-life: {staleness_tracker.halflife}s")

# 2. Run with custom components
os.makedirs(os.environ["BENCHMARK_OUTPUT_DIR"], exist_ok=True)
collector = ResultCollector(
    db_path=os.path.join(os.environ["BENCHMARK_OUTPUT_DIR"], "results.db")
)

runner = BenchmarkRunner(
    backends=[FAISSBackend(), ChromaDBBackend()],
    tasks=[ContradictionDetectionTask, PreferenceEvolutionTask],
    mock_mode=True,
    collector=collector,
    top_k=int(os.environ["BENCHMARK_TOP_K"]),
)

print("\nRunning customised benchmark …\n")
results = runner.run()
collector.close()

# 3. Show staleness directly
for backend_name in ["faiss", "chromadb"]:
    if backend_name in results:
        stale = results[backend_name]
        for task_name, metrics in stale.items():
            print(
                f"  {backend_name:12s} | {task_name:28s} | "
                f"staleness={metrics.get('staleness', 0):.4f}"
            )
