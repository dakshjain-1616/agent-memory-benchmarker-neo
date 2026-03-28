"""01 — Quick Start

Minimal working example: run a single backend against a single task
and print the accuracy score.  No API keys required (mock mode).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent_memory_benchma import BenchmarkRunner, ResultCollector
from agent_memory_benchma.backends import SQLiteBackend
from agent_memory_benchma.tasks import FactualRecallTask

# 1. Instantiate one backend and one task suite
backend = SQLiteBackend()
tasks = [FactualRecallTask]          # pass the class, not an instance

# 2. Create a collector backed by a temporary in-memory DB
collector = ResultCollector(db_path=":memory:")

# 3. Run the benchmark
runner = BenchmarkRunner(
    backends=[backend],
    tasks=tasks,
    mock_mode=True,                  # no API keys needed
    collector=collector,
)

results = runner.run()

# 4. Print the result
acc = results["sqlite"]["factual_recall"]["accuracy"]
lat = results["sqlite"]["factual_recall"]["latency_ms"]
print(f"SQLite × Factual Recall  |  accuracy={acc:.3f}  latency={lat:.1f}ms")
