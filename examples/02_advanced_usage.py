"""02 — Advanced Usage

Run all 4 backends against all 7 task suites, collect results into a
SQLite database, and display a summary sorted by mean accuracy.
No API keys required (mock mode).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent_memory_benchma import BenchmarkRunner, ResultCollector
from agent_memory_benchma.backends import BACKEND_REGISTRY
from agent_memory_benchma.tasks import ALL_TASKS

# 1. Instantiate all backends
backends = []
for name, cls in BACKEND_REGISTRY.items():
    try:
        backends.append(cls())
        print(f"  [+] {name} ready")
    except Exception as exc:
        print(f"  [-] {name} failed to initialise: {exc}")

# 2. Persist results to a file
os.makedirs("outputs", exist_ok=True)
collector = ResultCollector(db_path="outputs/advanced_results.db")
run_id = collector.record_run_metadata(model="mock", notes="02_advanced_usage example")

# 3. Run
runner = BenchmarkRunner(
    backends=backends,
    tasks=ALL_TASKS,
    mock_mode=True,
    collector=collector,
    top_k=5,                         # retrieve 5 chunks per query
)

print("\nBenchmarking all backends × all tasks …\n")
results = runner.run()
collector.complete_run(run_id)

# 4. Summary: rank backends by mean accuracy
print("\n--- Mean accuracy by backend ---")
for backend_name, task_data in results.items():
    accs = [m["accuracy"] for m in task_data.values()]
    mean = sum(accs) / len(accs) if accs else 0
    print(f"  {backend_name:12s}  {mean:.3f}")

collector.close()
