"""04 — Full Pipeline

End-to-end workflow:
  1. Run all backends × all tasks
  2. Persist results to SQLite
  3. Export a comparison CSV
  4. Generate all 5 chart types (radar, latency, token-efficiency,
     confidence, staleness)
  5. Build a PDF report
  6. Query run history
  7. Print the overall winner

No API keys required (mock mode).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json

from agent_memory_benchma import (
    BenchmarkRunner,
    ResultCollector,
    Reporter,
)
from agent_memory_benchma.backends import BACKEND_REGISTRY
from agent_memory_benchma.tasks import ALL_TASKS

OUTPUT_DIR = "outputs/full_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Run benchmarks ────────────────────────────────────────────────────

backends = []
for name, cls in BACKEND_REGISTRY.items():
    try:
        backends.append(cls())
    except Exception as exc:
        print(f"[warn] Could not create {name}: {exc}")

collector = ResultCollector(db_path=os.path.join(OUTPUT_DIR, "results.db"))
run_id = collector.record_run_metadata(model="mock", notes="04_full_pipeline example")

runner = BenchmarkRunner(
    backends=backends,
    tasks=ALL_TASKS,
    mock_mode=True,
    collector=collector,
)

print("Step 1/5 — running benchmarks …")
results = runner.run()
collector.complete_run(run_id)

# ── Step 2: Persist results JSON ──────────────────────────────────────────────

json_path = os.path.join(OUTPUT_DIR, "results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Step 2/5 — results JSON: {json_path}")

# ── Step 3–4: Export CSV and generate charts ──────────────────────────────────

reporter = Reporter(output_dir=OUTPUT_DIR)

csv_path = reporter.generate_comparison_csv(results)
print(f"Step 3/5 — CSV: {csv_path}")

print("Step 4/5 — generating charts …")
chart_paths = [
    reporter.generate_radar_chart(results),
    reporter.generate_latency_boxplot(results),
    reporter.generate_token_efficiency_chart(results),
    reporter.generate_confidence_chart(results),
    reporter.generate_staleness_chart(results),
]
for path in chart_paths:
    if path and os.path.exists(path):
        print(f"  chart: {path}")

# ── Step 5: PDF report ────────────────────────────────────────────────────────

pdf_path = reporter.generate_pdf_report(
    results,
    chart_paths=[p for p in chart_paths if p],
    filename="full_report.pdf",
)
print(f"Step 5/5 — PDF: {pdf_path}")

# ── Step 6: Run history ───────────────────────────────────────────────────────

history = collector.get_run_history()
print(f"\nRun history ({len(history)} run(s) in DB):")
for entry in history[:5]:
    print(f"  run_id={entry['run_id']}  started={entry['started_at']}")

collector.close()

# ── Step 7: Overall winner ────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else 0.0

backend_means = {
    b: mean([m["accuracy"] for m in td.values()])
    for b, td in results.items()
}
winner = max(backend_means, key=backend_means.get)
print(f"\nOverall winner: {winner} (mean accuracy {backend_means[winner]:.3f})")
print("\nAll outputs written to:", OUTPUT_DIR)
