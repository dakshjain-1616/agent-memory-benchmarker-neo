# Agent Memory Benchmarker – Stress-test LLM agent memory backends with standardized metrics

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-255%20passed-brightgreen.svg)]()

> Stop guessing which memory backend fits your agent — benchmark recall accuracy, retrieval latency, token cost, and staleness across ChromaDB, FAISS, Mem0, and SQLite in one command.

## Install

```bash
git clone https://github.com/dakshjain-1616/agent-memory-benchmarker
cd agent-memory-benchmarker
pip install -r requirements.txt
```

## What problem this solves

When deploying LLM agents with LangChain or LlamaIndex, teams select memory backends like ChromaDB, Mem0, or SQLite based on documentation rather than empirical data. The failure mode appears in production: an agent fails to resolve contradictions between old and new information, or burns 10× more tokens on retrieval than generation. Existing evaluation tools measure end-to-end task accuracy but cannot isolate whether the memory subsystem is the bottleneck. Agent Memory Benchmarker fixes this by running 5 reproducible task suites (factual recall, temporal ordering, entity tracking) against a unified scoring rubric to measure recall accuracy, retrieval latency, and memory staleness before you ship.

## Real world examples

```python
from agent_memory_benchma.benchmark_runner import run_suite
from agent_memory_benchma.backends.chromadb_backend import ChromaDBBackend
# Run factual recall suite against ChromaDB
results = run_suite("factual_recall", backend=ChromaDBBackend())
# Output: {'recall_accuracy': 0.92, 'avg_latency_ms': 45}
```

```python
from agent_memory_benchma.reporter import generate_pdf
# Generate performance report from collected results
generate_pdf(results, output_path="memory_benchmark_report.pdf")
# Output: PDF generated with radar charts and latency box plots
```

```bash
# Launch Gradio dashboard to visualize backend comparisons
python -m agent_memory_benchma.leaderboard --port 7860
# Output: Dashboard running at http://127.0.0.1:7860
```

## Who it's for

This tool is for ML Engineers and Agent Developers who are integrating long-term memory into production LLM applications. You need it when you are deciding between vector stores like Pinecone vs. FAISS, or when debugging why your agent hallucinates context after 50 conversation turns. It replaces anecdotal benchmarking with data-driven decisions for memory architecture.

## Quickstart

```python
from agent_memory_benchma.benchmark_runner import run_suite
from agent_memory_benchma.backends.mem0_backend import Mem0Backend

# Initialize backend and run the temporal ordering suite
backend = Mem0Backend()
metrics = run_suite("temporal_ordering", backend=backend)
print(f"Score: {metrics['score']}")
```

## Key features

- 5 reproducible task suites targeting factual recall, temporal ordering, entity tracking, contradiction detection, and long-range dependency.
- Unified scoring rubric combining exact-match overlap and semantic similarity for consistent backend comparison.
- Automated Gradio dashboard with per-backend radar charts and latency box plots, plus PDF report generation.

## Run tests

```bash
pytest tests/ -v  # 255 tests
```

## Project structure

```
agent-memory-benchmarker/
├── agent_memory_benchma/      ← main library
│   ├── __init__.py
│   ├── backends/              ← backend adapters
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chromadb_backend.py
│   │   ├── faiss_backend.py
│   │   ├── mem0_backend.py
│   │   └── sqlite_backend.py
│   ├── benchmark_runner.py
│   ├── collector.py
│   ├── diff_tracker.py
│   ├── leaderboard.py
│   ├── profiles.py
│   ├── reporter.py
│   ├── retry.py
│   └── scorer.py
├── tests/                     ← test suite
├── scripts/                   ← demo scripts
└── requirements.txt
```

---