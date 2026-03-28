# Examples

All scripts can be run from any directory — they insert the project root into
`sys.path` automatically, so no install step is required.

```
python examples/01_quick_start.py
python examples/02_advanced_usage.py
python examples/03_custom_config.py
python examples/04_full_pipeline.py
```

---

| Script | What it demonstrates |
|--------|----------------------|
| [`01_quick_start.py`](01_quick_start.py) | Minimal working example: one backend, one task, print accuracy. 20 lines. |
| [`02_advanced_usage.py`](02_advanced_usage.py) | All 4 backends × all 7 tasks; results persisted to SQLite; backends ranked by mean accuracy. |
| [`03_custom_config.py`](03_custom_config.py) | Override scorer weights, staleness half-life, and `top_k` via environment variables and constructor parameters. |
| [`04_full_pipeline.py`](04_full_pipeline.py) | Full end-to-end workflow: run → JSON → CSV → 5 charts → PDF report → run history → winner summary. |

---

## Requirements

No API keys are needed — all examples run in **mock mode** by default.

Install dependencies:

```
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and add real API keys to run against
live LLM services.
