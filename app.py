#!/usr/bin/env python3
"""Gradio web dashboard for the Agent Memory Benchmarker.

Launch with:  python app.py
"""

import os
import json
import time

from dotenv import load_dotenv

load_dotenv()

from agent_memory_benchma import BenchmarkRunner, Reporter, ResultCollector, Leaderboard
from agent_memory_benchma.backends import BACKEND_REGISTRY
from agent_memory_benchma.tasks import ALL_TASKS
from agent_memory_benchma.profiles import PROFILES, get_profile

_OUTPUT_DIR = os.getenv("BENCHMARK_OUTPUT_DIR", "outputs")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ── Model list (newest OpenRouter models) ─────────────────────────────────────

OPENROUTER_MODELS = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4",
    "openai/gpt-5.4-pro",
    "mistralai/mistral-small-2603",
    "x-ai/grok-4.20-beta",
    "x-ai/grok-4.20-multi-agent-beta",
    "minimax/minimax-m2.7",
    "xiaomi/mimo-v2-pro",
    "z-ai/glm-5-turbo",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
]

_DEFAULT_MODEL = os.getenv("BENCHMARK_MODEL", "openai/gpt-5.4-mini")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_backends(selected: list[str]) -> list:
    """Instantiate backend objects from a list of backend names."""
    backends = []
    for name in selected:
        cls = BACKEND_REGISTRY.get(name)
        if cls is None:
            continue
        try:
            backends.append(cls())
        except Exception as exc:
            print(f"  [warn] Could not instantiate backend '{name}': {exc}")
    return backends


def _make_tasks(selected: list[str]) -> list:
    """Filter ALL_TASKS to only those whose name appears in *selected*."""
    name_set = set(selected)
    return [t for t in ALL_TASKS if t.name in name_set]


def _collector() -> ResultCollector:
    """Return a ResultCollector pointed at the shared outputs DB."""
    return ResultCollector(db_path=os.path.join(_OUTPUT_DIR, "results.db"))


# ── Tab handlers ──────────────────────────────────────────────────────────────

def run_quick_benchmark(
    backend_names: list[str],
    task_names: list[str],
    model: str,
    top_k: int,
    mock: bool,
) -> str:
    """Run a quick benchmark and return JSON results."""
    backends = _make_backends(backend_names)
    tasks = _make_tasks(task_names)
    if not backends:
        return json.dumps({"error": "No valid backends selected."})
    if not tasks:
        return json.dumps({"error": "No valid tasks selected."})

    collector = _collector()
    runner = BenchmarkRunner(
        backends=backends,
        tasks=tasks,
        mock_mode=mock,
        top_k=int(top_k),
        collector=collector,
        model=model,
    )
    results = runner.run()
    collector.close()
    return json.dumps(
        {"model": model, "mock": mock, "results": results},
        indent=2,
    )


def run_full_benchmark_streaming(
    backend_names: list[str],
    task_names: list[str],
    model: str,
    top_k: int,
    mock: bool,
    profile_name: str,
):
    """Generator: stream progress updates for the Full Run tab.

    Yields ``(log_text, stats_text, results_json)`` tuples as each task
    completes.  When *profile_name* is not ``"custom"``, the profile's
    backend/task/top-k settings override the manual selections.
    """
    # If a profile is selected (non-empty), override backend/task selection
    if profile_name and profile_name != "custom":
        profile = get_profile(profile_name)
        backends = profile.resolve_backends(BACKEND_REGISTRY)
        tasks = profile.resolve_tasks(ALL_TASKS)
        top_k = profile.top_k
    else:
        backends = _make_backends(backend_names)
        tasks = _make_tasks(task_names)

    if not backends:
        yield "ERROR: No valid backends selected.", "—", ""
        return
    if not tasks:
        yield "ERROR: No valid tasks selected.", "—", ""
        return

    collector = _collector()
    runner = BenchmarkRunner(
        backends=backends,
        tasks=tasks,
        mock_mode=mock,
        top_k=int(top_k),
        collector=collector,
        model=model,
    )

    log_lines: list[str] = []
    t_start = time.time()
    last_update: dict = {}

    for update in runner.run_streaming():
        last_update = update
        log_lines.append(f"[{update['elapsed_s']}s] {update['status']}")
        log_text = "\n".join(log_lines)
        stats_text = (
            f"Tokens: {update['tokens']}  |  "
            f"Turns: {update['turns']}/{update['total']}  |  "
            f"Elapsed: {update['elapsed_s']}s"
        )
        results_json = ""
        if update.get("done"):
            results_json = json.dumps(update["partial_results"], indent=2)
        yield log_text, stats_text, results_json

    collector.close()


def load_leaderboard():
    """Return leaderboard table data and chart path."""
    try:
        collector = _collector()
        lb = Leaderboard(collector)
        rankings = lb.get_rankings()
        collector.close()

        if not rankings:
            return [], None

        reporter = Reporter(output_dir=_OUTPUT_DIR)
        chart_path = reporter.generate_leaderboard_chart(rankings)
        return rankings, chart_path if os.path.exists(chart_path) else None
    except Exception as exc:
        return [], None


def load_run_history():
    """Return run history records from the local results database."""
    try:
        collector = _collector()
        history = collector.get_run_history()
        collector.close()
        return history or []
    except Exception:
        return []


# ── App builder ───────────────────────────────────────────────────────────────

def build_app():
    """Construct and return the Gradio Blocks application."""
    import gradio as gr

    backend_choices = list(BACKEND_REGISTRY.keys())
    task_choices = [t.name for t in ALL_TASKS]

    # Profile radio choices with descriptions
    profile_radio_choices = [
        ("custom — manual backend/task selection", "custom"),
        ("quick — FAISS + SQLite × 2 tasks (fastest, ~2 min)", "quick"),
        ("vector — ChromaDB + FAISS × all 7 tasks", "vector"),
        ("standard — all 4 backends × 4 core tasks", "standard"),
        ("full — all 4 backends × all 7 tasks (most thorough)", "full"),
    ]

    with gr.Blocks(
        title="Agent Memory Benchmarker",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Agent Memory Benchmarker\n"
            "**Stress-test LLM agent memory backends across seven standardized task suites.**\n\n"
            "Measure recall accuracy · retrieval latency · token cost · memory staleness "
            "across ChromaDB, FAISS, Mem0, and SQLite in a single run.\n\n"
            "_Built autonomously by [NEO](https://heyneo.so) — your autonomous AI Agent · "
            "mock mode requires no API keys._"
        )

        # ── Tab 1: Quick Run ──────────────────────────────────────────────────
        with gr.Tab("Quick Run"):
            gr.Markdown("Run a benchmark synchronously and view the full JSON results instantly.")
            with gr.Row():
                q_backends = gr.CheckboxGroup(
                    choices=backend_choices,
                    value=backend_choices,
                    label="Backends",
                )
                q_tasks = gr.CheckboxGroup(
                    choices=task_choices,
                    value=task_choices[:3],
                    label="Task Suites",
                )
            with gr.Row():
                q_model = gr.Dropdown(
                    choices=OPENROUTER_MODELS,
                    value=_DEFAULT_MODEL,
                    label="Model (stored in metadata; used by Mem0 backend if API key set)",
                )
                q_top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top-K retrievals per query")
                q_mock = gr.Checkbox(label="Mock mode — no API keys needed (deterministic Jaccard scoring)", value=True)
            q_run_btn = gr.Button("Run Benchmark", variant="primary")
            q_output = gr.Code(language="json", label="Results JSON")

            q_run_btn.click(
                fn=run_quick_benchmark,
                inputs=[q_backends, q_tasks, q_model, q_top_k, q_mock],
                outputs=q_output,
            )

        # ── Tab 2: Full Run (streaming) ───────────────────────────────────────
        with gr.Tab("Full Run"):
            gr.Markdown(
                "Run a full benchmark with **live progress streaming**. "
                "Choose a preset profile or configure manually."
            )
            with gr.Accordion("Profile", open=True):
                f_profile = gr.Radio(
                    choices=profile_radio_choices,
                    value="standard",
                    label="Preset Profile",
                    info="Profiles override the manual backend/task selections below.",
                )
            with gr.Accordion("Model & Mode", open=True):
                with gr.Row():
                    f_model = gr.Dropdown(
                        choices=OPENROUTER_MODELS,
                        value=_DEFAULT_MODEL,
                        label="Model",
                    )
                    f_mock = gr.Checkbox(label="Mock mode (no API keys needed)", value=True)
                    f_top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top-K")
            with gr.Accordion("Manual Selection (used when profile = custom)", open=False):
                with gr.Row():
                    f_backends = gr.CheckboxGroup(
                        choices=backend_choices,
                        value=backend_choices,
                        label="Backends",
                    )
                    f_tasks = gr.CheckboxGroup(
                        choices=task_choices,
                        value=task_choices,
                        label="Task Suites",
                    )

            f_run_btn = gr.Button("Start Full Benchmark", variant="primary")

            # Live stats bar
            f_stats = gr.Textbox(
                label="Live Stats  (Tokens | Turns | Elapsed)",
                interactive=False,
                lines=1,
            )
            f_log = gr.Textbox(
                label="Progress Log",
                interactive=False,
                lines=12,
                max_lines=30,
                autoscroll=True,
            )
            f_results = gr.Code(language="json", label="Final Results JSON")

            f_run_btn.click(
                fn=run_full_benchmark_streaming,
                inputs=[f_backends, f_tasks, f_model, f_top_k, f_mock, f_profile],
                outputs=[f_log, f_stats, f_results],
            )

        # ── Tab 3: Leaderboard ────────────────────────────────────────────────
        with gr.Tab("Leaderboard"):
            gr.Markdown(
                "Historical rankings computed from all benchmark runs stored in the local DB. "
                "Click **Refresh** after running new benchmarks."
            )
            lb_refresh_btn = gr.Button("Refresh Leaderboard", variant="secondary")
            lb_table = gr.Dataframe(
                headers=["rank", "backend", "mean_accuracy", "best_accuracy",
                         "worst_accuracy", "mean_latency_ms", "mean_confidence", "run_count"],
                label="Backend Rankings",
                interactive=False,
            )
            lb_chart = gr.Image(label="Leaderboard Chart", type="filepath")

            def refresh_leaderboard():
                """Reload rankings from DB and regenerate chart."""
                rankings, chart_path = load_leaderboard()
                rows = [
                    [
                        r.get("rank"), r.get("backend"),
                        r.get("mean_accuracy"), r.get("best_accuracy"),
                        r.get("worst_accuracy"), r.get("mean_latency_ms"),
                        r.get("mean_confidence"), r.get("run_count"),
                    ]
                    for r in rankings
                ]
                return rows, chart_path

            lb_refresh_btn.click(
                fn=refresh_leaderboard,
                inputs=[],
                outputs=[lb_table, lb_chart],
            )

        # ── Tab 4: Run History ────────────────────────────────────────────────
        with gr.Tab("Run History"):
            gr.Markdown("All past benchmark runs recorded in the local results database.")
            rh_refresh_btn = gr.Button("Refresh History", variant="secondary")
            rh_table = gr.Dataframe(
                headers=["run_id", "started_at", "completed_at", "model", "notes"],
                label="Run History",
                interactive=False,
            )

            def refresh_history():
                """Reload run history from DB."""
                history = load_run_history()
                rows = [
                    [
                        h.get("run_id"), h.get("started_at"),
                        h.get("completed_at"), h.get("model"), h.get("notes"),
                    ]
                    for h in history
                ]
                return rows

            rh_refresh_btn.click(
                fn=refresh_history,
                inputs=[],
                outputs=rh_table,
            )

        # ── Tab 5: Scenario Cards ─────────────────────────────────────────────
        with gr.Tab("Scenario Cards"):
            gr.Markdown(
                "Each task suite below shows its description, sample memories, "
                "and example queries."
            )
            for task_cls in ALL_TASKS:
                task = task_cls()
                sample_mems = "\n".join(
                    f"• {m.content[:120]}{'…' if len(m.content) > 120 else ''}"
                    for m in task.memories[:3]
                )
                sample_qs = "\n".join(
                    f"Q: {q.text}  →  keywords: {', '.join(q.expected_keywords[:3])}"
                    for q in task.queries[:3]
                )
                with gr.Accordion(
                    label=f"{task.name.replace('_', ' ').title()}  —  {task.description}",
                    open=False,
                ):
                    gr.Markdown(
                        f"**Sample Memories ({len(task.memories)} total)**\n```\n{sample_mems}\n```\n\n"
                        f"**Sample Queries ({len(task.queries)} total)**\n```\n{sample_qs}\n```"
                    )

        # ── Tab 6: About ──────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(
                "## Backends\n"
                "| Backend | Description |\n"
                "|---------|-------------|\n"
                "| **chromadb** | In-process ChromaDB with hash embeddings |\n"
                "| **faiss** | FAISS flat-L2 index (no external deps) |\n"
                "| **mem0** | Mem0 managed memory (needs API key or uses mock) |\n"
                "| **sqlite** | SQLite FTS5 full-text search |\n\n"
                "## Task Suites\n"
                + "\n".join(
                    f"- **{task_cls().name}** — {task_cls().description}"
                    for task_cls in ALL_TASKS
                )
                + "\n\n"
                "## Profiles\n"
                + "\n".join(
                    f"- **{name}** — {p.description}"
                    for name, p in PROFILES.items()
                )
                + "\n\n"
                "## Environment Variables\n"
                "```\n"
                "BENCHMARK_MODEL=openai/gpt-5.4-mini\n"
                "BENCHMARK_TOP_K=3\n"
                "BENCHMARK_OUTPUT_DIR=outputs\n"
                "BENCHMARK_PROFILE=standard\n"
                "OPENROUTER_API_KEY=...\n"
                "GRADIO_HOST=127.0.0.1\n"
                "GRADIO_PORT=7860\n"
                "GRADIO_SHARE=false\n"
                "```\n\n"
                "---\n"
                "_Built autonomously by [NEO](https://heyneo.so) — your autonomous AI Agent._"
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
