"""Benchmark runner — orchestrates backends × tasks and collects results."""

from __future__ import annotations
import os
import time
import logging
from typing import Any, Generator

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .backends.base import MemoryBackend
from .tasks.base import TaskSuite
from .scorer import Scorer
from .collector import ResultCollector
from .staleness_tracker import StalenessTracker

logger = logging.getLogger(__name__)
console = Console()

_TOP_K = int(os.getenv("BENCHMARK_TOP_K", "3"))


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of *values*, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


class BenchmarkRunner:
    """Run all task suites against all backends and collect results."""

    def __init__(
        self,
        backends: list[MemoryBackend],
        tasks: list[TaskSuite],
        collector: ResultCollector | None = None,
        mock_mode: bool = False,
        top_k: int = _TOP_K,
        model: str = "",
    ) -> None:
        self.backends = backends
        self.tasks = tasks
        self.collector = collector or ResultCollector()
        self.mock_mode = mock_mode
        self.top_k = top_k
        self.model = model or os.getenv("BENCHMARK_MODEL", "openai/gpt-5.4-mini")
        self.scorer = Scorer(mock_mode=mock_mode)
        self.staleness_tracker = StalenessTracker()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Execute all backends × tasks.

        Returns a nested dict::

            {backend_name: {task_name: {accuracy, latency_ms, tokens, confidence, staleness}}}
        """
        run_id = self.collector.record_run_metadata(model=self.model, notes="auto run")
        results: dict[str, dict[str, dict[str, Any]]] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            total_steps = len(self.backends) * len(self.tasks)
            prog_task = progress.add_task("Benchmarking...", total=total_steps)

            for backend in self.backends:
                bname = backend.name
                results[bname] = {}
                self.staleness_tracker.clear(bname)

                for task_cls in self.tasks:
                    task = task_cls()
                    tname = task.name
                    progress.update(
                        prog_task,
                        description=f"[cyan]{bname}[/] × [yellow]{tname}[/]",
                    )

                    task_result = self._run_one(backend, task, run_id)
                    results[bname][tname] = task_result
                    progress.advance(prog_task)

        self.collector.complete_run(run_id)
        self._print_summary(results)
        return results

    def run_streaming(
        self,
    ) -> Generator[dict[str, Any], None, None]:
        """Generator version of :meth:`run` — yields a progress dict after each task.

        Each yielded dict::

            {
                "status":          str,   # human-readable step label
                "elapsed_s":       float,
                "tokens":          int,   # cumulative across all tasks so far
                "turns":           int,   # number of completed (backend × task) steps
                "total":           int,   # total steps
                "partial_results": dict,  # grows as tasks complete
                "done":            bool,
                "run_id":          str,   # available when done=True
            }

        The final yielded dict has ``done=True`` and contains the complete results.
        """
        run_id = self.collector.record_run_metadata(model=self.model, notes="streaming run")
        results: dict[str, dict[str, dict[str, Any]]] = {}
        total_steps = len(self.backends) * len(self.tasks)
        completed = 0
        cumulative_tokens = 0
        t_start = time.perf_counter()

        for backend in self.backends:
            bname = backend.name
            results[bname] = {}
            self.staleness_tracker.clear(bname)

            for task_cls in self.tasks:
                task = task_cls()
                tname = task.name
                completed += 1

                task_result = self._run_one(backend, task, run_id)
                results[bname][tname] = task_result
                cumulative_tokens += task_result.get("tokens", 0)

                yield {
                    "status": f"[{completed}/{total_steps}] {bname} × {tname} ✓  "
                              f"(acc={task_result['accuracy']:.3f}  "
                              f"lat={task_result['latency_ms']:.1f}ms)",
                    "elapsed_s": round(time.perf_counter() - t_start, 2),
                    "tokens": cumulative_tokens,
                    "turns": completed,
                    "total": total_steps,
                    "partial_results": results,
                    "done": False,
                    "run_id": run_id,
                }

        self.collector.complete_run(run_id)
        self._print_summary(results)

        yield {
            "status": "Complete!",
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "tokens": cumulative_tokens,
            "turns": completed,
            "total": total_steps,
            "partial_results": results,
            "done": True,
            "run_id": run_id,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_one(
        self, backend: MemoryBackend, task: TaskSuite, run_id: str
    ) -> dict[str, Any]:
        """Run a single task against a single backend."""
        bname = backend.name
        tname = task.name

        # 1. Load memories
        backend.clear()
        backend.reset_token_usage()
        for mem in task.memories:
            backend.add(mem.content, mem.metadata)
            self.staleness_tracker.record_addition(bname)

        staleness_report = self.staleness_tracker.get_staleness_report(bname)

        # 2. Query
        accuracies: list[float] = []
        latencies: list[float] = []
        confidences: list[float] = []

        for query in task.queries:
            t0 = time.perf_counter()
            retrieved = backend.query(query.text, top_k=self.top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            response = "\n".join(r.get("content", "") for r in retrieved)
            expected = query.expected_phrase or " ".join(query.expected_keywords[:2])
            accuracy = self.scorer.score_response(response, expected)
            confidence = _mean([float(r.get("score", 0.0)) for r in retrieved]) if retrieved else 0.0

            accuracies.append(accuracy)
            latencies.append(latency_ms)
            confidences.append(confidence)

            self.collector.record_result(
                backend=bname,
                task=tname,
                query=query.text,
                response=response,
                expected=expected,
                accuracy=accuracy,
                latency_ms=latency_ms,
                tokens=backend.get_token_usage(),
                confidence=confidence,
                staleness=staleness_report["avg_staleness"],
                run_id=run_id,
            )

        return {
            "accuracy": _mean(accuracies),
            "latency_ms": _mean(latencies),
            "tokens": backend.get_token_usage(),
            "confidence": _mean(confidences),
            "staleness": staleness_report["avg_staleness"],
        }

    def _print_summary(self, results: dict[str, dict[str, dict[str, Any]]]) -> None:
        """Rich table summary printed to console."""
        table = Table(title="Benchmark Summary", show_lines=True)
        table.add_column("Backend", style="cyan", no_wrap=True)
        table.add_column("Task", style="yellow")
        table.add_column("Accuracy", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Staleness", justify="right")

        for bname, task_data in results.items():
            for tname, m in task_data.items():
                table.add_row(
                    bname,
                    tname,
                    f"{m['accuracy']:.3f}",
                    f"{m['latency_ms']:.1f}",
                    str(m["tokens"]),
                    f"{m.get('confidence', 0.0):.3f}",
                    f"{m.get('staleness', 0.0):.3f}",
                )

        console.print(table)
        backend_means = {
            b: _mean([m["accuracy"] for m in td.values()])
            for b, td in results.items()
        }
        if backend_means:
            winner = max(backend_means, key=backend_means.get)
            console.print(
                f"\n[bold green]Overall winner:[/] [bold]{winner}[/] "
                f"(mean accuracy {backend_means[winner]:.3f})"
            )
        if self.model:
            console.print(f"[dim]Model: {self.model}[/]")
