#!/usr/bin/env python3
"""CLI entry point for the Agent Memory Benchmarker.

Usage examples:
    python demo.py
    python demo.py --backends chromadb faiss --tasks factual_recall temporal_ordering
    python demo.py --mock
    python demo.py --no-pdf
    python demo.py --profile quick
    python demo.py --model openai/gpt-5.4-mini
    python demo.py --verbose
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()


def detect_mock_mode(args_mock: bool) -> bool:
    """Return True if mock mode should be used.

    Forces mock when *args_mock* is True or when no API keys are present.
    """
    if args_mock:
        return True
    has_key = bool(
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    return not has_key


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    from agent_memory_benchma.backends import BACKEND_REGISTRY
    from agent_memory_benchma.tasks import ALL_TASKS
    from agent_memory_benchma.profiles import list_profiles

    parser = argparse.ArgumentParser(
        description="Agent Memory Benchmarker — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=list(BACKEND_REGISTRY.keys()),
        default=None,
        help="Backends to benchmark (default: all). Ignored when --profile is set.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[t.name for t in ALL_TASKS],
        default=None,
        help="Task suites to run (default: all). Ignored when --profile is set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("BENCHMARK_TOP_K", "3")),
        help="Number of memories to retrieve per query",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BENCHMARK_MODEL", "openai/gpt-5.4-mini"),
        help="LLM model ID (stored in run metadata; affects Mem0 backend if API key set)",
    )
    parser.add_argument(
        "--profile",
        choices=list_profiles(),
        default=os.getenv("BENCHMARK_PROFILE", None),
        help=(
            "Preset profile: quick (2 backends/2 tasks), vector (chroma+faiss/all), "
            "standard (all/4 tasks), full (all/all)"
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force mock/dry-run mode (no API calls)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        default=False,
        help="Skip PDF report generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-query breakdown after the run",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("BENCHMARK_OUTPUT_DIR", "outputs"),
        help="Directory for output files",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark, save outputs, and print a summary."""
    args = parse_args()
    mock = detect_mock_mode(args.mock)

    if mock:
        print("[demo] No API keys detected — running in mock mode (deterministic, no API calls).")

    from agent_memory_benchma import BenchmarkRunner, Reporter, ResultCollector
    from agent_memory_benchma.backends import BACKEND_REGISTRY
    from agent_memory_benchma.tasks import ALL_TASKS
    from agent_memory_benchma.profiles import get_profile

    # If a profile is specified, use it to resolve backends and tasks
    if args.profile:
        profile = get_profile(args.profile)
        print(f"[demo] Using profile: {profile.name} — {profile.description}")
        backends = profile.resolve_backends(BACKEND_REGISTRY)
        tasks = profile.resolve_tasks(ALL_TASKS)
        top_k = profile.top_k
    else:
        # Manual selection
        backend_names = args.backends or list(BACKEND_REGISTRY.keys())
        backends = []
        for name in backend_names:
            cls = BACKEND_REGISTRY.get(name)
            if cls:
                try:
                    backends.append(cls())
                except Exception as exc:
                    print(f"  [warn] Could not instantiate {name}: {exc}")

        task_name_set = set(args.tasks) if args.tasks else None
        tasks = [t for t in ALL_TASKS if task_name_set is None or t.name in task_name_set]
        top_k = args.top_k

    os.makedirs(args.output_dir, exist_ok=True)
    collector = ResultCollector(db_path=os.path.join(args.output_dir, "results.db"))

    runner = BenchmarkRunner(
        backends=backends,
        tasks=tasks,
        top_k=top_k,
        mock_mode=mock,
        collector=collector,
        model=args.model,
    )

    print(f"\nModel: {args.model}")
    print("Running benchmarks...\n")
    t_start = time.time()
    results = runner.run()
    elapsed = time.time() - t_start
    print(f"\nBenchmark completed in {elapsed:.1f}s\n")

    # Verbose per-query breakdown
    if args.verbose:
        print("\n=== Verbose Results ===")
        rows = collector.get_all_results()
        for row in rows:
            print(
                f"  [{row['backend']}] {row['task']} | "
                f"Q: {row['query'][:60]}…  "
                f"acc={row['accuracy']:.3f}  lat={row['latency_ms']:.1f}ms  "
                f"conf={row.get('confidence', 0):.3f}"
            )

    # Save raw results JSON
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "mode": "mock" if mock else "real",
                "model": args.model,
                "elapsed_s": round(elapsed, 2),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results JSON: {json_path}")

    # Reports
    reporter = Reporter(output_dir=args.output_dir)
    csv_path = reporter.generate_comparison_csv(results)
    print(f"\nCSV saved: {csv_path}")

    chart_paths = [
        reporter.generate_radar_chart(results),
        reporter.generate_latency_boxplot(results),
        reporter.generate_token_efficiency_chart(results),
        reporter.generate_confidence_chart(results),
        reporter.generate_staleness_chart(results),
    ]

    if not args.no_pdf:
        pdf_path = reporter.generate_pdf_report(
            results,
            chart_paths=[p for p in chart_paths if p],
            filename="benchmark_report.pdf",
            model=args.model,
        )
        print(f"PDF saved:  {pdf_path}")

    # Write summary.txt
    summary_path = os.path.join(args.output_dir, "summary.txt")
    backend_means = {
        b: sum(m["accuracy"] for m in td.values()) / len(td) if td else 0.0
        for b, td in results.items()
    }
    winner = max(backend_means, key=backend_means.get) if backend_means else "n/a"
    with open(summary_path, "w") as sf:
        sf.write(f"Mode: {'mock' if mock else 'real'}\n")
        sf.write(f"Model: {args.model}\n")
        sf.write(f"Elapsed: {elapsed:.2f}s\n")
        sf.write(f"Backends: {', '.join(results.keys())}\n")
        sf.write(f"Tasks: {', '.join(next(iter(results.values())).keys()) if results else ''}\n")
        sf.write(f"Winner: {winner} (mean accuracy {backend_means.get(winner, 0):.3f})\n")
        sf.write("\nPer-backend mean accuracy:\n")
        for b, acc in sorted(backend_means.items(), key=lambda x: x[1], reverse=True):
            sf.write(f"  {b}: {acc:.3f}\n")
    print(f"Summary:    {summary_path}")

    collector.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
