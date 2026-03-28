"""Chart generation and PDF report building."""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_OUTPUT_DIR = os.getenv("BENCHMARK_OUTPUT_DIR", "outputs")


class Reporter:
    """Generate charts and PDF reports from benchmark results."""

    def __init__(self, output_dir: str = _OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> str:
        """Return the absolute path for an output file inside *output_dir*."""
        return os.path.join(self.output_dir, filename)

    @staticmethod
    def _backends_tasks(results: Dict) -> tuple:
        """Extract ordered backend names and task names from a results dict."""
        backends = list(results.keys())
        tasks = list(next(iter(results.values())).keys()) if backends else []
        return backends, tasks

    # ------------------------------------------------------------------
    # Chart generators
    # ------------------------------------------------------------------

    def generate_radar_chart(self, results: Dict, filename: str = "radar.png") -> str:
        """Accuracy spider/radar plot for each backend."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            backends, tasks = self._backends_tasks(results)
            if not tasks:
                return ""

            angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([t.replace("_", "\n") for t in tasks], size=8)
            ax.set_ylim(0, 1)

            for backend in backends:
                values = [results[backend].get(t, {}).get("accuracy", 0) for t in tasks]
                values += values[:1]
                ax.plot(angles, values, linewidth=2, label=backend)
                ax.fill(angles, values, alpha=0.1)

            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.set_title("Accuracy by Backend & Task", pad=20)

            path = self._path(filename)
            plt.tight_layout()
            plt.savefig(path, dpi=100)
            plt.close()
            return path
        except Exception as exc:
            logger.warning("radar chart failed: %s", exc)
            return ""

    def generate_latency_boxplot(self, results: Dict, filename: str = "latency.png") -> str:
        """Bar chart of mean latency per backend."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            backends, tasks = self._backends_tasks(results)
            x = range(len(backends))
            for i, task in enumerate(tasks):
                values = [results[b].get(task, {}).get("latency_ms", 0) for b in backends]
                plt.bar([xi + i * 0.1 for xi in x], values, width=0.1, label=task)

            plt.xticks(x, backends, rotation=15)
            plt.ylabel("Latency (ms)")
            plt.title("Retrieval Latency by Backend")
            plt.legend(fontsize=7)
            plt.tight_layout()

            path = self._path(filename)
            plt.savefig(path, dpi=100)
            plt.close()
            return path
        except Exception as exc:
            logger.warning("latency chart failed: %s", exc)
            return ""

    def generate_token_efficiency_chart(self, results: Dict, filename: str = "tokens.png") -> str:
        """Scatter plot: tokens used vs accuracy."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            for backend, task_data in results.items():
                tokens = [d.get("tokens", 0) for d in task_data.values()]
                accs = [d.get("accuracy", 0) for d in task_data.values()]
                ax.scatter(tokens, accs, label=backend, alpha=0.8)

            ax.set_xlabel("Tokens Used")
            ax.set_ylabel("Accuracy")
            ax.set_title("Token Efficiency")
            ax.legend()
            plt.tight_layout()

            path = self._path(filename)
            plt.savefig(path, dpi=100)
            plt.close()
            return path
        except Exception as exc:
            logger.warning("token efficiency chart failed: %s", exc)
            return ""

    def generate_confidence_chart(self, results: Dict, filename: str = "confidence.png") -> str:
        """Bar chart of confidence per backend."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            backends, _ = self._backends_tasks(results)
            confidences = []
            for b in backends:
                vals = [d.get("confidence", 0) for d in results[b].values()]
                confidences.append(sum(vals) / len(vals) if vals else 0)

            plt.bar(backends, confidences, color="steelblue")
            plt.ylim(0, 1)
            plt.ylabel("Confidence")
            plt.title("Mean Confidence by Backend")
            plt.xticks(rotation=15)
            plt.tight_layout()

            path = self._path(filename)
            plt.savefig(path, dpi=100)
            plt.close()
            return path
        except Exception as exc:
            logger.warning("confidence chart failed: %s", exc)
            return ""

    def generate_staleness_chart(self, results: Dict, filename: str = "staleness.png") -> str:
        """Bar chart of staleness per backend."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            backends, _ = self._backends_tasks(results)
            stalenesses = []
            for b in backends:
                vals = [d.get("staleness", 0) for d in results[b].values()]
                stalenesses.append(sum(vals) / len(vals) if vals else 0)

            plt.bar(backends, stalenesses, color="coral")
            plt.ylim(0, 1)
            plt.ylabel("Staleness")
            plt.title("Mean Staleness by Backend")
            plt.xticks(rotation=15)
            plt.tight_layout()

            path = self._path(filename)
            plt.savefig(path, dpi=100)
            plt.close()
            return path
        except Exception as exc:
            logger.warning("staleness chart failed: %s", exc)
            return ""

    def generate_leaderboard_chart(
        self,
        leaderboard_data: List[Dict],
        filename: str = "leaderboard.png",
    ) -> str:
        """Horizontal bar chart of per-backend mean accuracy from leaderboard data.

        *leaderboard_data* should be the list returned by
        :meth:`~agent_memory_benchma.leaderboard.Leaderboard.get_rankings`.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not leaderboard_data:
                return ""

            backends = [e["backend"] for e in leaderboard_data]
            accs = [e["mean_accuracy"] for e in leaderboard_data]
            colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(backends))]

            fig, ax = plt.subplots(figsize=(8, max(3, len(backends) * 0.8)))
            bars = ax.barh(backends[::-1], accs[::-1], color=colors[::-1], edgecolor="white")

            for bar, acc in zip(bars, accs[::-1]):
                ax.text(
                    bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.3f}", va="center", fontsize=9,
                )

            ax.set_xlim(0, min(1.1, max(accs) * 1.15) if accs else 1.0)
            ax.set_xlabel("Mean Accuracy")
            ax.set_title("Backend Leaderboard — Mean Accuracy Across All Runs")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()

            path = self._path(filename)
            plt.savefig(path, dpi=100)
            plt.close()
            return path
        except Exception as exc:
            logger.warning("leaderboard chart failed: %s", exc)
            return ""

    def generate_comparison_csv(self, results: Dict, filename: str = "comparison.csv") -> str:
        """Flat CSV of all backend × task results."""
        path = self._path(filename)
        try:
            import csv
            rows = []
            for backend, task_data in results.items():
                for task, metrics in task_data.items():
                    rows.append({"backend": backend, "task": task, **metrics})
            if rows:
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as exc:
            logger.warning("CSV export failed: %s", exc)
        return path

    def generate_pdf_report(
        self,
        results: Dict,
        chart_paths: Optional[List[str]] = None,
        filename: str = "report.pdf",
        model: str = "",
        run_id: str = "",
    ) -> str:
        """Multi-page PDF with summary table, metadata, and embedded charts."""
        path = self._path(filename)
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle, Paragraph,
                Spacer, Image,
            )
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("Agent Memory Benchmarker — Results Report", styles["Title"]))
            story.append(Spacer(1, 6))

            # Metadata block
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            meta_lines = [f"Generated: {now}"]
            if model:
                meta_lines.append(f"Model: {model}")
            if run_id:
                meta_lines.append(f"Run ID: {run_id}")
            for line in meta_lines:
                story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 12))

            # Summary table
            headers = ["Backend", "Task", "Accuracy", "Latency (ms)", "Tokens", "Confidence", "Staleness"]
            table_data = [headers]
            for backend, task_data in results.items():
                for task, m in task_data.items():
                    table_data.append([
                        backend, task,
                        f"{m.get('accuracy', 0):.3f}",
                        f"{m.get('latency_ms', 0):.1f}",
                        str(m.get("tokens", 0)),
                        f"{m.get('confidence', 0):.3f}",
                        f"{m.get('staleness', 0):.3f}",
                    ])

            tbl = Table(table_data)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 20))

            # Charts
            for cp in (chart_paths or []):
                if cp and os.path.exists(cp):
                    story.append(Image(cp, width=420, height=280))
                    story.append(Spacer(1, 12))

            doc.build(story)
        except Exception as exc:
            logger.warning("PDF report failed: %s", exc)
        return path
