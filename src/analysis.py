"""
Results Analyzer for CloudRAG-Adapt.

Generates all figures and tables for the research paper:
  - Quality metrics comparison (bar chart)
  - Resource consumption comparison (bar charts)
  - Quality vs. Resource Pareto front (scatter)
  - Routing distribution (pie chart)
  - Latency breakdown per bundle (stacked bar)
  - Statistical tests (paired t-test, Cohen's d)
  - LaTeX-formatted summary tables
"""

import os
import glob
import json
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Style configuration ──
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = sns.color_palette("Set2", 8)


class ResultsAnalyzer:
    """Loads experiment CSVs and generates figures, tables, and statistical reports."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.experiments = self._load_all_experiments()

    def _load_all_experiments(self) -> Dict[str, pd.DataFrame]:
        """Load all non-intermediate experiment CSVs."""
        experiments = {}
        csv_files = sorted(glob.glob(os.path.join(self.results_dir, "*.csv")))
        for f in csv_files:
            basename = os.path.basename(f).replace(".csv", "")
            if "_intermediate" in basename or basename == "experiment_summary":
                continue
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    experiments[basename] = df
                    logger.info(f"Loaded {basename}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")
        return experiments

    # ──────────────────────────────────────────
    #  Quality Comparison
    # ──────────────────────────────────────────

    def plot_quality_comparison(self):
        """Bar chart: RAGAS metrics across all experiments."""
        if not self.experiments:
            return

        metrics = ["answer_relevancy", "context_precision", "context_recall", "faithfulness"]
        data = []

        for exp_name, df in self.experiments.items():
            for m in metrics:
                if m in df.columns:
                    data.append({
                        "Experiment": exp_name,
                        "Metric": m.replace("_", " ").title(),
                        "Value": df[m].mean(),
                        "Std": df[m].std(),
                    })

        if not data:
            return

        plot_df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=plot_df, x="Experiment", y="Value", hue="Metric",
                    palette="Set2", ax=ax)
        ax.set_title("Quality Metrics Comparison Across Experiments")
        ax.set_ylabel("Score (0-1)")
        ax.set_ylim(0, 1.05)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        path = os.path.join(self.figures_dir, "quality_comparison.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")

    # ──────────────────────────────────────────
    #  Resource Comparison
    # ──────────────────────────────────────────

    def plot_resource_comparison(self):
        """Bar charts: CPU time, memory, wall time across experiments."""
        if not self.experiments:
            return

        resource_specs = {
            "wall_time_seconds": ("Average Latency (seconds)", "Latency Comparison"),
            "cpu_time_seconds": ("CPU Time (seconds)", "CPU Time Comparison"),
            "memory_peak_mb": ("Peak Memory (MB)", "Memory Usage Comparison"),
        }

        for col, (ylabel, title) in resource_specs.items():
            data = []
            for exp_name, df in self.experiments.items():
                if col in df.columns:
                    data.append({
                        "Experiment": exp_name,
                        "Mean": df[col].mean(),
                        "Std": df[col].std(),
                    })

            if not data:
                continue

            plot_df = pd.DataFrame(data)
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(plot_df["Experiment"], plot_df["Mean"],
                          yerr=plot_df["Std"], capsize=4, color=PALETTE[:len(data)],
                          edgecolor="gray", linewidth=0.5)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            path = os.path.join(self.figures_dir, f"{col}_comparison.png")
            plt.savefig(path)
            plt.close()
            logger.info(f"Saved: {path}")

    # ──────────────────────────────────────────
    #  Pareto Front
    # ──────────────────────────────────────────

    def plot_pareto_front(self):
        """
        Scatter plot: Quality (faithfulness) vs. Resource (wall time).
        Highlights Pareto-optimal configurations.
        """
        if not self.experiments:
            return

        names, x_vals, y_vals = [], [], []

        for exp_name, df in self.experiments.items():
            if "faithfulness" in df.columns and "wall_time_seconds" in df.columns:
                names.append(exp_name)
                x_vals.append(df["wall_time_seconds"].mean())
                y_vals.append(df["faithfulness"].mean())

        if not names:
            return

        # Find Pareto-optimal points (maximize quality, minimize latency)
        pareto_mask = self._compute_pareto_mask(
            np.array(x_vals), np.array(y_vals), minimize_x=True, maximize_y=True
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot all points
        ax.scatter(x_vals, y_vals, c="steelblue", s=120, zorder=3, edgecolors="white", linewidth=1.5)

        # Highlight Pareto-optimal
        for i, is_pareto in enumerate(pareto_mask):
            if is_pareto:
                ax.scatter(x_vals[i], y_vals[i], c="red", s=180, zorder=4,
                           edgecolors="darkred", linewidth=1.5, marker="*")

        # Annotations
        for i, name in enumerate(names):
            ax.annotate(name, (x_vals[i], y_vals[i]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, fontweight="bold" if pareto_mask[i] else "normal")

        ax.set_xlabel("Average Latency (seconds) →", fontsize=12)
        ax.set_ylabel("Average Faithfulness →", fontsize=12)
        ax.set_title("Quality vs. Resource Trade-off (Pareto Front)", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Arrow indicating "ideal" direction
        ax.annotate("", xy=(min(x_vals) * 0.85, max(y_vals) * 1.02),
                     xytext=(max(x_vals) * 0.95, min(y_vals) * 0.95),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
        ax.text(min(x_vals) * 0.85, max(y_vals) * 1.04, "← Ideal",
                fontsize=9, color="gray", style="italic")

        plt.tight_layout()
        path = os.path.join(self.figures_dir, "pareto_front.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")

    @staticmethod
    def _compute_pareto_mask(x: np.ndarray, y: np.ndarray,
                              minimize_x: bool = True, maximize_y: bool = True) -> list:
        """Compute which points are Pareto-optimal."""
        n = len(x)
        is_pareto = [True] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x_better = (x[j] <= x[i]) if minimize_x else (x[j] >= x[i])
                y_better = (y[j] >= y[i]) if maximize_y else (y[j] <= y[i])
                x_strict = (x[j] < x[i]) if minimize_x else (x[j] > x[i])
                y_strict = (y[j] > y[i]) if maximize_y else (y[j] < y[i])

                if x_better and y_better and (x_strict or y_strict):
                    is_pareto[i] = False
                    break
        return is_pareto

    # ──────────────────────────────────────────
    #  Routing Distribution
    # ──────────────────────────────────────────

    def plot_routing_distribution(self):
        """Pie chart: how the adaptive router distributed queries across bundles."""
        adaptive_exps = [n for n in self.experiments if "Adaptive" in n or "Oracle" in n]

        for exp_name in adaptive_exps:
            df = self.experiments[exp_name]
            if "bundle" not in df.columns:
                continue

            counts = df["bundle"].value_counts()
            colors = {"direct": "#4CAF50", "light": "#2196F3",
                      "medium": "#FF9800", "heavy": "#F44336"}
            pie_colors = [colors.get(b, "#999") for b in counts.index]

            fig, ax = plt.subplots(figsize=(8, 7))
            wedges, texts, autotexts = ax.pie(
                counts, labels=counts.index, autopct="%1.1f%%",
                colors=pie_colors, startangle=90,
                textprops={"fontsize": 11}
            )
            ax.set_title(f"Bundle Distribution — {exp_name}", fontsize=13)
            plt.tight_layout()

            path = os.path.join(self.figures_dir, f"routing_{exp_name.lower()}.png")
            plt.savefig(path)
            plt.close()
            logger.info(f"Saved: {path}")

    # ──────────────────────────────────────────
    #  Latency Breakdown
    # ──────────────────────────────────────────

    def plot_latency_breakdown(self):
        """Stacked bar chart: component-level latency per experiment."""
        if not self.experiments:
            return

        timing_cols = ["classifier_time", "retrieval_time", "rerank_time",
                       "decompose_time", "generation_time"]
        timing_labels = ["Classifier", "Retrieval", "Reranking",
                         "Decomposition", "Generation"]

        data = {}
        for exp_name, df in self.experiments.items():
            row = []
            for col in timing_cols:
                if col in df.columns:
                    row.append(df[col].mean())
                else:
                    row.append(0.0)
            data[exp_name] = row

        if not data:
            return

        exp_names = list(data.keys())
        values = np.array([data[n] for n in exp_names])

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(exp_names))
        bottom = np.zeros(len(exp_names))
        colors = ["#8BC34A", "#03A9F4", "#FF9800", "#9C27B0", "#F44336"]

        for i, (label, color) in enumerate(zip(timing_labels, colors)):
            bars = ax.bar(x, values[:, i], bottom=bottom, label=label,
                          color=color, edgecolor="white", linewidth=0.5)
            bottom += values[:, i]

        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=30, ha="right")
        ax.set_ylabel("Average Time (seconds)")
        ax.set_title("Latency Breakdown by Component")
        ax.legend(loc="upper left")
        plt.tight_layout()

        path = os.path.join(self.figures_dir, "latency_breakdown.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")

    # ──────────────────────────────────────────
    #  Statistical Tests
    # ──────────────────────────────────────────

    def run_statistical_tests(self) -> Optional[str]:
        """
        Paired t-tests, paired Cohen's d, and Holm-adjusted p-values comparing
        P1-Adaptive vs. B1-Fixed-Heavy.
        Returns the report text and saves to file.
        """
        adaptive = self.experiments.get("P1-Adaptive")
        heavy = self.experiments.get("B1-Fixed-Heavy")

        if adaptive is None or heavy is None:
            logger.warning("Need both P1-Adaptive and B1-Fixed-Heavy for statistical tests.")
            return None

        lines = []
        lines.append("=" * 60)
        lines.append("STATISTICAL ANALYSIS: P1-Adaptive vs. B1-Fixed-Heavy")
        lines.append("=" * 60)

        # Test each metric
        test_cols = ["faithfulness", "answer_relevancy", "context_precision",
                     "context_recall", "wall_time_seconds", "cpu_time_seconds",
                     "memory_peak_mb"]

        results = []

        for col in test_cols:
            if col not in adaptive.columns or col not in heavy.columns:
                continue

            # Preserve row pairing by position: the experiment files are verified
            # to contain the same 500 query IDs in the same order.
            a = adaptive[col]
            h = heavy[col]
            valid = a.notna() & h.notna()
            a_vals = a.loc[valid].to_numpy()
            h_vals = h.loc[valid].to_numpy()

            n = min(len(a_vals), len(h_vals))
            if n < 5:
                continue

            a_vals = a_vals[:n]
            h_vals = h_vals[:n]

            t_stat, p_val = stats.ttest_rel(a_vals, h_vals)
            diff = a_vals - h_vals
            diff_std = diff.std(ddof=1)
            paired_d = diff.mean() / (diff_std + 1e-10)

            # Bootstrap 95% CI for the paired mean difference.
            boot_diffs = []
            rng = np.random.RandomState(42)
            for _ in range(1000):
                idx = rng.randint(0, n, n)
                boot_diffs.append(diff[idx].mean())
            ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

            pct_change = ((a_vals.mean() - h_vals.mean()) /
                          (h_vals.mean() + 1e-10)) * 100

            results.append({
                "col": col,
                "a_mean": a_vals.mean(),
                "a_std": a_vals.std(ddof=1),
                "h_mean": h_vals.mean(),
                "h_std": h_vals.std(ddof=1),
                "pct_change": pct_change,
                "t_stat": t_stat,
                "p_val": p_val,
                "paired_d": paired_d,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        # Holm-Bonferroni correction controls family-wise error rate across
        # the seven paired comparisons in this report.
        ordered = sorted(enumerate(results), key=lambda x: x[1]["p_val"])
        adjusted = [1.0] * len(results)
        running_max = 0.0
        m = len(results)
        for rank, (idx, result) in enumerate(ordered):
            adj = min((m - rank) * result["p_val"], 1.0)
            running_max = max(running_max, adj)
            adjusted[idx] = running_max

        for result, p_adj in zip(results, adjusted):
            lines.append(f"\n--- {result['col']} ---")
            lines.append(f"  Adaptive mean:    {result['a_mean']:.4f} ± {result['a_std']:.4f}")
            lines.append(f"  Fixed-Heavy mean: {result['h_mean']:.4f} ± {result['h_std']:.4f}")
            lines.append(f"  Change:           {result['pct_change']:+.2f}%")
            lines.append(f"  Paired t-test:    t={result['t_stat']:.4f}, p={result['p_val']:.4e}")
            lines.append(f"  Holm-adjusted p:  {p_adj:.4e}")
            lines.append(f"  Paired Cohen's d: {result['paired_d']:.4f}")
            lines.append(f"  95% CI (diff):    [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")

            sig = "YES" if p_adj < 0.05 else "NO"
            lines.append(f"  Significant after Holm correction (α=0.05): {sig}")

        report = "\n".join(lines)

        path = os.path.join(self.results_dir, "statistical_report.txt")
        with open(path, "w") as f:
            f.write(report)
        logger.info(f"Statistical report saved to {path}")

        print(report)
        return report

    # ──────────────────────────────────────────
    #  LaTeX Tables
    # ──────────────────────────────────────────

    def generate_latex_tables(self):
        """Generate LaTeX-formatted comparison tables for the paper."""
        if not self.experiments:
            return

        # Quality table
        quality_rows = []
        for exp_name, df in self.experiments.items():
            row = {"Experiment": exp_name}
            for col in ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]:
                if col in df.columns:
                    row[col] = f"{df[col].mean():.4f}"
            quality_rows.append(row)

        quality_df = pd.DataFrame(quality_rows).set_index("Experiment")
        quality_latex = quality_df.to_latex(escape=False)

        # Resource table
        resource_rows = []
        for exp_name, df in self.experiments.items():
            row = {"Experiment": exp_name}
            for col in ["wall_time_seconds", "cpu_time_seconds", "memory_peak_mb"]:
                if col in df.columns:
                    row[col] = f"{df[col].mean():.3f}"
            resource_rows.append(row)

        resource_df = pd.DataFrame(resource_rows).set_index("Experiment")
        resource_latex = resource_df.to_latex(escape=False)

        path = os.path.join(self.results_dir, "latex_tables.tex")
        with open(path, "w") as f:
            f.write("% Quality Metrics Table\n")
            f.write(quality_latex)
            f.write("\n\n% Resource Metrics Table\n")
            f.write(resource_latex)

        logger.info(f"LaTeX tables saved to {path}")

    # ──────────────────────────────────────────
    #  Full Report
    # ──────────────────────────────────────────

    def generate_full_report(self):
        """Run all analyses and generate complete report."""
        logger.info("=" * 60)
        logger.info("Generating Full Analysis Report")
        logger.info("=" * 60)

        if not self.experiments:
            logger.warning("No experiment results found. Run experiments first.")
            return

        self.plot_quality_comparison()
        self.plot_resource_comparison()
        self.plot_pareto_front()
        self.plot_routing_distribution()
        self.plot_latency_breakdown()
        self.run_statistical_tests()
        self.generate_latex_tables()

        logger.info(f"Analysis complete. All outputs in: {self.results_dir}")
