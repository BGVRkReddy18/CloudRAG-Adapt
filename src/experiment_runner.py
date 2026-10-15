"""
Experiment Runner for CloudRAG-Adapt.

Orchestrates all experiments from the research plan:
  - B1-B4: Fixed-depth baselines (heavy, medium, light, direct)
  - P1: Adaptive routing (CloudRAG-Adapt proposed method)
  - P2: Oracle routing (upper bound)
  - A1: Random routing (ablation)

Collects both quality metrics (RAGAS-style) and resource metrics (CPU, memory, time)
for every query, saving results to CSV for analysis.
"""

import os
import gc
import json
import time
import random
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from src.config import get_default_config, CloudRAGConfig
from src.rag_pipeline import RAGPipeline
from src.monitor import ResourceMonitor, PerQueryLogger
from src.evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrates experiments for CloudRAG-Adapt.

    For each query:
      1. Take resource snapshot
      2. Route query → bundle (or use fixed bundle for baselines)
      3. Run through RAG pipeline
      4. Measure resource consumption
      5. Compute quality metrics
      6. Log everything to CSV
    """

    def __init__(self, rag_pipeline: RAGPipeline,
                 evaluator: RAGEvaluator,
                 config: Optional[CloudRAGConfig] = None):
        self.pipeline = rag_pipeline
        self.evaluator = evaluator
        self.config = config or get_default_config()
        self.monitor = ResourceMonitor()
        self.results_dir = self.config.project.results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def _process_single_query(self, query_dict: Dict, bundle: str,
                               routing_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process one query through the RAG pipeline and collect all metrics.

        Args:
            query_dict: Must have 'id', 'question', 'ground_truth'
            bundle: Retrieval bundle to use
            routing_info: Optional routing metadata (from router)

        Returns:
            Dict with all fields matching PerQueryLogger.COLUMNS
        """
        query = query_dict.get("question", "")
        ground_truth = query_dict.get("ground_truth", "")
        query_id = query_dict.get("id", "")
        complexity = query_dict.get("complexity", "")

        # Take resource snapshot BEFORE processing
        snapshot = self.monitor.take_snapshot()

        # Run through RAG pipeline
        try:
            result = self.pipeline.process_query(query, bundle)
            error = None
        except Exception as e:
            logger.error(f"Pipeline error for query {query_id}: {e}")
            result = {"answer": "", "contexts": [], "timings": {}}
            error = str(e)

        # Measure resources AFTER processing
        resource_metrics = self.monitor.measure_since(snapshot)

        # Extract pipeline timings
        timings = result.get("timings", {})
        answer = result.get("answer", "")
        contexts = result.get("contexts", [])

        # Compute quality metrics
        try:
            quality = self.evaluator.evaluate_single(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            )
        except Exception as e:
            logger.warning(f"Evaluation error for query {query_id}: {e}")
            quality = {"answer_relevancy": 0.0, "context_precision": 0.0,
                       "context_recall": 0.0, "faithfulness": 0.0}

        # Assemble record
        record = {
            "query_id": query_id,
            "query": query,
            "ground_truth": ground_truth,
            "bundle": bundle,
            "complexity": routing_info.get("complexity", complexity) if routing_info else complexity,
            "answer": answer[:500],  # Truncate long answers for CSV
            # Quality
            "answer_relevancy": quality.get("answer_relevancy", 0.0),
            "context_precision": quality.get("context_precision", 0.0),
            "context_recall": quality.get("context_recall", 0.0),
            "faithfulness": quality.get("faithfulness", 0.0),
            # Resources
            "cpu_time_seconds": resource_metrics.get("cpu_time_seconds", 0.0),
            "memory_peak_mb": resource_metrics.get("memory_peak_mb", 0.0),
            "memory_delta_mb": resource_metrics.get("memory_delta_mb", 0.0),
            "wall_time_seconds": resource_metrics.get("wall_time_seconds", 0.0),
            "cpu_percent": resource_metrics.get("cpu_percent", 0.0),
            # Component timings
            "classifier_time": routing_info.get("routing_time_ms", 0.0) / 1000.0 if routing_info else 0.0,
            "retrieval_time": timings.get("retrieval_time", 0.0),
            "rerank_time": timings.get("rerank_time", 0.0),
            "decompose_time": timings.get("decompose_time", 0.0),
            "generation_time": timings.get("generation_time", 0.0),
            # Metadata
            "num_contexts": len(contexts),
            "error": error,
        }

        return record

    # ─────────────────────────────────────────
    #  Experiment Types
    # ─────────────────────────────────────────

    def run_fixed_baseline(self, queries: List[Dict], bundle: str,
                            experiment_name: str) -> pd.DataFrame:
        """Run all queries through a single fixed bundle (baselines B1-B4)."""
        logger.info(f"{'='*60}")
        logger.info(f"Running: {experiment_name} (fixed bundle={bundle})")
        logger.info(f"Queries: {len(queries)}")
        logger.info(f"{'='*60}")

        csv_path = os.path.join(self.results_dir, f"{experiment_name}.csv")
        per_query_logger = PerQueryLogger(csv_path)

        for i, q in enumerate(tqdm(queries, desc=experiment_name)):
            record = self._process_single_query(q, bundle)
            per_query_logger.log(record)

            # Periodic save
            if (i + 1) % self.config.experiment.save_interval == 0:
                per_query_logger.flush()
                logger.info(f"  [{experiment_name}] Saved checkpoint at query {i+1}")

        per_query_logger.flush()
        df = per_query_logger.load()
        self._print_summary(df, experiment_name)
        return df

    def run_adaptive(self, queries: List[Dict], router,
                      experiment_name: str) -> pd.DataFrame:
        """Run queries through the adaptive router (P1, P2, or A1)."""
        logger.info(f"{'='*60}")
        logger.info(f"Running: {experiment_name} (adaptive routing)")
        logger.info(f"Queries: {len(queries)}")
        logger.info(f"{'='*60}")

        csv_path = os.path.join(self.results_dir, f"{experiment_name}.csv")
        per_query_logger = PerQueryLogger(csv_path)

        for i, q in enumerate(tqdm(queries, desc=experiment_name)):
            query_text = q.get("question", "")

            # Route
            routing_info = router.route(query_text)
            bundle = routing_info["bundle"]

            record = self._process_single_query(q, bundle, routing_info=routing_info)
            per_query_logger.log(record)

            if (i + 1) % self.config.experiment.save_interval == 0:
                per_query_logger.flush()
                logger.info(f"  [{experiment_name}] Saved checkpoint at query {i+1}")

        per_query_logger.flush()
        df = per_query_logger.load()
        self._print_summary(df, experiment_name)
        return df

    # ─────────────────────────────────────────
    #  Run All Experiments
    # ─────────────────────────────────────────

    def run_all_experiments(self, queries: List[Dict],
                            adaptive_router=None,
                            oracle_router=None,
                            random_router=None) -> Dict[str, pd.DataFrame]:
        """
        Run the complete experiment suite from the research plan.

        Experiments:
          B1-Fixed-Heavy, B2-Fixed-Medium, B3-Fixed-Light, B4-Direct
          P1-Adaptive (requires adaptive_router)
          P2-Oracle (requires oracle_router)
          A1-Random (requires random_router)
        """
        all_results = {}

        # Baselines
        for name, bundle in [("B1-Fixed-Heavy", "heavy"),
                              ("B2-Fixed-Medium", "medium"),
                              ("B3-Fixed-Light", "light"),
                              ("B4-Direct", "direct")]:
            gc.collect()
            all_results[name] = self.run_fixed_baseline(queries, bundle, name)

        # Proposed method
        if adaptive_router:
            gc.collect()
            all_results["P1-Adaptive"] = self.run_adaptive(queries, adaptive_router, "P1-Adaptive")

        # Oracle upper bound
        if oracle_router:
            gc.collect()
            all_results["P2-Oracle"] = self.run_adaptive(queries, oracle_router, "P2-Oracle")

        # Random ablation
        if random_router:
            gc.collect()
            all_results["A1-Random"] = self.run_adaptive(queries, random_router, "A1-Random")

        # Save combined summary
        self._save_combined_summary(all_results)

        return all_results

    # ─────────────────────────────────────────
    #  Reporting
    # ─────────────────────────────────────────

    def _print_summary(self, df: pd.DataFrame, experiment_name: str):
        """Print summary statistics for an experiment."""
        print(f"\n{'─'*50}")
        print(f"  Summary: {experiment_name}")
        print(f"{'─'*50}")

        n = len(df)
        errors = df["error"].notna().sum() if "error" in df.columns else 0
        print(f"  Total queries: {n}")
        print(f"  Errors: {errors} ({errors/max(n,1)*100:.1f}%)")

        # Quality
        quality_cols = ["answer_relevancy", "context_precision", "context_recall", "faithfulness"]
        for col in quality_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")

        # Resources
        resource_cols = ["wall_time_seconds", "cpu_time_seconds", "memory_peak_mb"]
        for col in resource_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")

        # Bundle distribution (for adaptive experiments)
        if "bundle" in df.columns:
            dist = df["bundle"].value_counts()
            print(f"  Bundle distribution:")
            for bundle, count in dist.items():
                print(f"    {bundle}: {count} ({count/n*100:.1f}%)")

        print(f"{'─'*50}\n")

    def _save_combined_summary(self, all_results: Dict[str, pd.DataFrame]):
        """Save a combined summary CSV comparing all experiments."""
        rows = []
        for exp_name, df in all_results.items():
            row = {"experiment": exp_name, "n_queries": len(df)}

            for col in ["answer_relevancy", "context_precision", "context_recall",
                         "faithfulness", "wall_time_seconds", "cpu_time_seconds",
                         "memory_peak_mb"]:
                if col in df.columns:
                    row[f"{col}_mean"] = df[col].mean()
                    row[f"{col}_std"] = df[col].std()

            if "bundle" in df.columns:
                dist = df["bundle"].value_counts(normalize=True).to_dict()
                row["bundle_distribution"] = json.dumps(dist)

            rows.append(row)

        summary_df = pd.DataFrame(rows)
        summary_path = os.path.join(self.results_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Combined summary saved to {summary_path}")
