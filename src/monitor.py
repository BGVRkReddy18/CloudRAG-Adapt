"""
Resource Monitor for CloudRAG-Adapt.

Measures per-query CPU time, memory usage, and wall-clock time using psutil.
Designed for single-process monitoring on M1 MacBook Air.
For containerized experiments, Docker/cAdvisor metrics supplement this.
"""

import os
import csv
import time
import json
import platform
import logging
from typing import Callable, Any, Tuple, Dict, List, Optional

import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Measures CPU time, memory, and wall time for individual operations.

    Usage:
        monitor = ResourceMonitor()
        snapshot = monitor.take_snapshot()
        # ... do work ...
        metrics = monitor.measure_since(snapshot)
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # Prime the cpu_percent measurement
        self.process.cpu_percent(interval=None)

    def take_snapshot(self) -> Dict[str, float]:
        """Take a snapshot of current resource state. Call before the work."""
        cpu_times = self.process.cpu_times()
        mem_info = self.process.memory_info()
        return {
            "cpu_user": cpu_times.user,
            "cpu_system": cpu_times.system,
            "memory_rss": mem_info.rss,
            "wall_time": time.perf_counter(),
        }

    def measure_since(self, snapshot: Dict[str, float]) -> Dict[str, float]:
        """
        Measure resource consumption since a snapshot.

        Returns:
            cpu_time_seconds: Total user+system CPU time consumed
            memory_peak_mb: Current RSS memory in MB
            memory_delta_mb: Change in RSS since snapshot
            wall_time_seconds: Wall-clock time elapsed
            cpu_percent: CPU utilization percentage
        """
        now_wall = time.perf_counter()
        cpu_times = self.process.cpu_times()
        mem_info = self.process.memory_info()
        cpu_pct = self.process.cpu_percent(interval=None)

        cpu_used = (cpu_times.user - snapshot["cpu_user"]) + \
                   (cpu_times.system - snapshot["cpu_system"])

        return {
            "cpu_time_seconds": max(0.0, cpu_used),
            "memory_peak_mb": round(mem_info.rss / (1024 * 1024), 2),
            "memory_delta_mb": round((mem_info.rss - snapshot["memory_rss"]) / (1024 * 1024), 2),
            "wall_time_seconds": round(now_wall - snapshot["wall_time"], 4),
            "cpu_percent": round(cpu_pct, 1),
        }

    def measure_call(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Convenience: measure a function call and return (result, metrics)."""
        snapshot = self.take_snapshot()
        result = func(*args, **kwargs)
        metrics = self.measure_since(snapshot)
        return result, metrics

    def get_system_info(self) -> Dict[str, Any]:
        """Get system hardware info for the paper's experimental setup section."""
        vm = psutil.virtual_memory()
        return {
            "platform": platform.platform(),
            "processor": platform.processor() or "Apple M1",
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_ram_gb": round(vm.total / (1024 ** 3), 2),
            "available_ram_gb": round(vm.available / (1024 ** 3), 2),
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] if platform.system() == "Darwin" else platform.release(),
        }


class PerQueryLogger:
    """
    Logs per-query results (answer, quality metrics, resource metrics, timings)
    to a CSV file for later analysis.
    """

    COLUMNS = [
        "query_id", "query", "ground_truth", "bundle", "complexity",
        "answer",
        # Quality metrics
        "answer_relevancy", "context_precision", "context_recall", "faithfulness",
        # Resource metrics
        "cpu_time_seconds", "memory_peak_mb", "memory_delta_mb",
        "wall_time_seconds", "cpu_percent",
        # Component timings
        "classifier_time", "retrieval_time", "rerank_time",
        "decompose_time", "generation_time",
        # Metadata
        "num_contexts", "error"
    ]

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.records: List[Dict[str, Any]] = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Start each experiment CSV fresh.
        with open(output_path, "w", newline="", encoding="utf-8") as f:
          writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
          writer.writeheader()

    def log(self, record: Dict[str, Any]):
        """Buffer a single query result."""
        self.records.append(record)

    def flush(self):
        """Write buffered records to CSV."""
        if not self.records:
            return

        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            writer.writerows(self.records)

        count = len(self.records)
        self.records = []
        logger.debug(f"Flushed {count} records to {self.output_path}")

    def load(self):
        """Load results from CSV into a pandas DataFrame."""
        import pandas as pd
        if os.path.exists(self.output_path):
            return pd.read_csv(self.output_path)
        return pd.DataFrame(columns=self.COLUMNS)
