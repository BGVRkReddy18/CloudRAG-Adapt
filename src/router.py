"""
Query Router for CloudRAG-Adapt.

Maps classifier predictions (SIMPLE/MODERATE/COMPLEX/MULTI_HOP)
to retrieval depth bundles (direct/light/medium/heavy).
"""

import time
import random
import logging
from typing import List, Dict, Optional

from src.classifier import QueryComplexityClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class QueryRouter:
    """Routes queries to retrieval bundles based on complexity classification."""

    COMPLEXITY_TO_BUNDLE = {
        "SIMPLE": "direct",
        "MODERATE": "light",
        "COMPLEX": "medium",
        "MULTI_HOP": "heavy",
    }

    def __init__(self, classifier: QueryComplexityClassifier):
        self.classifier = classifier

    def route(self, query: str) -> Dict[str, any]:
        """
        Route a query to a retrieval bundle.

        Returns:
            Dict with 'bundle', 'complexity', 'confidence', 'routing_time_ms'
        """
        t0 = time.perf_counter()
        complexity = self.classifier.predict(query)
        probas = self.classifier.predict_proba(query)
        routing_time = (time.perf_counter() - t0) * 1000  # ms

        bundle = self.COMPLEXITY_TO_BUNDLE.get(complexity, "medium")
        confidence = probas.get(complexity, 0.0)

        return {
            "bundle": bundle,
            "complexity": complexity,
            "confidence": confidence,
            "probabilities": probas,
            "routing_time_ms": round(routing_time, 2),
        }

    def route_simple(self, query: str) -> str:
        """Simple route: return just the bundle name."""
        complexity = self.classifier.predict(query)
        return self.COMPLEXITY_TO_BUNDLE.get(complexity, "medium")

    def route_batch(self, queries: List[str]) -> List[Dict]:
        """Route a batch of queries."""
        return [self.route(q) for q in queries]

    def get_distribution(self, queries: List[str]) -> Dict[str, int]:
        """Get distribution of bundle assignments across queries."""
        dist: Dict[str, int] = {}
        for q in queries:
            bundle = self.route_simple(q)
            dist[bundle] = dist.get(bundle, 0) + 1
        return dist


class RandomRouter:
    """Ablation baseline: routes queries randomly to bundles."""

    BUNDLES = ["direct", "light", "medium", "heavy"]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def route(self, query: str) -> Dict[str, any]:
        bundle = self._rng.choice(self.BUNDLES)
        return {
            "bundle": bundle,
            "complexity": "RANDOM",
            "confidence": 0.25,
            "probabilities": {b: 0.25 for b in self.BUNDLES},
            "routing_time_ms": 0.0,
        }

    def route_simple(self, query: str) -> str:
        return self._rng.choice(self.BUNDLES)


class OracleRouter:
    """Upper-bound baseline: routes using ground-truth complexity labels."""

    COMPLEXITY_TO_BUNDLE = QueryRouter.COMPLEXITY_TO_BUNDLE

    def __init__(self, labels: Dict[str, str]):
        """
        Args:
            labels: Dict mapping query text -> ground-truth complexity label
        """
        self.labels = labels

    def route(self, query: str) -> Dict[str, any]:
        complexity = self.labels.get(query, "MODERATE")
        bundle = self.COMPLEXITY_TO_BUNDLE.get(complexity, "medium")
        return {
            "bundle": bundle,
            "complexity": complexity,
            "confidence": 1.0,
            "probabilities": {complexity: 1.0},
            "routing_time_ms": 0.0,
        }

    def route_simple(self, query: str) -> str:
        complexity = self.labels.get(query, "MODERATE")
        return self.COMPLEXITY_TO_BUNDLE.get(complexity, "medium")
