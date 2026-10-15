"""
Query Complexity Classifier for CloudRAG-Adapt.

Lightweight logistic regression classifier that categorizes queries into
4 complexity levels: SIMPLE, MODERATE, COMPLEX, MULTI_HOP.

Uses hand-crafted features (fast to compute, <10ms per query) so routing
overhead doesn't negate resource savings.
"""

import os
import logging
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class QueryComplexityClassifier:
    """
    Classifies query complexity using hand-crafted features + logistic regression.

    Features extracted per query:
        1. word_count       - Number of words
        2. char_count       - Number of characters
        3. entity_count     - Named entity count (spaCy NER)
        4. has_comparison   - Whether comparison keywords are present
        5. has_reasoning    - Whether reasoning keywords (why, how, explain) are present
        6. has_multi_part   - Whether multi-part indicators (and, both, ?) are present
        7. question_type    - Encoded question word (who=0, what=1, ..., other=7)
        8. num_clauses      - Approximate clause count (commas, semicolons, conjunctions)
        9. avg_word_length  - Average word length
        10. has_relative_clause - Presence of "that/which/who" relative clause patterns
    """

    FEATURE_NAMES = [
        "word_count", "char_count", "entity_count", "has_comparison",
        "has_reasoning", "has_multi_part", "question_type", "num_clauses",
        "avg_word_length", "has_relative_clause"
    ]

    COMPLEXITY_LEVELS = ["SIMPLE", "MODERATE", "COMPLEX", "MULTI_HOP"]

    Q_TYPE_MAP = {
        "who": 0, "what": 1, "when": 2, "where": 3,
        "which": 4, "how": 5, "why": 6
    }

    def __init__(self):
        self._nlp = None  # Lazy-loaded spaCy model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self._is_trained = False

    def _get_nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except OSError:
                logger.info("Downloading spaCy en_core_web_sm model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"],
                               check=True, capture_output=True)
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        return self._nlp

    def extract_features(self, query: str) -> np.ndarray:
        """Extract feature vector from a query string."""
        nlp = self._get_nlp()
        doc = nlp(query)

        q_lower = query.lower()
        words = query.split()
        word_count = len(words)

        # Basic length features
        char_count = len(query)
        avg_word_length = char_count / max(word_count, 1)

        # NER count
        entity_count = len(doc.ents)

        # Keyword features
        comparison_kws = {"compare", "difference", "vs", "versus", "between",
                          "similar", "contrast", "unlike", "differ"}
        reasoning_kws = {"why", "how", "explain", "reason", "cause", "because",
                         "effect", "impact", "consequence"}
        multi_part_kws = {"and", "also", "both", "additionally", "furthermore"}

        has_comparison = float(any(w in q_lower for w in comparison_kws))
        has_reasoning = float(any(w in q_lower for w in reasoning_kws))
        has_multi_part = float(
            any(w in q_lower.split() for w in multi_part_kws) or query.count("?") > 1
        )

        # Question type encoding
        question_type = 7.0  # default = "other"
        for token in doc:
            q_type = self.Q_TYPE_MAP.get(token.text.lower())
            if q_type is not None:
                question_type = float(q_type)
                break

        # Clause count approximation
        num_clauses = float(
            query.count(",") + query.count(";") +
            q_lower.split().count("and") + q_lower.split().count("or")
        )

        # Relative clause detection
        has_relative = float(bool(re.search(
            r"\b(that|which|who|whom|whose)\b.*\b(is|was|are|were|has|had|did)\b",
            q_lower
        )))

        return np.array([
            float(word_count), float(char_count), float(entity_count),
            has_comparison, has_reasoning, has_multi_part,
            question_type, num_clauses, avg_word_length, has_relative
        ])

    def _extract_batch(self, queries: List[str]) -> np.ndarray:
        """Extract features for a batch of queries."""
        return np.array([self.extract_features(q) for q in queries])

    def train(self, queries: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Train the classifier on labeled queries.

        Returns dict with training metrics (accuracy, cross-val scores).
        """
        logger.info(f"Training classifier on {len(queries)} queries...")

        X = self._extract_batch(queries)
        y = np.array(labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="accuracy")
        logger.info(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Fit on full data
        self.model.fit(X_scaled, y)
        self._is_trained = True

        train_acc = self.model.score(X_scaled, y)
        logger.info(f"Training accuracy: {train_acc:.3f}")

        return {
            "train_accuracy": float(train_acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "n_samples": len(queries),
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
        }

    def predict(self, query: str) -> str:
        """Predict complexity level for a single query."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call train() or load() first.")

        X = self.extract_features(query).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return str(self.model.predict(X_scaled)[0])

    def predict_proba(self, query: str) -> Dict[str, float]:
        """Get probability distribution over complexity levels."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call train() or load() first.")

        X = self.extract_features(query).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]
        return {cls: round(float(p), 4) for cls, p in zip(self.model.classes_, probs)}

    def save(self, path: str):
        """Save trained model, scaler, and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.FEATURE_NAMES,
            "is_trained": self._is_trained,
        }, path)
        logger.info(f"Classifier saved to {path}")

    def load(self, path: str):
        """Load a trained classifier from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._is_trained = data.get("is_trained", True)
        logger.info(f"Classifier loaded from {path}")
