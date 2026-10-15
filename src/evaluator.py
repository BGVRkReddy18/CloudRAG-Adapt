"""
RAG Evaluator for CloudRAG-Adapt.

Implements simplified RAGAS-style metrics using local embeddings
(no external LLM API calls). Metrics computed:

  1. Answer Relevancy   - Cosine similarity between question and answer embeddings
  2. Context Precision  - Fraction of retrieved contexts relevant to ground truth
  3. Context Recall     - Fraction of ground truth keywords found in contexts
  4. Faithfulness       - Fraction of answer claims supported by retrieved contexts

All metrics are 0-1, higher is better.
"""

import re
import logging
from typing import List, Dict, Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Stopwords for keyword extraction
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
    "are", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "what", "which", "who", "whom",
})


class RAGEvaluator:
    """
    Evaluates RAG pipeline outputs using simplified RAGAS-style metrics.
    All computation is local — no API calls needed.
    """

    def __init__(self):
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy-load sentence-transformer for answer relevancy."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model for evaluation...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """Extract meaningful keywords from text."""
        if not isinstance(text, str) or not text.strip():
            return set()
        words = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 2}

    def compute_answer_relevancy(self, query: str, answer: str) -> float:
        """Cosine similarity between query and answer embeddings."""
        if not query or not answer:
            return 0.0

        try:
            model = self._get_embedding_model()
            embeddings = model.encode([query, answer], convert_to_numpy=True)
            # Cosine similarity
            q_norm = embeddings[0] / (np.linalg.norm(embeddings[0]) + 1e-10)
            a_norm = embeddings[1] / (np.linalg.norm(embeddings[1]) + 1e-10)
            similarity = float(np.dot(q_norm, a_norm))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"Answer relevancy computation failed: {e}")
            return 0.0

    def compute_context_precision(self, contexts: List[str], ground_truth: str) -> float:
        """
        Average precision of retrieved contexts relative to ground truth.
        Each context is scored by keyword overlap with ground truth.
        """
        if not contexts or not ground_truth:
            return 0.0

        gt_keywords = self._extract_keywords(ground_truth)
        if not gt_keywords:
            return 0.0

        precision_scores = []
        for ctx in contexts:
            ctx_keywords = self._extract_keywords(ctx)
            overlap = len(ctx_keywords & gt_keywords)
            precision_scores.append(overlap / len(gt_keywords))

        return float(np.mean(precision_scores))

    def compute_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Fraction of ground truth keywords found in the combined contexts.
        """
        if not ground_truth:
            return 0.0

        gt_keywords = self._extract_keywords(ground_truth)
        if not gt_keywords:
            return 0.0

        combined_ctx = " ".join(contexts) if contexts else ""
        ctx_keywords = self._extract_keywords(combined_ctx)

        overlap = len(gt_keywords & ctx_keywords)
        return float(overlap / len(gt_keywords))

    def compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Fraction of answer keywords that appear in the retrieved contexts.
        Measures whether the answer is grounded in the retrieved evidence.
        """
        if not answer:
            return 0.0

        answer_keywords = self._extract_keywords(answer)
        if not answer_keywords:
            return 0.0

        combined_ctx = " ".join(contexts) if contexts else ""
        ctx_keywords = self._extract_keywords(combined_ctx)

        overlap = len(answer_keywords & ctx_keywords)
        return float(overlap / len(answer_keywords))

    def evaluate_single(self, query: str, answer: str,
                        contexts: List[str], ground_truth: str) -> Dict[str, float]:
        """
        Compute all 4 RAGAS-style metrics for a single query.

        Args:
            query: The input question
            answer: The generated answer
            contexts: List of retrieved context strings
            ground_truth: The reference answer

        Returns:
            Dict with keys: answer_relevancy, context_precision,
                            context_recall, faithfulness
        """
        return {
            "answer_relevancy": self.compute_answer_relevancy(query, answer),
            "context_precision": self.compute_context_precision(contexts, ground_truth),
            "context_recall": self.compute_context_recall(contexts, ground_truth),
            "faithfulness": self.compute_faithfulness(answer, contexts),
        }

    def evaluate_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Compute metrics for a batch of records.
        Each record must have: query, answer, contexts, ground_truth.
        """
        results = []
        for rec in records:
            metrics = self.evaluate_single(
                query=rec.get("query", ""),
                answer=rec.get("answer", ""),
                contexts=rec.get("contexts", []),
                ground_truth=rec.get("ground_truth", ""),
            )
            results.append(metrics)
        return results
