"""
RAG Pipeline for CloudRAG-Adapt.

Implements 4 retrieval depth bundles:
  - Direct:  No retrieval, LLM-only generation
  - Light:   Top-3 retrieval, no reranking
  - Medium:  Top-5 retrieval + cross-encoder reranking (top-3)
  - Heavy:   Multi-query decomposition + top-10 retrieval + reranking (top-5)

Each bundle records per-component timing for resource analysis.
"""

import time
import logging
from typing import List, Dict, Any, Optional

import requests

from src.config import get_default_config, CloudRAGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline with 4 retrieval depth bundles.

    Uses Ollama for LLM generation and an optional cross-encoder for reranking.
    All timing is measured per-component for resource analysis.
    """

    def __init__(self, vector_store, config: Optional[CloudRAGConfig] = None):
        """
        Args:
            vector_store: A VectorStore instance (from indexer.py) with a loaded index.
            config: CloudRAGConfig. If None, uses default.
        """
        self.vector_store = vector_store
        self.config = config or get_default_config()
        self._reranker = None

    # ─────────────────────────────────────────
    #  LLM (Ollama) Interface
    # ─────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama's local API for text generation."""
        url = f"{self.config.llm.base_url}/api/generate"
        payload = {
            "model": self.config.llm.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.llm.temperature,
                "num_predict": self.config.llm.max_tokens,
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.config.llm.request_timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running? Start with: ollama serve")
            return "[ERROR: Ollama not running]"
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.config.llm.request_timeout}s")
            return "[ERROR: Ollama timeout]"
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return f"[ERROR: {e}]"

    # ─────────────────────────────────────────
    #  Reranker (Cross-Encoder)
    # ─────────────────────────────────────────

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            model_name = self.config.reranker.model_name
            logger.info(f"Loading reranker: {model_name}")
            self._reranker = CrossEncoder(model_name)
        return self._reranker

    def _rerank(self, query: str, texts: List[str], top_n: int) -> List[str]:
        """Rerank text passages using cross-encoder scores."""
        if not texts or top_n <= 0:
            return texts

        reranker = self._get_reranker()
        pairs = [[query, t] for t in texts]
        scores = reranker.predict(pairs)

        scored = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
        return [text for text, score in scored[:top_n]]

    # ─────────────────────────────────────────
    #  Multi-Query Decomposition
    # ─────────────────────────────────────────

    def _decompose_query(self, query: str) -> List[str]:
        """
        Break a complex query into 2-3 simpler sub-queries using the LLM.
        Falls back to the original query if decomposition fails.
        """
        prompt = (
            "Break the following complex question into 2-3 simpler sub-questions. "
            "Output ONLY the sub-questions, one per line, with no numbering or bullets.\n\n"
            f"Question: {query}\n\n"
            "Sub-questions:"
        )
        response = self._call_llm(prompt)
        sub_queries = [line.strip() for line in response.split("\n")
                       if line.strip() and not line.strip().startswith("[ERROR")]
        return sub_queries[:3] if sub_queries else [query]

    # ─────────────────────────────────────────
    #  Prompt Formatting
    # ─────────────────────────────────────────

    def _format_rag_prompt(self, query: str, contexts: List[str]) -> str:
        """Format the RAG prompt with retrieved contexts."""
        if not contexts:
            return f"Answer the following question concisely.\n\nQuestion: {query}\n\nAnswer:"

        context_block = "\n\n".join(
            f"[Document {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
        )
        return (
            "Use ONLY the following documents to answer the question. "
            "If the answer is not in the documents, say 'I don't have enough information.'\n\n"
            f"{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    # ─────────────────────────────────────────
    #  Retrieval Bundles
    # ─────────────────────────────────────────

    def _run_direct(self, query: str) -> Dict[str, Any]:
        """Bundle: Direct — No retrieval, LLM only."""
        timings = {"classifier_time": 0.0, "retrieval_time": 0.0,
                    "rerank_time": 0.0, "decompose_time": 0.0, "generation_time": 0.0}

        t0 = time.perf_counter()
        prompt = f"Answer the following question concisely.\n\nQuestion: {query}\n\nAnswer:"
        answer = self._call_llm(prompt)
        timings["generation_time"] = time.perf_counter() - t0

        return {"answer": answer, "contexts": [], "timings": timings}

    def _run_light(self, query: str) -> Dict[str, Any]:
        """Bundle: Light — Top-3 retrieval, no reranking."""
        timings = {"classifier_time": 0.0, "retrieval_time": 0.0,
                    "rerank_time": 0.0, "decompose_time": 0.0, "generation_time": 0.0}

        # Retrieve
        t0 = time.perf_counter()
        contexts = self.vector_store.search_texts(query, top_k=3)
        timings["retrieval_time"] = time.perf_counter() - t0

        # Generate
        t0 = time.perf_counter()
        prompt = self._format_rag_prompt(query, contexts)
        answer = self._call_llm(prompt)
        timings["generation_time"] = time.perf_counter() - t0

        return {"answer": answer, "contexts": contexts, "timings": timings}

    def _run_medium(self, query: str) -> Dict[str, Any]:
        """Bundle: Medium — Top-5 retrieval + reranking to top-3."""
        timings = {"classifier_time": 0.0, "retrieval_time": 0.0,
                    "rerank_time": 0.0, "decompose_time": 0.0, "generation_time": 0.0}

        # Retrieve
        t0 = time.perf_counter()
        contexts = self.vector_store.search_texts(query, top_k=5)
        timings["retrieval_time"] = time.perf_counter() - t0

        # Rerank
        t0 = time.perf_counter()
        bundle_cfg = self.config.retriever.bundles["medium"]
        contexts = self._rerank(query, contexts, top_n=bundle_cfg.rerank_top_n)
        timings["rerank_time"] = time.perf_counter() - t0

        # Generate
        t0 = time.perf_counter()
        prompt = self._format_rag_prompt(query, contexts)
        answer = self._call_llm(prompt)
        timings["generation_time"] = time.perf_counter() - t0

        return {"answer": answer, "contexts": contexts, "timings": timings}

    def _run_heavy(self, query: str) -> Dict[str, Any]:
        """Bundle: Heavy — Multi-query decomposition + top-10 retrieval + reranking to top-5."""
        timings = {"classifier_time": 0.0, "retrieval_time": 0.0,
                    "rerank_time": 0.0, "decompose_time": 0.0, "generation_time": 0.0}

        # Decompose
        t0 = time.perf_counter()
        sub_queries = self._decompose_query(query)
        timings["decompose_time"] = time.perf_counter() - t0

        # Retrieve for each sub-query and deduplicate
        t0 = time.perf_counter()
        all_contexts = []
        seen = set()
        for sq in sub_queries:
            docs = self.vector_store.search_texts(sq, top_k=10)
            for doc in docs:
                if doc not in seen:
                    seen.add(doc)
                    all_contexts.append(doc)
        timings["retrieval_time"] = time.perf_counter() - t0

        # Rerank
        t0 = time.perf_counter()
        bundle_cfg = self.config.retriever.bundles["heavy"]
        contexts = self._rerank(query, all_contexts, top_n=bundle_cfg.rerank_top_n)
        timings["rerank_time"] = time.perf_counter() - t0

        # Generate
        t0 = time.perf_counter()
        prompt = self._format_rag_prompt(query, contexts)
        answer = self._call_llm(prompt)
        timings["generation_time"] = time.perf_counter() - t0

        return {"answer": answer, "contexts": contexts, "timings": timings}

    # ─────────────────────────────────────────
    #  Public Interface
    # ─────────────────────────────────────────

    _BUNDLE_METHODS = {
        "direct": "_run_direct",
        "light": "_run_light",
        "medium": "_run_medium",
        "heavy": "_run_heavy",
    }

    def process_query(self, query: str, bundle: str) -> Dict[str, Any]:
        """
        Process a query through the specified retrieval bundle.

        Args:
            query: The user's question.
            bundle: One of 'direct', 'light', 'medium', 'heavy'.

        Returns:
            Dict with keys: 'answer', 'contexts' (List[str]), 'timings' (Dict[str, float])
        """
        method_name = self._BUNDLE_METHODS.get(bundle)
        if method_name is None:
            logger.warning(f"Unknown bundle '{bundle}', defaulting to 'medium'")
            method_name = "_run_medium"

        method = getattr(self, method_name)
        return method(query)

    # Backward-compatible alias
    def generate(self, query: str, bundle: str) -> Dict[str, Any]:
        """Alias for process_query() — backward compatibility."""
        return self.process_query(query, bundle)
