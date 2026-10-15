"""
FAISS Vector Store for CloudRAG-Adapt.

Handles embedding generation, FAISS index building, persistence,
and similarity search. Uses all-MiniLM-L6-v2 for embeddings.
"""

import os
import gc
import json
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

from src.config import get_default_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with sentence-transformer embeddings."""

    def __init__(self, index_dir: Optional[str] = None):
        config = get_default_config()
        self.index_dir = index_dir or config.project.index_dir
        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.metadata_path = os.path.join(self.index_dir, "metadata.json")

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._model = None
        self._config = config

    def _get_model(self):
        """Lazy-load embedding model to save memory."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self._config.embedding.model_name
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
        return self._model

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from a list of document dicts.

        Each document must have a 'text' field. Optional fields:
        'id', 'title' — stored in metadata for retrieval.
        """
        if not documents:
            logger.warning("No documents provided. Index not built.")
            return

        model = self._get_model()
        dimension = self._config.embedding.dimension
        batch_size = self._config.embedding.batch_size

        # Use IndexFlatIP (inner product on normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

        total_batches = (len(documents) - 1) // batch_size + 1
        logger.info(f"Building index from {len(documents)} documents in {total_batches} batches...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]

            # Encode and normalize for cosine similarity
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)

            self.index.add(embeddings)

            for doc in batch:
                self.metadata.append({
                    "id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "text": doc["text"]
                })

            del embeddings
            gc.collect()

            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"  Processed batch {batch_num}/{total_batches} "
                            f"({self.index.ntotal} vectors)")

        logger.info(f"Index built with {self.index.ntotal} vectors.")

    def save(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("Index is not built yet. Call build_index() first.")

        os.makedirs(self.index_dir, exist_ok=True)

        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        logger.info(f"Index ({self.index.ntotal} vectors) saved to {self.index_dir}")

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found at {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors, "
                     f"{len(self.metadata)} metadata entries.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for the top_k most similar documents.

        Returns a list of dicts, each with:
          - id: document ID
          - title: document title
          - text: document text
          - score: cosine similarity score
        """
        if self.index is None:
            raise ValueError("Index not loaded or built. Call load() or build_index() first.")
        if top_k <= 0:
            return []

        model = self._get_model()
        query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            result = self.metadata[idx].copy()
            result["score"] = float(dist)
            results.append(result)

        return results

    def search_texts(self, query: str, top_k: int = 5) -> List[str]:
        """
        Convenience method: search and return only the text content.
        Used by the RAG pipeline where only text strings are needed.
        """
        results = self.search(query, top_k)
        return [r["text"] for r in results]

    @property
    def is_loaded(self) -> bool:
        """Check if the index is loaded/built."""
        return self.index is not None and self.index.ntotal > 0


def build_index_from_corpus(corpus_path: str, index_dir: Optional[str] = None) -> VectorStore:
    """
    Convenience function: read corpus JSONL, build and save FAISS index.
    Returns the populated VectorStore.
    """
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)

    logger.info(f"Read {len(documents)} documents from {corpus_path}")

    store = VectorStore(index_dir=index_dir)
    store.build_index(documents)
    store.save()

    return store
