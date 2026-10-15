"""
CloudRAG-Adapt Configuration Module.

Centralizes all configuration for the project: paths, model choices,
retrieval bundle specs, experiment parameters, and monitoring settings.
Optimized defaults for M1 MacBook Air (8GB RAM).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path

# Dynamically resolve project root — works regardless of where the script is run from
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


@dataclass
class ProjectConfig:
    """Directory paths for data, results, indexes, and saved models."""
    project_root: str = PROJECT_ROOT
    data_dir: str = field(default="")
    results_dir: str = field(default="")
    index_dir: str = field(default="")
    models_dir: str = field(default="")
    figures_dir: str = field(default="")

    def __post_init__(self):
        self.data_dir = self.data_dir or os.path.join(self.project_root, "data")
        self.results_dir = self.results_dir or os.path.join(self.project_root, "results_final_200")
        self.index_dir = self.index_dir or os.path.join(self.project_root, "index")
        self.models_dir = self.models_dir or os.path.join(self.project_root, "models")
        self.figures_dir = self.figures_dir or os.path.join(self.project_root, "results_final_200", "figures")

    def ensure_dirs(self):
        """Create all project directories if they don't exist."""
        for d in [self.data_dir, self.results_dir, self.index_dir,
                  self.models_dir, self.figures_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32  # Small batches for M1 Air memory


@dataclass
class ChunkingConfig:
    """Document chunking parameters."""
    chunk_size: int = 512    # tokens
    chunk_overlap: int = 64  # tokens
    separator: str = "\n"


@dataclass
class BundleConfig:
    """Configuration for a single retrieval depth bundle."""
    top_k: int = 0
    use_reranker: bool = False
    use_multi_query: bool = False
    rerank_top_n: int = 3


@dataclass
class RetrieverConfig:
    """All retrieval depth bundles."""
    bundles: Dict[str, BundleConfig] = field(default_factory=lambda: {
        "direct": BundleConfig(top_k=0, use_reranker=False, use_multi_query=False, rerank_top_n=0),
        "light": BundleConfig(top_k=3, use_reranker=False, use_multi_query=False, rerank_top_n=0),
        "medium": BundleConfig(top_k=5, use_reranker=True, use_multi_query=False, rerank_top_n=3),
        "heavy": BundleConfig(top_k=10, use_reranker=True, use_multi_query=True, rerank_top_n=5),
    })


@dataclass
class LLMConfig:
    """LLM (Ollama) configuration. Defaults to phi3:mini for M1 Air 8GB."""
    model_name: str = "phi3:mini"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    request_timeout: int = 120  # seconds


@dataclass
class RerankerConfig:
    """Cross-encoder reranker configuration."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass
class ExperimentConfig:
    """Experiment parameters."""
    num_eval_queries: int = 500
    num_repetitions: int = 3
    random_seed: int = 42
    save_interval: int = 25  # Save intermediate results every N queries


@dataclass
class CloudRAGConfig:
    """Top-level configuration aggregating all sub-configs."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a flat dict for components that expect dict-style access."""
        return {
            "project_root": self.project.project_root,
            "data_dir": self.project.data_dir,
            "results_dir": self.project.results_dir,
            "index_dir": self.project.index_dir,
            "models_dir": self.project.models_dir,
            "figures_dir": self.project.figures_dir,
            "embedding_model": self.embedding.model_name,
            "embedding_dim": self.embedding.dimension,
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
            "llm_model": self.llm.model_name,
            "llm_base_url": self.llm.base_url,
            "llm_temperature": self.llm.temperature,
            "llm_max_tokens": self.llm.max_tokens,
            "reranker_model": self.reranker.model_name,
            "num_eval_queries": self.experiment.num_eval_queries,
            "num_repetitions": self.experiment.num_repetitions,
            "random_seed": self.experiment.random_seed,
        }


def get_default_config() -> CloudRAGConfig:
    """Return the default CloudRAGConfig instance."""
    config = CloudRAGConfig()
    config.project.ensure_dirs()
    return config


# Alias for backward compatibility — main.py imports get_config()
def get_config() -> Dict[str, Any]:
    """Return config as a dict (for components expecting dict-style access)."""
    return get_default_config().to_dict()
