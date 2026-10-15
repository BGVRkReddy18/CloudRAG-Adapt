"""
CloudRAG-Adapt: Query-Complexity-Aware Retrieval Depth Routing
for Resource-Efficient Cloud-Deployed RAG Pipelines.

Modules:
    config             - Central configuration (paths, models, bundles)
    data_preparation   - Dataset download, corpus extraction, complexity labeling
    indexer            - FAISS vector index build/load/search
    classifier         - Query complexity classifier (logistic regression)
    router             - Maps complexity → retrieval bundle
    rag_pipeline       - 4 retrieval bundles (direct/light/medium/heavy)
    monitor            - Per-query CPU, memory, latency measurement
    evaluator          - RAGAS-style quality metrics (simplified, local)
    experiment_runner   - Orchestrates baseline + adaptive experiments
    analysis           - Visualization, Pareto fronts, statistical tests
"""

__version__ = "0.1.0"
