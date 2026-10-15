# CloudRAG-Adapt

**Query-Complexity-Aware Retrieval Depth Routing for Resource-Efficient Cloud-Deployed RAG Pipelines**

---

## Overview

CloudRAG-Adapt is a research framework that demonstrates how adaptive query routing in Retrieval-Augmented Generation (RAG) pipelines can reduce cloud resource consumption while maintaining answer quality. Instead of applying expensive heavy retrieval to every query, a lightweight complexity classifier routes queries to the appropriate retrieval depth:

| Complexity | Bundle | Retrieval | Reranker | Multi-Query |
|-----------|--------|-----------|----------|-------------|
| SIMPLE | Direct | None | No | No |
| MODERATE | Light | Top-3 | No | No |
| COMPLEX | Medium | Top-5 | Yes (→ top-3) | No |
| MULTI_HOP | Heavy | Top-10 | Yes (→ top-5) | Yes |

## Research Contribution

1. **Primary**: First empirical study linking RAG retrieval depth routing to actual containerized resource consumption (CPU, memory, latency)
2. **Secondary**: Open-source benchmarking framework for quality-resource Pareto analysis
3. **Tertiary**: Practical guidelines for adaptive routing in cloud RAG deployments

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- ~8GB disk space for datasets and models

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Pull LLM model via Ollama
ollama pull phi3:mini      # For 8GB RAM MacBook Air
# OR
ollama pull mistral:7b-instruct-v0.3-q4_0  # For 16GB RAM
```

### Run the Pipeline

```bash
# Step-by-step
python main.py prepare            # Download NQ-Open + HotpotQA, label by complexity
python main.py index              # Build FAISS vector index
python main.py train-classifier   # Train complexity classifier
python main.py smoke-test         # Quick validation (5 queries × 4 bundles)
python main.py run-experiments    # Full experiment suite (all baselines + adaptive)
python main.py analyze            # Generate figures, tables, statistical report

# Or all at once
python main.py run-all

# Quick test with limited queries
python main.py run-experiments --limit 20
```

## Project Structure

```
CloudRAG-Adapt/
├── main.py                    # CLI entry point (8 commands)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image for the app
├── docker-compose.yml         # Full containerized setup
├── monitoring/
│   └── prometheus.yml         # Prometheus scrape config
├── src/
│   ├── __init__.py
│   ├── config.py              # Central configuration
│   ├── data_preparation.py    # Dataset download, corpus extraction, labeling
│   ├── indexer.py             # FAISS vector store (build, save, load, search)
│   ├── classifier.py          # Query complexity classifier (logistic regression)
│   ├── router.py              # Routes queries → bundles (Adaptive, Random, Oracle)
│   ├── rag_pipeline.py        # 4 retrieval bundles (direct/light/medium/heavy)
│   ├── monitor.py             # CPU, memory, latency measurement (psutil)
│   ├── evaluator.py           # RAGAS-style quality metrics (local, no API)
│   ├── experiment_runner.py   # Orchestrates all experiments
│   └── analysis.py            # Figures, Pareto fronts, statistical tests, LaTeX
├── data/                      # Downloaded datasets (auto-created)
├── index/                     # FAISS index files (auto-created)
├── models/                    # Trained classifier (auto-created)
└── results/                   # Experiment CSVs + figures (auto-created)
```

## Experiments

| ID | Name | Description |
|----|------|-------------|
| B1 | Fixed-Heavy | All queries → heavy retrieval (upper bound quality) |
| B2 | Fixed-Medium | All queries → medium retrieval |
| B3 | Fixed-Light | All queries → light retrieval |
| B4 | Direct | All queries → no retrieval (lower bound resources) |
| P1 | **Adaptive** | Classifier-based routing (proposed method) |
| P2 | Oracle | Ground-truth complexity routing (upper bound) |
| A1 | Random | Random bundle assignment (ablation) |

## Metrics

**Quality (RAGAS-style)**:
- Answer Relevancy, Context Precision, Context Recall, Faithfulness

**Resource**:
- CPU time (seconds), Memory peak (MB), Wall time (seconds)

**Analysis**:
- Quality-Resource Pareto front
- Component latency breakdown
- Paired t-tests with bootstrap 95% CI and Cohen's d

## Docker Deployment (for Paper)

```bash
# Start containerized setup with monitoring
docker-compose up -d

# Pull the LLM model inside the Ollama container
docker exec cloudrag-ollama ollama pull phi3:mini

# Run experiments inside the app container
docker exec cloudrag-app python main.py run-all
```

## Hardware Requirements

| Component | Minimum (8GB RAM) | Recommended (16GB RAM) |
|-----------|-------------------|----------------------|
| LLM | phi3:mini (~2.5GB) | mistral:7b-q4 (~5GB) |
| Embedding | all-MiniLM-L6-v2 (~80MB) | Same |
| Reranker | ms-marco-MiniLM-L6-v2 (~80MB) | Same |
| FAISS Index | ~200MB (5K docs) | Same |

## Target Venues

- IEEE International Conference on Cloud Computing (IEEE CLOUD)
- ACM/IEEE Utility and Cloud Computing (UCC)
- Springer Journal of Cloud Computing (JoCCC)
- MDPI AI / IEEE Access (backup)

## Citation

```bibtex
@article{cloudrag-adapt-2026,
  title={CloudRAG-Adapt: Query-Complexity-Aware Retrieval Depth Routing
         for Resource-Efficient Cloud-Deployed RAG Pipelines},
  year={2026}
}
```

## License

MIT
