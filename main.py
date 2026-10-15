#!/usr/bin/env python3
"""
CloudRAG-Adapt CLI — Main Entry Point.

Commands:
    prepare           Download datasets and create labeled evaluation subsets
    index             Build FAISS vector index from corpus
    train-classifier  Train the query complexity classifier
    smoke-test        Run a quick smoke test (5 queries per bundle)
    run-experiments   Run the full experiment suite
    analyze           Generate figures, tables, and statistical report
    run-all           Run everything end-to-end
    system-info       Print system hardware information
"""

import argparse
import json
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cloudrag-adapt")


def cmd_prepare(args):
    """Download datasets and create labeled evaluation subsets."""
    from src.data_preparation import main as prepare_main
    prepare_main()


def cmd_index(args):
    """Build FAISS vector index from the extracted corpus."""
    from src.config import get_default_config
    from src.indexer import build_index_from_corpus

    config = get_default_config()


    corpus_path = os.path.join(config.project.data_dir, "corpus.jsonl")

    if not os.path.exists(corpus_path):
        logger.error(f"Corpus not found at {corpus_path}. Run 'prepare' first.")
        sys.exit(1)

    build_index_from_corpus(corpus_path, index_dir=config.project.index_dir)
    logger.info("Index built successfully.")


def cmd_train_classifier(args):
    """Train the query complexity classifier on labeled data."""
    from src.config import get_default_config
    from src.classifier import QueryComplexityClassifier

    config = get_default_config()
    labeled_path = os.path.join(config.project.data_dir, "labeled_queries.jsonl")

    if not os.path.exists(labeled_path):
        logger.error(f"Labeled data not found at {labeled_path}. Run 'prepare' first.")
        sys.exit(1)

    # Load labeled queries
    queries, labels = [], []
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["question"])
            labels.append(item["complexity"])

    logger.info(f"Loaded {len(queries)} labeled queries.")

    # Train
    classifier = QueryComplexityClassifier()
    metrics = classifier.train(queries, labels)

    # Save
    model_path = os.path.join(config.project.models_dir, "classifier.joblib")
    classifier.save(model_path)

    print(f"\nTraining Results:")
    print(f"  Training accuracy: {metrics['train_accuracy']:.3f}")
    print(f"  5-fold CV:         {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    print(f"  Class distribution: {metrics['class_distribution']}")
    print(f"  Model saved to:    {model_path}")


def cmd_smoke_test(args):
    """Run a quick smoke test: 5 queries through each bundle."""
    from src.config import get_default_config
    from src.indexer import VectorStore
    from src.rag_pipeline import RAGPipeline
    from src.evaluator import RAGEvaluator
    from src.monitor import ResourceMonitor

    config = get_default_config()

    # Load index
    vs = VectorStore()
    try:
        vs.load()
    except FileNotFoundError:
        logger.error("FAISS index not found. Run 'index' first.")
        sys.exit(1)

    # Initialize pipeline
    pipeline = RAGPipeline(vector_store=vs, config=config)
    evaluator = RAGEvaluator()
    monitor = ResourceMonitor()

    # Load a few eval queries
    eval_path = os.path.join(config.project.data_dir, "eval_subset.jsonl")
    if not os.path.exists(eval_path):
        logger.error(f"Eval subset not found at {eval_path}. Run 'prepare' first.")
        sys.exit(1)

    queries = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            queries.append(json.loads(line))

    print(f"\n{'='*60}")
    print("SMOKE TEST — 5 queries × 4 bundles")
    print(f"{'='*60}")

    for bundle in ["direct", "light", "medium", "heavy"]:
        print(f"\n── Bundle: {bundle.upper()} ──")
        for q in queries:
            question = q["question"]
            ground_truth = q.get("ground_truth", "")

            snapshot = monitor.take_snapshot()
            result = pipeline.process_query(question, bundle)
            metrics = monitor.measure_since(snapshot)

            answer = result["answer"][:100]
            quality = evaluator.evaluate_single(
                question, result["answer"], result["contexts"], ground_truth
            )

            print(f"  Q: {question[:60]}...")
            print(f"  A: {answer}...")
            print(f"  Faithfulness: {quality['faithfulness']:.3f} | "
                  f"Relevancy: {quality['answer_relevancy']:.3f} | "
                  f"Time: {metrics['wall_time_seconds']:.2f}s")
            print()

    print("Smoke test complete!")


def cmd_run_experiments(args):
    """Run the full experiment suite."""
    from src.config import get_default_config
    from src.indexer import VectorStore
    from src.rag_pipeline import RAGPipeline
    from src.evaluator import RAGEvaluator
    from src.classifier import QueryComplexityClassifier
    from src.router import QueryRouter, RandomRouter, OracleRouter
    from src.experiment_runner import ExperimentRunner

    config = get_default_config()

    if args.output_dir:
        config.project.results_dir = os.path.abspath(args.output_dir)
        os.makedirs(config.project.results_dir, exist_ok=True)

    # Load index
    vs = VectorStore()
    vs.load()

    # Initialize components
    pipeline = RAGPipeline(vector_store=vs, config=config)
    evaluator = RAGEvaluator()

    # Load classifier
    model_path = os.path.join(config.project.models_dir, "classifier.joblib")
    classifier = QueryComplexityClassifier()
    classifier.load(model_path)

    # Create routers
    adaptive_router = QueryRouter(classifier)
    random_router = RandomRouter(seed=config.experiment.random_seed)

    # Load eval queries
    eval_path = os.path.join(config.project.data_dir, "eval_subset.jsonl")
    queries = []
    oracle_labels = {}
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item)
            oracle_labels[item["question"]] = item.get("complexity", "MODERATE")

    oracle_router = OracleRouter(oracle_labels)

    # Limit queries if --limit flag is set
    if args.limit and args.limit > 0:
        queries = queries[:args.limit]
        logger.info(f"Limited to {len(queries)} queries (--limit {args.limit})")

    logger.info(f"Running experiments with {len(queries)} queries...")

    # Run
    runner = ExperimentRunner(pipeline, evaluator, config)
    results = runner.run_all_experiments(
        queries=queries,
        adaptive_router=adaptive_router,
        oracle_router=oracle_router,
        random_router=random_router,
    )

    logger.info(f"All experiments complete. Results in: {config.project.results_dir}")


def cmd_analyze(args):
    """Generate all analysis figures, tables, and statistical report."""
    from src.config import get_default_config
    from src.analysis import ResultsAnalyzer

    config = get_default_config()
    analyzer = ResultsAnalyzer(config.project.results_dir)
    analyzer.generate_full_report()


def cmd_run_all(args):
    """Run the complete pipeline end-to-end."""
    print("="*60)
    print("CloudRAG-Adapt — Full Pipeline")
    print("="*60)

    print("\n[1/5] Preparing datasets...")
    cmd_prepare(args)

    print("\n[2/5] Building FAISS index...")
    cmd_index(args)

    print("\n[3/5] Training classifier...")
    cmd_train_classifier(args)

    print("\n[4/5] Running experiments...")
    cmd_run_experiments(args)

    print("\n[5/5] Generating analysis...")
    cmd_analyze(args)

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


def cmd_system_info(args):
    """Print system hardware information for the paper."""
    from src.monitor import ResourceMonitor
    monitor = ResourceMonitor()
    info = monitor.get_system_info()

    print("\nSystem Information (for paper Section 7.3):")
    print("-" * 40)
    for k, v in info.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="CloudRAG-Adapt: Query-Complexity-Aware Retrieval Depth Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py prepare              # Download & label datasets
  python main.py index                # Build FAISS index
  python main.py train-classifier     # Train complexity classifier
  python main.py smoke-test           # Quick test (5 queries)
  python main.py run-experiments      # Run full experiment suite
  python main.py run-experiments --limit 20   # Run with 20 queries only
  python main.py analyze              # Generate figures & stats
  python main.py run-all              # Run everything
  python main.py system-info          # Print hardware info
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("prepare", help="Download and prepare datasets")
    subparsers.add_parser("index", help="Build FAISS index from corpus")
    subparsers.add_parser("train-classifier", help="Train the complexity classifier")
    subparsers.add_parser("smoke-test", help="Run quick smoke test (5 queries × 4 bundles)")

    exp_parser = subparsers.add_parser("run-experiments", help="Run full experiment suite")
    exp_parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of queries (0 = no limit)")
    exp_parser.add_argument("--output-dir", type=str, default="",
                        help="Directory for experiment results")

    subparsers.add_parser("analyze", help="Generate analysis report")

    all_parser = subparsers.add_parser("run-all", help="Run everything end-to-end")
    all_parser.add_argument("--limit", type=int, default=0,
                            help="Limit number of queries for experiments")

    subparsers.add_parser("system-info", help="Print system hardware info")

    args = parser.parse_args()

    commands = {
        "prepare": cmd_prepare,
        "index": cmd_index,
        "train-classifier": cmd_train_classifier,
        "smoke-test": cmd_smoke_test,
        "run-experiments": cmd_run_experiments,
        "analyze": cmd_analyze,
        "run-all": cmd_run_all,
        "system-info": cmd_system_info,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
