"""
Data preparation module for CloudRAG-Adapt.

Downloads NQ-Open and HotpotQA from HuggingFace, extracts a Wikipedia
corpus for indexing, labels queries by complexity, and creates stratified
evaluation subsets.
"""

import os
import json
import random
import logging
import re
from typing import List, Dict, Any, Optional
from collections import Counter

from tqdm import tqdm

from src.config import get_default_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Dataset Download
# ─────────────────────────────────────────────

def download_natural_questions(data_dir: str) -> str:
    """Download NQ-Open validation split. Returns path to saved JSONL."""
    from datasets import load_dataset

    output_path = os.path.join(data_dir, "nq_open.jsonl")
    if os.path.exists(output_path):
        logger.info(f"NQ-Open already exists at {output_path}, skipping download.")
        return output_path

    logger.info("Downloading NQ-Open (validation split) from HuggingFace...")
    dataset = load_dataset("google-research-datasets/nq_open", split="validation")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Saving NQ-Open"):
            record = {
                "question": item["question"],
                "answer": item["answer"],  # list of acceptable answers
                "source": "nq_open"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(dataset)} NQ-Open questions to {output_path}")
    return output_path


def download_hotpotqa(data_dir: str) -> str:
    """Download HotpotQA distractor validation split. Returns path to saved JSONL."""
    from datasets import load_dataset

    output_path = os.path.join(data_dir, "hotpot_qa.jsonl")
    if os.path.exists(output_path):
        logger.info(f"HotpotQA already exists at {output_path}, skipping download.")
        return output_path

    logger.info("Downloading HotpotQA (distractor, validation split) from HuggingFace...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Saving HotpotQA"):
            record = {
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],              # 'comparison' or 'bridge'
                "level": item["level"],             # 'easy', 'medium', 'hard'
                "supporting_facts": {
                    "title": item["supporting_facts"]["title"],
                    "sent_id": item["supporting_facts"]["sent_id"],
                },
                "context": {
                    "title": item["context"]["title"],
                    "sentences": item["context"]["sentences"],
                },
                "source": "hotpot_qa"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(dataset)} HotpotQA questions to {output_path}")
    return output_path


# ─────────────────────────────────────────────
#  Corpus Extraction (for FAISS indexing)
# ─────────────────────────────────────────────

def extract_corpus(data_dir: str, max_documents: int = 5000) -> str:
    """
    Extract Wikipedia passages from HotpotQA contexts as the document corpus.
    Each context paragraph becomes one document. Deduplicates by title.
    Returns path to corpus JSONL file.
    """
    output_path = os.path.join(data_dir, "corpus.jsonl")
    if os.path.exists(output_path):
        logger.info(f"Corpus already exists at {output_path}, skipping extraction.")
        return output_path

    hotpot_path = os.path.join(data_dir, "hotpot_qa.jsonl")
    if not os.path.exists(hotpot_path):
        logger.error("HotpotQA not found. Run download_hotpotqa() first.")
        return output_path

    logger.info("Extracting corpus from HotpotQA contexts...")
    seen_titles = set()
    doc_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        with open(hotpot_path, "r", encoding="utf-8") as in_f:
            for line in tqdm(in_f, desc="Extracting corpus"):
                if doc_count >= max_documents:
                    break

                item = json.loads(line)
                titles = item.get("context", {}).get("title", [])
                sentences_list = item.get("context", {}).get("sentences", [])

                for title, sentences in zip(titles, sentences_list):
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    text = " ".join(sentences).strip()
                    if len(text) < 50:  # skip very short passages
                        continue

                    doc = {
                        "id": f"doc_{doc_count}",
                        "title": title,
                        "text": text
                    }
                    out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    doc_count += 1

                    if doc_count >= max_documents:
                        break

    logger.info(f"Extracted {doc_count} unique documents to {output_path}")
    return output_path


# ─────────────────────────────────────────────
#  Query Complexity Labeling
# ─────────────────────────────────────────────

def _count_question_words(question: str) -> int:
    """Count interrogative words (who, what, when, where, which, how, why)."""
    q_words = {"who", "what", "when", "where", "which", "how", "why"}
    return sum(1 for w in question.lower().split() if w in q_words)


def _has_multi_hop_indicators(question: str) -> bool:
    """Detect indicators of multi-hop reasoning."""
    q = question.lower()
    # Patterns: "who ... that ...", "what ... of the ... that ...",
    # explicit conjunctions linking separate facts
    multi_hop_patterns = [
        r"\b(who|what|which)\b.*\b(that|who|which)\b",   # relative clauses
        r"\band\b.*\b(who|what|where|when)\b",            # compound questions
        r"\bboth\b",
        r"\bneither\b.*\bnor\b",
        r"\bcompare\b|\bcomparison\b|\bdifference between\b",
    ]
    return any(re.search(p, q) for p in multi_hop_patterns)


def label_query_complexity(question: str, answer: str = "",
                           q_type: str = "", level: str = "") -> str:
    """
    Assign complexity label to a query using multiple signals.

    Labels: SIMPLE, MODERATE, COMPLEX, MULTI_HOP

    Signals used:
      - Word count of the question
      - Presence of comparison/reasoning keywords
      - Multi-hop indicators (relative clauses, compound questions)
      - HotpotQA metadata (type=comparison/bridge, level=hard)
      - Number of question words
    """
    q = question.lower().strip()
    words = q.split()
    word_count = len(words)

    # HotpotQA metadata-based labeling (most reliable signal)
    if q_type == "comparison":
        return "COMPLEX"
    if q_type == "bridge" and level == "hard":
        return "MULTI_HOP"
    if q_type == "bridge" and level in ("medium", ""):
        return "COMPLEX"

    # Multi-hop detection
    if _has_multi_hop_indicators(question):
        return "MULTI_HOP"

    # Reasoning / comparison keywords
    reasoning_words = {"why", "how", "explain", "reason", "cause", "effect", "impact"}
    comparison_words = {"compare", "difference", "vs", "versus", "contrast", "similar", "unlike"}

    if any(w in q for w in comparison_words):
        return "COMPLEX"
    if any(w in q for w in reasoning_words) and word_count > 10:
        return "COMPLEX"

    # Length-based heuristics
    if word_count > 15:
        return "COMPLEX"
    if word_count > 8:
        return "MODERATE"

    # Short factoid questions
    return "SIMPLE"


def create_labeled_dataset(data_dir: str) -> str:
    """
    Create a labeled dataset combining NQ-Open and HotpotQA questions
    with complexity labels. Returns path to labeled JSONL.
    """
    output_path = os.path.join(data_dir, "labeled_queries.jsonl")

    nq_path = os.path.join(data_dir, "nq_open.jsonl")
    hotpot_path = os.path.join(data_dir, "hotpot_qa.jsonl")

    records = []

    # Process NQ-Open
    if os.path.exists(nq_path):
        logger.info("Labeling NQ-Open questions...")
        with open(nq_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                answers = item.get("answer", [])
                answer_str = answers[0] if isinstance(answers, list) and answers else str(answers)

                complexity = label_query_complexity(item["question"], answer_str)
                records.append({
                    "id": f"nq_{i}",
                    "question": item["question"],
                    "answer": answer_str,
                    "ground_truth": answer_str,
                    "complexity": complexity,
                    "source": "nq_open"
                })

    # Process HotpotQA
    if os.path.exists(hotpot_path):
        logger.info("Labeling HotpotQA questions...")
        with open(hotpot_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                complexity = label_query_complexity(
                    item["question"],
                    item.get("answer", ""),
                    q_type=item.get("type", ""),
                    level=item.get("level", "")
                )
                records.append({
                    "id": f"hotpot_{i}",
                    "question": item["question"],
                    "answer": item["answer"],
                    "ground_truth": item["answer"],
                    "complexity": complexity,
                    "source": "hotpot_qa"
                })

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print distribution
    dist = Counter(r["complexity"] for r in records)
    logger.info(f"Labeled {len(records)} total queries. Distribution: {dict(dist)}")
    logger.info(f"Saved to {output_path}")

    return output_path


def create_eval_subset(data_dir: str, n: int = 500, seed: int = 42) -> str:
    """
    Create a stratified evaluation subset, balanced across complexity levels.
    Returns path to the eval subset JSONL.
    """
    output_path = os.path.join(data_dir, "eval_subset.jsonl")
    labeled_path = os.path.join(data_dir, "labeled_queries.jsonl")

    if not os.path.exists(labeled_path):
        logger.error("Labeled dataset not found. Run create_labeled_dataset() first.")
        return output_path

    # Load and group by complexity
    by_complexity: Dict[str, List] = {}
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            c = item["complexity"]
            by_complexity.setdefault(c, []).append(item)

    # Stratified sampling
    random.seed(seed)
    per_class = n // len(by_complexity)
    remainder = n % len(by_complexity)

    subset = []
    for i, (complexity, items) in enumerate(sorted(by_complexity.items())):
        count = per_class + (1 if i < remainder else 0)
        count = min(count, len(items))
        subset.extend(random.sample(items, count))
        logger.info(f"  {complexity}: sampled {count} / {len(items)} available")

    random.shuffle(subset)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in subset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Created eval subset with {len(subset)} queries at {output_path}")
    return output_path


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def main():
    """Run the full data preparation pipeline."""
    config = get_default_config()
    config.project.ensure_dirs()
    data_dir = config.project.data_dir

    logger.info("=" * 60)
    logger.info("CloudRAG-Adapt Data Preparation Pipeline")
    logger.info("=" * 60)

    # Step 1: Download datasets
    download_natural_questions(data_dir)
    download_hotpotqa(data_dir)

    # Step 2: Extract corpus for indexing
    extract_corpus(data_dir, max_documents=5000)

    # Step 3: Label queries by complexity
    create_labeled_dataset(data_dir)

    # Step 4: Create evaluation subset
    create_eval_subset(data_dir, n=config.experiment.num_eval_queries, seed=config.experiment.random_seed)

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
