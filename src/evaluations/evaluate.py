# -*- coding: utf-8 -*-
"""
Evaluate retrieval performance with metrics like Recall@k, Hit Rate, MRR, and nDCG.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from retrieval.retriever import retrieve_top_k
from retrieval.reranker import rerank_chunks


def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    """Load test dataset with ground truth."""
    with open(test_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Recall@k."""
    if not relevant_ids:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set)


def calculate_hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Hit Rate@k."""
    if not relevant_ids:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    return 1.0 if len(retrieved_k & relevant_set) > 0 else 0.0


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / i

    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate nDCG@k."""
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_k = retrieved_ids[:k]

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 1)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    test_data: List[Dict[str, Any]],
    model_type: str = "large",
    chunk_method: str = "semantic",
    top_k: int = 10,
    use_reranker: bool = False
) -> Dict[str, Any]:
    """Evaluate retrieval performance."""
    metrics = {
        'recall@1': [],
        'recall@3': [],
        'recall@5': [],
        'recall@10': [],
        'hit_rate@1': [],
        'hit_rate@3': [],
        'hit_rate@5': [],
        'hit_rate@10': [],
        'mrr': [],
        'ndcg@3': [],
        'ndcg@5': [],
        'ndcg@10': []
    }

    for i, item in enumerate(test_data, 1):
        query = item['query']
        relevant_ids = item.get('relevant_chunk_ids', [])

        if not relevant_ids:
            continue

        print(f"[{i}/{len(test_data)}] Evaluating: {query[:50]}...")

        try:
            retrieved_chunks = retrieve_top_k(
                query=query,
                top_k=top_k,
                chunk_method=chunk_method,
                model_type=model_type
            )

            if use_reranker:
                retrieved_chunks = rerank_chunks(
                    query=query,
                    chunks=retrieved_chunks,
                    return_scores=True
                )

            retrieved_ids = [chunk.get('chunk_id', '') for chunk in retrieved_chunks]

            metrics['recall@1'].append(calculate_recall_at_k(retrieved_ids, relevant_ids, 1))
            metrics['recall@3'].append(calculate_recall_at_k(retrieved_ids, relevant_ids, 3))
            metrics['recall@5'].append(calculate_recall_at_k(retrieved_ids, relevant_ids, 5))
            metrics['recall@10'].append(calculate_recall_at_k(retrieved_ids, relevant_ids, 10))

            metrics['hit_rate@1'].append(calculate_hit_rate_at_k(retrieved_ids, relevant_ids, 1))
            metrics['hit_rate@3'].append(calculate_hit_rate_at_k(retrieved_ids, relevant_ids, 3))
            metrics['hit_rate@5'].append(calculate_hit_rate_at_k(retrieved_ids, relevant_ids, 5))
            metrics['hit_rate@10'].append(calculate_hit_rate_at_k(retrieved_ids, relevant_ids, 10))

            metrics['mrr'].append(calculate_mrr(retrieved_ids, relevant_ids))

            metrics['ndcg@3'].append(calculate_ndcg_at_k(retrieved_ids, relevant_ids, 3))
            metrics['ndcg@5'].append(calculate_ndcg_at_k(retrieved_ids, relevant_ids, 5))
            metrics['ndcg@10'].append(calculate_ndcg_at_k(retrieved_ids, relevant_ids, 10))

        except Exception as e:
            print(f"  Evaluation failed: {e}")
            continue

    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}

    return avg_metrics


def run_evaluation(
    test_file: str,
    output_dir: str,
    models_to_test: Optional[List[str]] = None,
    use_reranker: bool = True,
    chunk_method: str = "semantic",
    top_k: int = 10
) -> Dict[str, Any]:
    """Run complete evaluation."""
    print("=" * 60)
    print("Retrieval Evaluation")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 1: Loading test data...")
    test_data = load_test_data(test_file)
    print(f"Loaded {len(test_data)} test queries")

    if models_to_test is None:
        models_to_test = ["mini", "large", "finetuned"]

    results = {}

    print(f"\nStep 2: Evaluating models...")
    print("-" * 60)

    for model_type in models_to_test:
        print(f"\nEvaluating model: {model_type}")
        print(f"Use reranker: {use_reranker}")

        metrics = evaluate_retrieval(
            test_data=test_data,
            model_type=model_type,
            chunk_method=chunk_method,
            top_k=top_k,
            use_reranker=use_reranker
        )

        results[model_type] = metrics

        print(f"\nResults for {model_type}:")
        print(f"  Recall@1: {metrics['recall@1']:.4f}")
        print(f"  Recall@3: {metrics['recall@3']:.4f}")
        print(f"  Recall@5: {metrics['recall@5']:.4f}")
        print(f"  Recall@10: {metrics['recall@10']:.4f}")
        print(f"  Hit Rate@1: {metrics['hit_rate@1']:.4f}")
        print(f"  Hit Rate@5: {metrics['hit_rate@5']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        print(f"  nDCG@5: {metrics['ndcg@5']:.4f}")

    print(f"\nStep 3: Saving results...")
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {results_file}")

    print(f"\nStep 4: Generating report...")
    report_file = output_path / "evaluation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Retrieval Evaluation Report\n\n")
        f.write(f"**Test queries**: {len(test_data)}\n")
        f.write(f"**Top-k**: {top_k}\n")
        f.write(f"**Use reranker**: {use_reranker}\n\n")
        f.write("---\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Model | Recall@1 | Recall@5 | Recall@10 | Hit Rate@5 | MRR | nDCG@5 |\n")
        f.write("|-------|----------|----------|-----------|------------|-----|--------|\n")

        for model_type, metrics in results.items():
            f.write(f"| {model_type} | {metrics['recall@1']:.4f} | {metrics['recall@5']:.4f} | "
                   f"{metrics['recall@10']:.4f} | {metrics['hit_rate@5']:.4f} | "
                   f"{metrics['mrr']:.4f} | {metrics['ndcg@5']:.4f} |\n")

        f.write("\n## Detailed Metrics\n\n")
        for model_type, metrics in results.items():
            f.write(f"### {model_type}\n\n")
            for metric_name, value in metrics.items():
                f.write(f"- **{metric_name}**: {value:.4f}\n")
            f.write("\n")

        f.write("## Analysis\n\n")
        if 'finetuned' in results and 'mini' in results:
            improvement = {}
            for metric in ['recall@5', 'hit_rate@5', 'mrr', 'ndcg@5']:
                baseline = results['mini'][metric]
                finetuned = results['finetuned'][metric]
                if baseline > 0:
                    improvement[metric] = ((finetuned - baseline) / baseline) * 100

            f.write("### Fine-tuned vs Baseline (MiniLM)\n\n")
            for metric, pct in improvement.items():
                f.write(f"- **{metric}**: {pct:+.2f}%\n")

    print(f"Report saved: {report_file}")

    print(f"\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

    return results
