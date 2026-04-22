# -*- coding: utf-8 -*-
"""
Generate triplet training data by retrieving chunks with bge-large and scoring with reranker.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import random

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from retrieval.retriever import retrieve_top_k
from retrieval.reranker import rerank_chunks


def load_queries(query_file: str) -> List[str]:
    """Load queries from file."""
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


def retrieve_and_score(
    query: str,
    top_k: int = 10,
    chunk_method: str = "semantic",
    model_type: str = "large"
) -> List[Dict[str, Any]]:
    """Retrieve and score chunks for a query."""
    print(f"  Retrieving top-{top_k}...")
    retrieved_chunks = retrieve_top_k(
        query=query,
        top_k=top_k,
        chunk_method=chunk_method,
        model_type=model_type
    )

    if not retrieved_chunks:
        print(f"  No chunks retrieved")
        return []

    print(f"  Scoring with reranker...")
    reranked_chunks = rerank_chunks(
        query=query,
        chunks=retrieved_chunks,
        return_scores=True
    )

    return reranked_chunks


def select_positive_negative(
    chunks: List[Dict[str, Any]],
    positive_threshold: float = 0.8,
    negative_min_score: float = 0.5,
    negative_max_score: float = 0.7
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select positive and negative samples from chunks."""
    positives = []
    negatives = []

    for chunk in chunks:
        score = chunk.get('reranker_score', 0)

        if score >= positive_threshold:
            positives.append(chunk)
        elif negative_min_score <= score <= negative_max_score:
            negatives.append(chunk)

    return positives, negatives


def generate_triplets(
    query: str,
    positives: List[Dict[str, Any]],
    negatives: List[Dict[str, Any]],
    max_triplets: int = 4
) -> List[Dict[str, Any]]:
    """Generate triplet combinations."""
    if not positives or not negatives:
        return []

    triplets = []

    for pos in positives:
        for neg in negatives:
            triplet = {
                'query': query,
                'positive': pos.get('text', ''),
                'negative': neg.get('text', ''),
                'positive_score': pos.get('reranker_score', 0),
                'negative_score': neg.get('reranker_score', 0),
                'positive_source': pos.get('source', ''),
                'negative_source': neg.get('source', '')
            }
            triplets.append(triplet)

            if len(triplets) >= max_triplets:
                break

        if len(triplets) >= max_triplets:
            break

    return triplets


def filter_triplets(triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter triplets by quality rules."""
    filtered = []

    for triplet in triplets:
        pos_text = triplet['positive']
        neg_text = triplet['negative']
        pos_score = triplet['positive_score']
        neg_score = triplet['negative_score']

        if pos_text == neg_text:
            continue

        if pos_score <= neg_score:
            continue

        if len(pos_text) < 20 or len(neg_text) < 20:
            continue

        filtered.append(triplet)

    return filtered


def generate_training_data(
    query_file: str,
    output_dir: str,
    top_k: int = 10,
    positive_threshold: float = 0.8,
    negative_min_score: float = 0.5,
    negative_max_score: float = 0.7,
    max_triplets_per_query: int = 4,
    chunk_method: str = "semantic",
    model_type: str = "large",
    max_queries: Optional[int] = None
) -> Dict[str, Any]:
    """Generate complete training dataset."""
    print("=" * 60)
    print("Generating Training Data")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 1: Loading queries...")
    queries = load_queries(query_file)
    if max_queries:
        queries = queries[:max_queries]
    print(f"Loaded {len(queries)} queries")

    stats = {
        'total_queries': len(queries),
        'processed_queries': 0,
        'failed_queries': 0,
        'total_triplets': 0,
        'queries_with_no_positives': 0,
        'queries_with_no_negatives': 0
    }

    all_triplets = []
    all_triplets_with_scores = []

    print(f"\nStep 2: Processing queries...")
    print("-" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query[:50]}...")

        try:
            chunks = retrieve_and_score(
                query=query,
                top_k=top_k,
                chunk_method=chunk_method,
                model_type=model_type
            )

            if not chunks:
                stats['failed_queries'] += 1
                continue

            positives, negatives = select_positive_negative(
                chunks=chunks,
                positive_threshold=positive_threshold,
                negative_min_score=negative_min_score,
                negative_max_score=negative_max_score
            )

            print(f"  Positive: {len(positives)}, Negative: {len(negatives)}")

            if not positives:
                stats['queries_with_no_positives'] += 1
                print(f"  No positive samples")
                continue

            if not negatives:
                stats['queries_with_no_negatives'] += 1
                print(f"  No negative samples")
                continue

            triplets = generate_triplets(
                query=query,
                positives=positives,
                negatives=negatives,
                max_triplets=max_triplets_per_query
            )

            triplets = filter_triplets(triplets)

            print(f"  Generated {len(triplets)} triplets")

            all_triplets_with_scores.extend(triplets)

            for triplet in triplets:
                all_triplets.append({
                    'query': triplet['query'],
                    'positive': triplet['positive'],
                    'negative': triplet['negative']
                })

            stats['processed_queries'] += 1
            stats['total_triplets'] += len(triplets)

        except Exception as e:
            print(f"  Processing failed: {e}")
            stats['failed_queries'] += 1
            continue

    print(f"\nStep 3: Saving training data...")
    print("-" * 60)

    triplets_file = output_path / "triplets.json"
    with open(triplets_file, 'w', encoding='utf-8') as f:
        json.dump(all_triplets, f, ensure_ascii=False, indent=2)
    print(f"Training data saved: {triplets_file}")

    triplets_with_scores_file = output_path / "triplets_with_scores.json"
    with open(triplets_with_scores_file, 'w', encoding='utf-8') as f:
        json.dump(all_triplets_with_scores, f, ensure_ascii=False, indent=2)
    print(f"Data with scores saved: {triplets_with_scores_file}")

    stats_file = output_path / "generation_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved: {stats_file}")

    print(f"\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total queries: {stats['total_queries']}")
    print(f"Processed: {stats['processed_queries']}")
    print(f"Failed: {stats['failed_queries']}")
    print(f"No positives: {stats['queries_with_no_positives']}")
    print(f"No negatives: {stats['queries_with_no_negatives']}")
    print(f"Total triplets: {stats['total_triplets']}")
    print(f"Avg per query: {stats['total_triplets'] / max(stats['processed_queries'], 1):.2f} triplets")
    print("=" * 60)

    return stats
