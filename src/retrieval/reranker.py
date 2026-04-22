# -*- coding: utf-8 -*-
"""
Cross-encoder reranker using bge-reranker-v2-gemma for scoring query-chunk pairs.
"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config

_reranker_model = None
_reranker_tokenizer = None


def load_reranker_model():
    """Load bge-reranker-v2-gemma model."""
    global _reranker_model, _reranker_tokenizer

    if _reranker_model is not None:
        return _reranker_model, _reranker_tokenizer

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = load_config()
        model_name = config.get('embedding', {}).get('model_teacher_name', 'BAAI/bge-reranker-v2-gemma')

        local_model_path = Path(PROJECT_ROOT) / "models" / model_name.replace("/", "--")

        print(f"Loading reranker model: {local_model_path}")

        _reranker_tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(str(local_model_path))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _reranker_model = _reranker_model.to(device)
        _reranker_model.eval()

        print(f"Reranker model loaded (device: {device})")

        return _reranker_model, _reranker_tokenizer

    except Exception as e:
        print(f"Failed to load reranker model: {e}")
        raise


def score_single_pair(query: str, chunk: str) -> float:
    """Score a single query-chunk pair."""
    model, tokenizer = load_reranker_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        [query], [chunk],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = torch.sigmoid(logits).squeeze().item()

    return float(score)


def score_batch(query: str, chunks: List[str], batch_size: int = 32) -> List[float]:
    """Score query against multiple chunks in batches."""
    model, tokenizer = load_reranker_model()
    device = next(model.parameters()).device

    scores = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_queries = [query] * len(batch_chunks)

        inputs = tokenizer(
            batch_queries, batch_chunks,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_scores = torch.sigmoid(logits).squeeze()

            if batch_scores.dim() == 0:
                batch_scores = batch_scores.unsqueeze(0)

            scores.extend(batch_scores.cpu().numpy().tolist())

    return scores


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    return_scores: bool = True
) -> List[Dict[str, Any]]:
    """Rerank retrieved chunks by relevance score."""
    if not chunks:
        return []

    chunk_texts = []
    for chunk in chunks:
        text = chunk.get('text') or chunk.get('content', '')
        chunk_texts.append(text)

    scores = score_batch(query, chunk_texts)

    reranked_chunks = []
    for chunk, score in zip(chunks, scores):
        chunk_copy = chunk.copy()
        if return_scores:
            chunk_copy['reranker_score'] = round(float(score), 4)
        reranked_chunks.append((chunk_copy, score))

    reranked_chunks.sort(key=lambda x: x[1], reverse=True)

    result = [chunk for chunk, score in reranked_chunks]

    if top_k is not None:
        result = result[:top_k]

    return result


def get_reranker():
    """Get reranker interface."""
    return {
        'score_single': score_single_pair,
        'score_batch': score_batch,
        'rerank': rerank_chunks,
        'model': load_reranker_model
    }
