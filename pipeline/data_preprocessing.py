# -*- coding: utf-8 -*-
"""
数据预处理工具函数 - 用于 query/chunk 清洗、验证、去重
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==================== 文本清洗 ====================

def clean_text(text: str) -> str:
    """清洗文本：去除多余空白、特殊字符等。"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[　\t]+', ' ', text)
    return text


# ==================== Query/Chunk 预处理 ====================

def preprocess_queries_and_chunks(
    raw_queries: List[str],
    raw_chunks: List[Dict[str, Any]],
    embedder_model: Any = None,
    clean_text_: bool = True,
    min_query_length: int = 5,
    check_perplexity: bool = False,
    verbose: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    """
    预处理 queries 和 chunks。

    返回
    -------
    Tuple[List[str], List[Dict], Dict[str, int]]
        (清洗后的 queries, chunks, 统计信息)
    """
    stats = {"total": len(raw_queries), "removed_short": 0, "removed_empty": 0}

    cleaned_queries = []
    for q in raw_queries:
        q = clean_text(q) if clean_text_ else q.strip()
        if not q:
            stats["removed_empty"] += 1
            continue
        if len(q) < min_query_length:
            stats["removed_short"] += 1
            continue
        cleaned_queries.append(q)

    return cleaned_queries, raw_chunks, stats


# ==================== Chunk 质量验证 ====================

TAX_KEYWORDS = [
    '增值税', '所得税', '税率', '纳税', '申报', '发票', '扣除', '征收',
    '税款', '退税', '计税', '税务', '减免', '优惠', '缴纳', '应税',
]


def validate_chunk_quality(
    content: str,
    min_length: int = 50,
    max_length: int = 1000,
    min_density: float = 0.0,
    check_containment: bool = False,
) -> Tuple[bool, str]:
    """
    验证 chunk 质量。

    返回
    -------
    Tuple[bool, str]
        (是否有效, 原因)
    """
    if not content or not content.strip():
        return False, "empty"

    if len(content) < min_length:
        return False, f"too_short:{len(content)}<{min_length}"

    if len(content) > max_length:
        return False, f"too_long:{len(content)}>{max_length}"

    if min_density > 0:
        words = content.split()
        if len(words) > 0:
            density = sum(1 for kw in TAX_KEYWORDS if kw in content) / len(words)
            if density < min_density:
                return False, f"low_density:{density:.4f}<{min_density}"

    return True, "ok"


# ==================== Chunk 去重 ====================

def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    embedder_model: Any,
    similarity_threshold: float = 0.95,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    基于 embedding 相似度去重。

    返回
    -------
    Tuple[List[Dict], int]
        (去重后的 chunks, 去除数量)
    """
    if not chunks:
        return [], 0

    contents = [c.get("content", "") for c in chunks]
    if not any(contents):
        return chunks, 0

    embeddings = embedder_model.encode(contents, normalize_embeddings=True)

    kept = []
    seen_indices = []
    removed = 0

    for i, emb in enumerate(embeddings):
        is_dup = False
        for j in seen_indices:
            sim = float(np.dot(emb, embeddings[j]))
            if sim >= similarity_threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(chunks[i])
            seen_indices.append(i)

    return kept, removed
