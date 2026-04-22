# -*- coding: utf-8 -*-
"""
筛选流水线模块（调整版）

本模块筛选第4部分（蒸馏+代码）生成的{query, chunks}对数据，用于训练数据准备。
筛选标准：相关性（优先reranker分数）、质量（长度+信息密度）、多样性（去重+来源平衡）、一致性（模型一致性）、平衡性（hard negative）。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from collections import defaultdict

# 添加 src 目录到导入路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from embedding.embedder import get_embedder


def load_query_chunk_pairs(data_path: str) -> List[Dict]:
    """
    加载{query, chunks}对数据。

    参数
    ----------
    data_path : str
        数据文件路径（JSONL格式，每行一个对：{"query": str, "chunks": [str], "metadata": {"reranker_score": float, "source": str, ...}}）。

    返回
    -------
    List[Dict]
        数据对列表。
    """
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


def compute_similarity(query: str, chunks: List[str], embedder) -> List[float]:
    """
    计算query与chunks的相似度（备用，当reranker分数不可用时）。

    参数
    ----------
    query : str
        查询字符串。
    chunks : List[str]
        Chunks列表。
    embedder : dict
        嵌入模型。

    返回
    -------
    List[float]
        相似度分数列表。
    """
    model = embedder['load']()
    query_emb = model.encode([query])[0]
    chunk_embs = model.encode(chunks)
    similarities = [
        np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
        for chunk_emb in chunk_embs
    ]
    return similarities


def filter_pairs(
    pairs: List[Dict],
    embedder,
    relevance_threshold: float = 0.7,
    reranker_threshold: float = 0.8,
    min_chunk_len: int = 50,
    max_chunk_len: int = 1000,
    dedup_threshold: float = 0.95,
    info_density_threshold: float = 0.1,
    max_single_doc_ratio: float = 0.3,
) -> List[Dict]:
    """
    筛选{query, chunks}对（调整版）。

    参数
    ----------
    pairs : List[Dict]
        原始数据对。
    embedder : dict
        嵌入模型。
    relevance_threshold : float
        备用相关性阈值（bi-encoder相似度）。
    reranker_threshold : float
        Reranker分数阈值。
    min_chunk_len : int
        最小chunk长度。
    max_chunk_len : int
        最大chunk长度。
    dedup_threshold : float
        去重阈值。
    info_density_threshold : float
        信息密度阈值（税务关键词密度）。
    max_single_doc_ratio : float
        单文档chunks最大比例。

    返回
    -------
    List[Dict]
        筛选后的数据对。
    """
    filtered = []
    seen_chunks = set()
    doc_count = defaultdict(int)

    for pair in pairs:
        query = pair['query']
        chunks = pair['chunks']
        metadata = pair.get('metadata', {})

        # 一致性：验证reranker分数和来源存在
        if 'reranker_score' not in metadata or 'source' not in metadata:
            continue

        # 相关性：优先用reranker分数
        reranker_score = metadata['reranker_score']
        if reranker_score < reranker_threshold:
            continue

        # 质量：长度 + 信息密度（税务关键词密度）
        tax_keywords = ['增值税', '所得税', '税率', '纳税', '申报', '发票', '扣除', '征收']
        valid_chunks = []
        for chunk in chunks:
            if min_chunk_len <= len(chunk) <= max_chunk_len:
                word_count = len(chunk.split())
                if word_count > 0:
                    density = sum(1 for kw in tax_keywords if kw in chunk) / word_count
                    if density >= info_density_threshold:
                        valid_chunks.append(chunk)

        if not valid_chunks:
            continue

        # 多样性：去重 + 来源平衡
        deduped_chunks = []
        current_total = len(filtered)
        for chunk in valid_chunks:
            source = metadata['source']
            if current_total > 0 and doc_count[source] / current_total > max_single_doc_ratio:
                continue

            is_dup = any(
                np.dot(embedder['load']().encode([chunk])[0], embedder['load']().encode([seen])[0])
                / (
                    np.linalg.norm(embedder['load']().encode([chunk])[0])
                    * np.linalg.norm(embedder['load']().encode([seen])[0])
                )
                > dedup_threshold
                for seen in seen_chunks
            )

            if not is_dup:
                deduped_chunks.append(chunk)
                seen_chunks.add(chunk)
                doc_count[source] += 1

        if deduped_chunks:
            # 平衡性：保留hard negative（中等相关性chunks作为负样本）
            hard_negatives = [
                chunk for chunk in valid_chunks
                if chunk not in deduped_chunks and compute_similarity(query, [chunk], embedder)[0] > 0.5
            ]
            final_chunks = deduped_chunks + hard_negatives[:1]

            filtered.append(
                {
                    'query': query,
                    'chunks': final_chunks,
                    'metadata': metadata,
                    'reranker_score': reranker_score,
                }
            )

    return filtered


def save_filtered_pairs(filtered_pairs: List[Dict], output_path: str):
    """
    保存筛选后的数据。

    参数
    ----------
    filtered_pairs : List[Dict]
        筛选后的数据对。
    output_path : str
        输出文件路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in filtered_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def run_filter_pipeline(
    input_path: str,
    output_path: str,
    model_type: str = "large",
    reranker_threshold: float = 0.8,
):
    """
    运行筛选流水线（调整版）。

    参数
    ----------
    input_path : str
        输入数据路径。
    output_path : str
        输出数据路径。
    model_type : str
        嵌入模型类型。
    reranker_threshold : float
        Reranker分数阈值。
    """
    print("加载数据...")
    pairs = load_query_chunk_pairs(input_path)
    print(f"加载 {len(pairs)} 个数据对")

    print("加载嵌入模型...")
    embedder = get_embedder(model_type)

    print("筛选数据...")
    filtered = filter_pairs(pairs, embedder, reranker_threshold=reranker_threshold)
    print(f"筛选后剩余 {len(filtered)} 个数据对")

    print("保存筛选数据...")
    save_filtered_pairs(filtered, output_path)
    print(f"筛选数据已保存到 {output_path}")


if __name__ == "__main__":
    input_path = "data/processed/query_chunk_pairs.jsonl"
    output_path = "data/processed/filtered_query_chunk_pairs.jsonl"
    run_filter_pipeline(input_path, output_path)