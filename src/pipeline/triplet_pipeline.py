# -*- coding: utf-8 -*-
"""
Triplet 数据构建流水线

覆盖任务：
1. 读取/生成 query 列表（支持 txt/json/jsonl）
2. 用 bge large 对每个 query 检索 top10 chunk
3. 用 reranker 对 top10 每个 chunk 打分
4. 按 reranker 分数重排并筛样本
5. 从 top10 选择 positive/negative
6. 转为 triplet（每个 query 多条）
7. 过滤并保留带分数 triplet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.embedding.vectorstore import load_vectorstore, search
from src.reranking.reranker import score_query_chunks


def load_queries(query_path: str) -> List[str]:
    """从 txt/json/jsonl 读取 query。"""
    path = Path(query_path)
    if not path.exists():
        raise FileNotFoundError(f"query 文件不存在: {query_path}")

    suffix = path.suffix.lower()
    queries: List[str] = []

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    q = item.strip()
                elif isinstance(item, dict):
                    q = str(item.get("query", "")).strip()
                else:
                    q = ""
                if q:
                    queries.append(q)
    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, str):
                    q = row.strip()
                else:
                    q = str(row.get("query", "")).strip()
                if q:
                    queries.append(q)
    else:
        raise ValueError("仅支持 .txt / .json / .jsonl")

    # 保序去重
    seen = set()
    unique_queries: List[str] = []
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        unique_queries.append(q)
    return unique_queries


def retrieve_topk_chunks(
    query: str,
    vectorstore: Dict[str, Any],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """步骤 2：检索 top_k chunk。"""
    results = search(query, vectorstore, top_k=top_k)
    rows: List[Dict[str, Any]] = []
    for chunk, score in results:
        rows.append(
            {
                "chunk": chunk.get("content", ""),
                "retrieval_score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return rows


def rerank_topk(query: str, topk_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """步骤 3,4：对 top_k chunk 打分并重排。"""
    chunks = [row["chunk"] for row in topk_rows]
    rerank_scores = score_query_chunks(query, chunks)

    scored = []
    for row, rr_score in zip(topk_rows, rerank_scores):
        item = dict(row)
        item["reranker_score"] = float(rr_score)
        scored.append(item)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return scored


def select_pos_neg(
    reranked_rows: Sequence[Dict[str, Any]],
    min_pos_score: float,
    max_neg_score: float,
    min_margin: float,
    max_negatives: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """步骤 5：从 top_k 中选择 positive / negative。"""
    positives = [r for r in reranked_rows if r["reranker_score"] >= min_pos_score]
    negatives = [r for r in reranked_rows if r["reranker_score"] <= max_neg_score]

    if not positives and reranked_rows:
        positives = [reranked_rows[0]]

    if max_negatives > 0 and len(negatives) > max_negatives:
        negatives = negatives[-max_negatives:]

    # margin 过滤：negative 必须明显低于最好 positive
    if positives:
        best_pos = max(p["reranker_score"] for p in positives)
        negatives = [n for n in negatives if (best_pos - n["reranker_score"]) >= min_margin]

    return positives, negatives


def build_triplets_for_query(
    query: str,
    positives: Sequence[Dict[str, Any]],
    negatives: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """步骤 6：一个 query 生成多条 triplet。"""
    triplets: List[Dict[str, Any]] = []
    for pos in positives:
        for neg in negatives:
            triplets.append(
                {
                    "query": query,
                    "positive": pos["chunk"],
                    "negative": neg["chunk"],
                    "positive_score": pos["reranker_score"],
                    "negative_score": neg["reranker_score"],
                    "score_margin": pos["reranker_score"] - neg["reranker_score"],
                    "positive_metadata": pos.get("metadata", {}),
                    "negative_metadata": neg.get("metadata", {}),
                }
            )
    return triplets


def save_jsonl(rows: Sequence[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_triplet_pipeline(
    query_path: str,
    top10_output_path: str,
    triplet_output_path: str,
    chunk_method: str = "semantic",
    model_type: str = "large",
    top_k: int = 10,
    min_pos_score: float = 0.5,
    max_neg_score: float = 0.0,
    min_margin: float = 0.2,
    max_negatives_per_query: int = 3,
) -> Dict[str, int]:
    """
    运行完整 triplet 构建流水线。

    返回统计信息。
    """
    print("[1/7] 加载 query...")
    queries = load_queries(query_path)
    print(f"  - query 数量: {len(queries)}")

    print("[2/7] 加载向量库并检索 top10...")
    vectorstore = load_vectorstore(chunk_method=chunk_method, model_type=model_type)

    top10_rows: List[Dict[str, Any]] = []
    triplets: List[Dict[str, Any]] = []

    for i, query in enumerate(queries, start=1):
        retrieved = retrieve_topk_chunks(query, vectorstore=vectorstore, top_k=top_k)
        if not retrieved:
            continue

        print(f"  - [{i}/{len(queries)}] rerank query")
        reranked = rerank_topk(query, retrieved)

        top10_rows.append(
            {
                "query": query,
                "top_chunks": reranked,
            }
        )

        positives, negatives = select_pos_neg(
            reranked_rows=reranked,
            min_pos_score=min_pos_score,
            max_neg_score=max_neg_score,
            min_margin=min_margin,
            max_negatives=max_negatives_per_query,
        )
        query_triplets = build_triplets_for_query(query, positives, negatives)
        triplets.extend(query_triplets)

    print("[7/7] 输出结果...")
    save_jsonl(top10_rows, top10_output_path)
    save_jsonl(triplets, triplet_output_path)

    stats = {
        "query_count": len(queries),
        "top10_records": len(top10_rows),
        "triplet_count": len(triplets),
    }
    print(f"完成: {stats}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建带分数 triplet 训练数据")
    parser.add_argument("--query-path", required=True, help="query 文件路径，支持 txt/json/jsonl")
    parser.add_argument(
        "--top10-output",
        default="data/processed/query_top10_scored.jsonl",
        help="每个 query 的 top10 + 打分输出路径",
    )
    parser.add_argument(
        "--triplet-output",
        default="data/processed/triplets_scored.jsonl",
        help="triplet 输出路径",
    )
    parser.add_argument("--chunk-method", default="semantic", choices=["semantic", "sliding_window"])
    parser.add_argument("--model-type", default="large", choices=["large", "small", "student"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-pos-score", type=float, default=0.5)
    parser.add_argument("--max-neg-score", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=0.2)
    parser.add_argument("--max-negatives-per-query", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_triplet_pipeline(
        query_path=args.query_path,
        top10_output_path=args.top10_output,
        triplet_output_path=args.triplet_output,
        chunk_method=args.chunk_method,
        model_type=args.model_type,
        top_k=args.top_k,
        min_pos_score=args.min_pos_score,
        max_neg_score=args.max_neg_score,
        min_margin=args.min_margin,
        max_negatives_per_query=args.max_negatives_per_query,
    )
