# -*- coding: utf-8 -*-
"""
统一 Triplet 生成管道 - 支持多版本 chunks

为两个 chunk 版本（semantic 和 sliding）分别执行：
1. 加载/生成 query
2. 检索 top10 chunks
3. Reranker 打分
4. 筛选 positive/negative
5. 生成 triplet
6. 输出 {query, positive_chunk_id, negative_chunk_id} 格式
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional
import sys
# -*- coding: utf-8 -*-
"""
统一 Triplet 生成管道 - 支持多版本 chunks

为两个 chunk 版本（semantic 和 sliding）分别执行：
1. 加载/生成 query
2. 检索 top10 chunks
3. Reranker 打分
4. 筛选 positive/negative
5. 生成 triplet
6. 输出 {query, positive_chunk_id, negative_chunk_id} 格式
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional
import sys

from src.embedding.vectorstore import load_vectorstore, search, build_vectorstore
from src.reranking.reranker import score_query_chunks
from src.embedding.embedder import get_embedder
from src.pipeline.data_preprocessing import (
    preprocess_queries_and_chunks,
    clean_text,
    validate_chunk_quality,
    deduplicate_chunks,
)
import numpy as np


# chunk_version 到 vectorstore chunk_method 的映射
CHUNK_METHOD_MAP = {
    "semantic": "semantic",
    "sliding": "sliding_window",
}


def _get_chunk_method(chunk_version: str) -> str:
    """将 chunk_version 映射为 vectorstore 使用的 chunk_method。"""
    return CHUNK_METHOD_MAP.get(chunk_version, chunk_version)


def ensure_vectorstore_exists(chunk_version: str, chunk_file: str) -> bool:
    """
    检查向量库是否存在，不存在则构建。
    """
    from src.embedding.vectorstore import get_vectorstore_dir, build_vectorstore

    chunk_method = _get_chunk_method(chunk_version)
    vectorstore_dir = get_vectorstore_dir()
    
    # 修复：检查正确的文件组合
    prefix = f"{chunk_method}_large"
    embeddings_file = vectorstore_dir / f"embeddings_{prefix}.npy"
    metadata_file = vectorstore_dir / f"metadata_{prefix}.pkl"
    
    if embeddings_file.exists() and metadata_file.exists():
        print(f"  ✓ 向量库已存在：{vectorstore_dir}")
        print(f"    - {embeddings_file.name}")
        print(f"    - {metadata_file.name}")
        return True

    print(f"  ⚠️  向量库不存在，正在构建...")
    print(f"     需要文件: {embeddings_file.name} 和 {metadata_file.name}")
    print(f"     从 {chunk_file} 构建...")

    try:
        import json as _json
        with open(chunk_file, "r", encoding="utf-8") as _f:
            _raw = _f.read().strip()
            if _raw.startswith("["):
                _chunks = _json.loads(_raw)
            else:
                _f.seek(0)
                _chunks = [_json.loads(_line) for _line in _f if _line.strip()]

        build_vectorstore(
            chunks=_chunks,
            chunk_method=chunk_method,
            model_type="large",
            batch_size=32,
            save_path=str(vectorstore_dir),
        )
        print(f"  ✓ 向量库构建完成")
        return True
    except Exception as e:
        import traceback
        print(f"  ❌ 向量库构建失败: {e}")
        traceback.print_exc()
        return False


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
                "chunk_id": chunk.get("id", ""),
                "content": chunk.get("content", ""),
                "retrieval_score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return rows


def rerank_topk(query: str, topk_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """步骤 3,4：对 top_k chunk 打分并重排。"""
    chunks = [row["content"] for row in topk_rows]
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
    min_pos_score: float = 0.8,
    max_neg_score: float = 0.7,
    min_margin: float = 0.05,
    max_negatives: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """使用 top1 和 top2 作为 positive"""
    
    if not reranked_rows:
        return [], []
    
    # 方案B：用 top1 + top2 作为 positive
    num_positives = 2
    
    # 安全过滤：top1 和 top2 差距不能太小
    if len(reranked_rows) >= 2:
        top_margin = reranked_rows[0]["reranker_score"] - reranked_rows[1]["reranker_score"]
        if top_margin < min_margin:
            print(f"  ⚠️  top1-top2 margin={top_margin:.4f} < {min_margin}，只用 top1 作为 positive")
            num_positives = 1  # 降级为只用 top1
    
    positives = reranked_rows[:num_positives]
    
    # 负样本：从 positive 之后开始取
    hard_negative_start = num_positives
    hard_negative_end = min(hard_negative_start + max_negatives, len(reranked_rows))
    negatives = reranked_rows[hard_negative_start:hard_negative_end]
    
    return positives, negatives

def compute_chunk_similarity(
    content1: str,
    content2: str,
    embedder_model: Any,
) -> float:
    """计算两个 chunk 之间的 embedding 相似度。"""
    try:
        emb1 = embedder_model.encode([content1])[0]
        emb2 = embedder_model.encode([content2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        print(f"  ⚠️ 相似度计算失败: {e}")
        return 0.0


def build_triplets_for_query(
    query: str,
    positives: Sequence[Dict[str, Any]],
    negatives: Sequence[Dict[str, Any]],
    embedder_model: Any = None,
    min_chunk_diff: float = 0.05,
    max_chunk_similarity: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    步骤 6：一个 query 生成多条 triplet。

    参数
    ----------
    query : str
        查询字符串
    positives : Sequence[Dict]
        正样本列表
    negatives : Sequence[Dict]
        负样本列表
    embedder_model : Any
        嵌入模型，用于计算 chunk 相似度
    min_chunk_diff : float
        Positive 和 negative chunk 内容相似度的最小差异
        - 如果 |content_sim(pos) - content_sim(neg)| < min_chunk_diff，跳过该组合
    max_chunk_similarity : float
        Positive 和 negative chunk 之间的最大允许相似度
        - 如果 chunk_similarity(pos, neg) > max_chunk_similarity，跳过（太相似，无法区分）

    返回
    -------
    List[Dict]
        生成的 triplet 列表
    """
    triplets: List[Dict[str, Any]] = []

    for pos in positives:
        for neg in negatives:
            # 基础 triplet 信息
            triplet = {
                "query": query,
                "positive_chunk_id": pos["chunk_id"],
                "negative_chunk_id": neg["chunk_id"],
                "positive_score": pos["reranker_score"],
                "negative_score": neg["reranker_score"],
                "score_margin": pos["reranker_score"] - neg["reranker_score"],
            }

            # 如果提供了嵌入模型，添加 chunk 相似度检查
            if embedder_model is not None:
                pos_content = pos.get("content", "")
                neg_content = neg.get("content", "")

                if pos_content and neg_content:
                    # 计算 positive/negative chunk 之间的相似度
                    chunk_similarity = compute_chunk_similarity(
                        pos_content, neg_content, embedder_model
                    )
                    triplet["chunk_similarity"] = chunk_similarity

                    # 过滤条件：chunks 不能太相似（否则模型难以学习区分）
                    if chunk_similarity > max_chunk_similarity:
                        continue  # 跳过此组合

                    # 计算 query-content 相似度（可选的额外信息）
                    query_pos_sim = compute_chunk_similarity(query, pos_content, embedder_model)
                    query_neg_sim = compute_chunk_similarity(query, neg_content, embedder_model)
                    triplet["query_positive_similarity"] = query_pos_sim
                    triplet["query_negative_similarity"] = query_neg_sim

                    # 过滤：query 与 positive 的相似度应显著大于与 negative 的相似度
                    if (query_pos_sim - query_neg_sim) < min_chunk_diff:
                        continue  # 跳过此组合

            triplets.append(triplet)

    return triplets


def save_jsonl(rows: Sequence[Dict[str, Any]], output_path: str) -> None:
    """保存为 JSONL 或 JSON 格式（根据文件扩展名自动判断）。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith('.json'):
        # 保存为 JSON 数组格式
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(rows), f, ensure_ascii=False, indent=2)
    else:
        # 保存为 JSONL 格式（每行一个 JSON）
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_unified_pipeline(
    query_path: str,
    chunk_version: str,
    triplet_output_path: str,
    chunk_file: Optional[str] = None,
    top_k: int = 10,
    min_pos_score: float = 0.8,
    max_neg_score: float = 0.7,
    min_margin: float = 0.1,
    max_negatives_per_query: int = 1,
    top10_output_path: Optional[str] = None,
    min_chunk_diff: float = 0.05,
    max_chunk_similarity: float = 0.95,
    use_chunk_similarity_filter: bool = True,
    # ===== 新增预处理参数 =====
    enable_preprocessing: bool = True,
    clean_texts: bool = True,
    min_query_length: int = 5,
    min_chunk_length: int = 50,
    max_chunk_length: int = 1000,
    min_info_density: float = 0.1,
    check_chunk_containment: bool = True,
    deduplicate: bool = True,
    dedup_similarity_threshold: float = 0.95,
    check_perplexity: bool = False,
) -> Dict[str, int]:
    """
    运行统一 triplet 构建流水线（支持多版本 chunks）。

    参数
    ----------
    query_path : str
        query 文件路径 (txt/json/jsonl)
    chunk_version : str
        chunk 版本：'semantic' 或 'sliding'
    triplet_output_path : str
        输出 triplet 文件路径
    chunk_file : str, optional
        chunk 数据文件路径（自动查找：data/processed/chunks_{chunk_version}_cleaned.json）
    top_k : int
        检索 top_k
    min_pos_score : float
        Positive reranker score 阈值
    max_neg_score : float
        Negative reranker score 上界
    min_margin : float
        Positive-Negative 最小 margin
    max_negatives_per_query : int
        每个 query 的最大 negative 数量
    top10_output_path : str, optional
        可选：保存 top10 reranked 结果
    min_chunk_diff : float
        Query 与 Positive/Negative chunk 相似度的最小差异
    max_chunk_similarity : float
        Positive 和 Negative chunk 之间的最大允许相似度
    use_chunk_similarity_filter : bool
        是否启用 chunk 相似度过滤

    返回
    -------
    Dict[str, int]
        统计信息
    """
    print("=" * 70)
    print(f"【统一 Triplet 生成管道】- {chunk_version.upper()} 版本")
    print("=" * 70)

    print("\n[1/8] 加载 query...")
    raw_queries = load_queries(query_path)
    print(f"  ✓ 加载 {len(raw_queries)} 条 query")

    # ===== 新增：预处理 Query =====
    if enable_preprocessing:
        print("\n[1.5/8] 预处理 Query...")
        queries, _, preprocess_stats = preprocess_queries_and_chunks(
            raw_queries,
            [],  # 暂时不处理 chunks
            embedder_model=None,
            clean_text_=clean_texts,
            min_query_length=min_query_length,
            check_perplexity=check_perplexity,
            verbose=True,
        )
        print(f"  ✓ 预处理后保留：{len(queries)} 条 query")
    else:
        queries = raw_queries
        print("  ⚠️  跳过预处理")

    if not queries:
        print("❌ 没有有效的 query，无法继续")
        sys.exit(1)

    print("\n[2/8] 加载向量库...")

    # 自动确定 chunk 文件路径
    if chunk_file is None:
        project_root = Path(__file__).parent.parent.parent
        chunk_file = project_root / f"data/processed/chunks_{chunk_version}_cleaned.json"
        if not chunk_file.exists():
            print(f"  ❌ Chunk 文件不存在: {chunk_file}")
            sys.exit(1)

    # 检查或构建向量库
    if not ensure_vectorstore_exists(chunk_version, str(chunk_file)):
        print(f"  ❌ 无法加载或构建向量库")
        sys.exit(1)

    vectorstore = load_vectorstore(chunk_method=_get_chunk_method(chunk_version), model_type="large")
    print(f"  ✓ 向量库加载完成 ({chunk_version})")

    # 加载嵌入模型（用于计算 chunk 相似度）
    embedder_model = None
    if use_chunk_similarity_filter:
        print("\n[加载嵌入模型用于相似度计算...]")
        embedder_config = get_embedder(model_type="large")
        embedder_model = embedder_config["load"]()
        print(f"  ✓ 嵌入模型加载完成")

    top10_rows: List[Dict[str, Any]] = []
    triplets: List[Dict[str, Any]] = []
    stats = {
        "total_queries": len(queries),
        "queries_with_positives": 0,
        "queries_with_negatives": 0,
        "triplet_count": 0,
        "failed_queries": 0,
    }

    print("\n[3-6/7] 逐 query 处理...")
    print("-" * 70)

    for i, query in enumerate(queries, start=1):
        try:
            # 步骤 4：检索 top_k
            retrieved = retrieve_topk_chunks(query, vectorstore=vectorstore, top_k=top_k)
            if not retrieved:
                stats["failed_queries"] += 1
                continue

            # ===== 新增：质量过滤 =====
            if enable_preprocessing:
                filtered_retrieved = []
                for chunk in retrieved:
                    content = chunk.get("content", "")
                    valid, reason = validate_chunk_quality(
                        content,
                        min_length=min_chunk_length,
                        max_length=max_chunk_length,
                        min_density=min_info_density,
                        check_containment=check_chunk_containment,
                    )
                    if valid:
                        filtered_retrieved.append(chunk)

                if not filtered_retrieved:
                    stats["failed_queries"] += 1
                    continue

                retrieved = filtered_retrieved

            # ===== 新增：去重 =====
            if deduplicate and embedder_model is not None:
                retrieved, _ = deduplicate_chunks(
                    retrieved,
                    embedder_model,
                    similarity_threshold=dedup_similarity_threshold,
                )
                if not retrieved:
                    stats["failed_queries"] += 1
                    continue

            # 步骤 5,6：Rerank
            reranked = rerank_topk(query, retrieved)

            top10_rows.append(
                {
                    "query": query,
                    "top_chunks": reranked,
                }
            )

            # 步骤 7：选择 positive/negative
            positives, negatives = select_pos_neg(
                reranked_rows=reranked,
                min_pos_score=min_pos_score,
                max_neg_score=max_neg_score,
                min_margin=min_margin,
                max_negatives=max_negatives_per_query,
            )

            if positives:
                stats["queries_with_positives"] += 1
            if negatives:
                stats["queries_with_negatives"] += 1

            # 步骤 8：生成 triplets
            query_triplets = build_triplets_for_query(
                query,
                positives,
                negatives,
                embedder_model=embedder_model if use_chunk_similarity_filter else None,
                min_chunk_diff=min_chunk_diff,
                max_chunk_similarity=max_chunk_similarity,
            )
            triplets.extend(query_triplets)

            if i % max(1, len(queries) // 10) == 0 or i == len(queries):
                print(f"  [{i:4d}/{len(queries):4d}] ✓ {len(query_triplets):2d} triplets | "
                      f"正数 {len(positives):2d} | 负数 {len(negatives):2d}")

            stats["triplet_count"] += len(query_triplets)

        except Exception as e:
            print(f"  [FAIL] Query {i}: {e}")
            stats["failed_queries"] += 1
            continue

    print("-" * 70)

    # 步骤 8/9：输出结果
    print("\n[8/9] 输出结果...")

    save_jsonl(triplets, triplet_output_path)
    print(f"  ✓ Triplet 已保存: {triplet_output_path}")

    if top10_output_path:
        save_jsonl(top10_rows, top10_output_path)
        print(f"  ✓ Top10 已保存: {top10_output_path}")

    # 输出统计
    print("\n" + "=" * 70)
    print("【统计信息】")
    print("=" * 70)
    print(f"总 Query 数:         {stats['total_queries']}")
    print(f"有 Positive 的:      {stats['queries_with_positives']}")
    print(f"有 Negative 的:      {stats['queries_with_negatives']}")
    print(f"生成 Triplet 总数:   {stats['triplet_count']}")
    print(f"失败的 Query:        {stats['failed_queries']}")
    if stats["queries_with_positives"] > 0:
        avg_triplets = stats["triplet_count"] / stats["queries_with_positives"]
        print(f"平均每 query triplet: {avg_triplets:.2f}")
    print("=" * 70)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一 Triplet 生成管道 - 支持多版本 chunks"
    )
    parser.add_argument(
        "--query-path",
        required=True,
        help="query 文件路径 (txt/json/jsonl)"
    )
    parser.add_argument(
        "--chunk-version",
        required=True,
        choices=["semantic", "sliding"],
        help="Chunk 版本：'semantic' 或 'sliding'"
    )
    parser.add_argument(
        "--triplet-output",
        required=True,
        help="Triplet 输出路径"
    )
    parser.add_argument(
        "--chunk-file",
        default=None,
        help="Chunk 数据文件路径（可选，自动查找 data/processed/chunks_{version}_cleaned.json）"
    )
    parser.add_argument(
        "--top10-output",
        default=None,
        help="可选：Top10 reranked 结果输出路径"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-pos-score", type=float, default=0.8)
    parser.add_argument("--max-neg-score", type=float, default=0.7)
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--max-negatives-per-query", type=int, default=1)

    # chunk 相似度过滤参数（使用 store_false 保持 default=True）
    parser.add_argument(
        "--no-use-chunk-similarity-filter",
        action="store_false",
        dest="use_chunk_similarity_filter",
        default=True,
        help="禁用 chunk 相似度过滤"
    )
    parser.add_argument(
        "--min-chunk-diff",
        type=float,
        default=0.05,
        help="Query-Positive 与 Query-Negative 相似度的最小差异（默认：0.05）"
    )
    parser.add_argument(
        "--max-chunk-similarity",
        type=float,
        default=0.95,
        help="Positive 和 Negative chunk 之间的最大允许相似度（默认：0.95）"
    )

    # ===== 预处理参数 =====
    parser.add_argument("--enable-preprocessing", action="store_true", default=True, help="启用预处理")
    parser.add_argument("--clean-texts", action="store_true", default=True, help="启用文本清洗")
    parser.add_argument("--min-query-length", type=int, default=5)
    parser.add_argument("--min-chunk-length", type=int, default=50)
    parser.add_argument("--max-chunk-length", type=int, default=1000)
    parser.add_argument("--min-info-density", type=float, default=0.1)
    parser.add_argument("--check-chunk-containment", action="store_true", default=True)
    parser.add_argument("--deduplicate", action="store_true", default=True)
    parser.add_argument("--dedup-similarity-threshold", type=float, default=0.95)
    parser.add_argument("--check-perplexity", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # use_chunk_similarity_filter 为 True 表示启用（除非显式传了 --disable-chunk-similarity-filter）
    use_sim_filter = not getattr(args, 'use_chunk_similarity_filter', True)

    run_unified_pipeline(
        query_path=args.query_path,
        chunk_version=args.chunk_version,
        triplet_output_path=args.triplet_output,
        chunk_file=args.chunk_file,
        top10_output_path=args.top10_output,
        top_k=args.top_k,
        min_pos_score=args.min_pos_score,
        max_neg_score=args.max_neg_score,
        min_margin=args.min_margin,
        max_negatives_per_query=args.max_negatives_per_query,
        use_chunk_similarity_filter=use_sim_filter,
        min_chunk_diff=args.min_chunk_diff,
        max_chunk_similarity=args.max_chunk_similarity,
        enable_preprocessing=getattr(args, 'enable_preprocessing', True),
        clean_texts=getattr(args, 'clean_texts', True),
        min_query_length=getattr(args, 'min_query_length', 5),
        min_chunk_length=getattr(args, 'min_chunk_length', 50),
        max_chunk_length=getattr(args, 'max_chunk_length', 1000),
        min_info_density=getattr(args, 'min_info_density', 0.1),
        check_chunk_containment=getattr(args, 'check_chunk_containment', True),
        deduplicate=getattr(args, 'deduplicate', True),
        dedup_similarity_threshold=getattr(args, 'dedup_similarity_threshold', 0.95),
        check_perplexity=getattr(args, 'check_perplexity', False),
    )

from src.embedding.vectorstore import load_vectorstore, search, build_vectorstore
from src.reranking.reranker import score_query_chunks
from src.embedding.embedder import get_embedder
from src.pipeline.data_preprocessing import (
    preprocess_queries_and_chunks,
    clean_text,
    validate_chunk_quality,
    deduplicate_chunks,
)
import numpy as np


def _parse_bool(v) -> bool:
    """解析命令行 bool 参数。"""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('true', '1', 'yes', 'y')


# chunk_version 到 vectorstore chunk_method 的映射
CHUNK_METHOD_MAP = {
    "semantic": "semantic",
    "sliding": "sliding_window",
}


def _get_chunk_method(chunk_version: str) -> str:
    """将 chunk_version 映射为 vectorstore 使用的 chunk_method。"""
    return CHUNK_METHOD_MAP.get(chunk_version, chunk_version)


def ensure_vectorstore_exists(chunk_version: str, chunk_file: str) -> bool:
    """
    检查向量库是否存在，不存在则构建。

    参数
    ----------
    chunk_version : str
        chunk 版本：'semantic' 或 'sliding'
    chunk_file : str
        chunk 文件路径

    返回
    -------
    bool
        向量库是否可用
    """
    from src.embedding.vectorstore import get_vectorstore_dir

    chunk_method = _get_chunk_method(chunk_version)
    vectorstore_dir = get_vectorstore_dir()
    vectorstore_file = vectorstore_dir / f"{chunk_method}_large.pkl"

    if vectorstore_file.exists():
        print(f"  ✓ 向量库已存在：{vectorstore_file}")
        return True

    print(f"  ⚠️  向量库不存在，正在构建...")
    print(f"     从 {chunk_file} 构建...")

    try:
        # 从文件加载 chunks（支持 JSON 数组和 JSONL 格式）
        import json as _json
        with open(chunk_file, "r", encoding="utf-8") as _f:
            _raw = _f.read().strip()
            if _raw.startswith("["):
                _chunks = _json.loads(_raw)
            else:
                _f.seek(0)
                _chunks = [_json.loads(_line) for _line in _f if _line.strip()]

        build_vectorstore(
            chunks=_chunks,
            chunk_method=chunk_method,
            model_type="large",
            batch_size=32,
            save_path=str(vectorstore_dir),
        )
        print(f"  ✓ 向量库构建完成")
        return True
    except Exception as e:
        import traceback
        print(f"  ❌ 向量库构建失败: {e}")
        traceback.print_exc()
        return False


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
                "chunk_id": chunk.get("id", ""),
                "content": chunk.get("content", ""),
                "retrieval_score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return rows


def rerank_topk(query: str, topk_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """步骤 3,4：对 top_k chunk 打分并重排。"""
    chunks = [row["content"] for row in topk_rows]
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
    min_pos_score: float = 0.8,
    max_neg_score: float = 0.7,
    min_margin: float = 0.1,
    max_negatives: int = 1,
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


def compute_chunk_similarity(
    content1: str,
    content2: str,
    embedder_model: Any,
) -> float:
    """计算两个 chunk 之间的 embedding 相似度。"""
    try:
        emb1 = embedder_model.encode([content1])[0]
        emb2 = embedder_model.encode([content2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        print(f"  ⚠️ 相似度计算失败: {e}")
        return 0.0


def build_triplets_for_query(
    query: str,
    positives: Sequence[Dict[str, Any]],
    negatives: Sequence[Dict[str, Any]],
    embedder_model: Any = None,
    min_chunk_diff: float = 0.05,
    max_chunk_similarity: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    步骤 6：一个 query 生成多条 triplet。

    参数
    ----------
    query : str
        查询字符串
    positives : Sequence[Dict]
        正样本列表
    negatives : Sequence[Dict]
        负样本列表
    embedder_model : Any
        嵌入模型，用于计算 chunk 相似度
    min_chunk_diff : float
        Positive 和 negative chunk 内容相似度的最小差异
        - 如果 |content_sim(pos) - content_sim(neg)| < min_chunk_diff，跳过该组合
    max_chunk_similarity : float
        Positive 和 negative chunk 之间的最大允许相似度
        - 如果 chunk_similarity(pos, neg) > max_chunk_similarity，跳过（太相似，无法区分）

    返回
    -------
    List[Dict]
        生成的 triplet 列表
    """
    triplets: List[Dict[str, Any]] = []

    for pos in positives:
        for neg in negatives:
            # 基础 triplet 信息
            triplet = {
                "query": query,
                "positive_chunk_id": pos["chunk_id"],
                "negative_chunk_id": neg["chunk_id"],
                "positive_score": pos["reranker_score"],
                "negative_score": neg["reranker_score"],
                "score_margin": pos["reranker_score"] - neg["reranker_score"],
            }

            # 如果提供了嵌入模型，添加 chunk 相似度检查
            if embedder_model is not None:
                pos_content = pos.get("content", "")
                neg_content = neg.get("content", "")

                if pos_content and neg_content:
                    # 计算 positive/negative chunk 之间的相似度
                    chunk_similarity = compute_chunk_similarity(
                        pos_content, neg_content, embedder_model
                    )
                    triplet["chunk_similarity"] = chunk_similarity

                    # 过滤条件：chunks 不能太相似（否则模型难以学习区分）
                    if chunk_similarity > max_chunk_similarity:
                        continue  # 跳过此组合

                    # 计算 query-content 相似度（可选的额外信息）
                    query_pos_sim = compute_chunk_similarity(query, pos_content, embedder_model)
                    query_neg_sim = compute_chunk_similarity(query, neg_content, embedder_model)
                    triplet["query_positive_similarity"] = query_pos_sim
                    triplet["query_negative_similarity"] = query_neg_sim

                    # 过滤：query 与 positive 的相似度应显著大于与 negative 的相似度
                    if (query_pos_sim - query_neg_sim) < min_chunk_diff:
                        continue  # 跳过此组合

            triplets.append(triplet)

    return triplets


def save_jsonl(rows: Sequence[Dict[str, Any]], output_path: str) -> None:
    """保存为 JSONL 或 JSON 格式（根据文件扩展名自动判断）。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith('.json'):
        # 保存为 JSON 数组格式
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(rows), f, ensure_ascii=False, indent=2)
    else:
        # 保存为 JSONL 格式（每行一个 JSON）
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_unified_pipeline(
    query_path: str,
    chunk_version: str,
    triplet_output_path: str,
    chunk_file: Optional[str] = None,
    top_k: int = 10,
    min_pos_score: float = 0.8,
    max_neg_score: float = 0.7,
    min_margin: float = 0.1,
    max_negatives_per_query: int = 1,
    top10_output_path: Optional[str] = None,
    min_chunk_diff: float = 0.05,
    max_chunk_similarity: float = 0.95,
    use_chunk_similarity_filter: bool = True,
    # ===== 新增预处理参数 =====
    enable_preprocessing: bool = True,
    clean_texts: bool = True,
    min_query_length: int = 5,
    min_chunk_length: int = 50,
    max_chunk_length: int = 1000,
    min_info_density: float = 0.1,
    check_chunk_containment: bool = True,
    deduplicate: bool = True,
    dedup_similarity_threshold: float = 0.95,
    check_perplexity: bool = False,
) -> Dict[str, int]:
    """
    运行统一 triplet 构建流水线（支持多版本 chunks）。

    参数
    ----------
    query_path : str
        query 文件路径 (txt/json/jsonl)
    chunk_version : str
        chunk 版本：'semantic' 或 'sliding'
    triplet_output_path : str
        输出 triplet 文件路径
    chunk_file : str, optional
        chunk 数据文件路径（自动查找：data/processed/chunks_{chunk_version}_cleaned.json）
    top_k : int
        检索 top_k
    min_pos_score : float
        Positive reranker score 阈值
    max_neg_score : float
        Negative reranker score 上界
    min_margin : float
        Positive-Negative 最小 margin
    max_negatives_per_query : int
        每个 query 的最大 negative 数量
    top10_output_path : str, optional
        可选：保存 top10 reranked 结果
    min_chunk_diff : float
        Query 与 Positive/Negative chunk 相似度的最小差异
    max_chunk_similarity : float
        Positive 和 Negative chunk 之间的最大允许相似度
    use_chunk_similarity_filter : bool
        是否启用 chunk 相似度过滤

    返回
    -------
    Dict[str, int]
        统计信息
    """
    print("=" * 70)
    print(f"【统一 Triplet 生成管道】- {chunk_version.upper()} 版本")
    print("=" * 70)

    print("\n[1/8] 加载 query...")
    raw_queries = load_queries(query_path)
    print(f"  ✓ 加载 {len(raw_queries)} 条 query")

    # ===== 新增：预处理 Query =====
    if enable_preprocessing:
        print("\n[1.5/8] 预处理 Query...")
        queries, _, preprocess_stats = preprocess_queries_and_chunks(
            raw_queries,
            [],  # 暂时不处理 chunks
            embedder_model=None,
            clean_text_=clean_texts,
            min_query_length=min_query_length,
            check_perplexity=check_perplexity,
            verbose=True,
        )
        print(f"  ✓ 预处理后保留：{len(queries)} 条 query")
    else:
        queries = raw_queries
        print("  ⚠️  跳过预处理")

    if not queries:
        print("❌ 没有有效的 query，无法继续")
        sys.exit(1)

    print("\n[2/8] 加载向量库...")

    # 自动确定 chunk 文件路径
    if chunk_file is None:
        project_root = Path(__file__).parent.parent.parent
        chunk_file = project_root / f"data/processed/chunks_{chunk_version}_cleaned.json"
        if not chunk_file.exists():
            print(f"  ❌ Chunk 文件不存在: {chunk_file}")
            sys.exit(1)

    # 检查或构建向量库
    if not ensure_vectorstore_exists(chunk_version, str(chunk_file)):
        print(f"  ❌ 无法加载或构建向量库")
        sys.exit(1)

    vectorstore = load_vectorstore(chunk_method=_get_chunk_method(chunk_version), model_type="large")
    print(f"  ✓ 向量库加载完成 ({chunk_version})")

    # 加载嵌入模型（用于计算 chunk 相似度）
    embedder_model = None
    if use_chunk_similarity_filter:
        print("\n[加载嵌入模型用于相似度计算...]")
        embedder_config = get_embedder(model_type="large")
        embedder_model = embedder_config["load"]()
        print(f"  ✓ 嵌入模型加载完成")

    top10_rows: List[Dict[str, Any]] = []
    triplets: List[Dict[str, Any]] = []
    stats = {
        "total_queries": len(queries),
        "queries_with_positives": 0,
        "queries_with_negatives": 0,
        "triplet_count": 0,
        "failed_queries": 0,
    }

    print("\n[3-6/7] 逐 query 处理...")
    print("-" * 70)

    for i, query in enumerate(queries, start=1):
        try:
            # 步骤 4：检索 top_k
            retrieved = retrieve_topk_chunks(query, vectorstore=vectorstore, top_k=top_k)
            if not retrieved:
                stats["failed_queries"] += 1
                continue

            # ===== 新增：质量过滤 =====
            if enable_preprocessing:
                filtered_retrieved = []
                for chunk in retrieved:
                    content = chunk.get("content", "")
                    valid, reason = validate_chunk_quality(
                        content,
                        min_length=min_chunk_length,
                        max_length=max_chunk_length,
                        min_density=min_info_density,
                        check_containment=check_chunk_containment,
                    )
                    if valid:
                        filtered_retrieved.append(chunk)

                if not filtered_retrieved:
                    stats["failed_queries"] += 1
                    continue

                retrieved = filtered_retrieved

            # ===== 新增：去重 =====
            if deduplicate and embedder_model is not None:
                retrieved, _ = deduplicate_chunks(
                    retrieved,
                    embedder_model,
                    similarity_threshold=dedup_similarity_threshold,
                )
                if not retrieved:
                    stats["failed_queries"] += 1
                    continue

            # 步骤 5,6：Rerank
            reranked = rerank_topk(query, retrieved)

            top10_rows.append(
                {
                    "query": query,
                    "top_chunks": reranked,
                }
            )

            # 步骤 7：选择 positive/negative
            positives, negatives = select_pos_neg(
                reranked_rows=reranked,
                min_pos_score=min_pos_score,
                max_neg_score=max_neg_score,
                min_margin=min_margin,
                max_negatives=max_negatives_per_query,
            )

            if positives:
                stats["queries_with_positives"] += 1
            if negatives:
                stats["queries_with_negatives"] += 1

            # 步骤 8：生成 triplets
            query_triplets = build_triplets_for_query(
                query,
                positives,
                negatives,
                embedder_model=embedder_model if use_chunk_similarity_filter else None,
                min_chunk_diff=min_chunk_diff,
                max_chunk_similarity=max_chunk_similarity,
            )
            triplets.extend(query_triplets)

            if i % max(1, len(queries) // 10) == 0 or i == len(queries):
                print(f"  [{i:4d}/{len(queries):4d}] ✓ {len(query_triplets):2d} triplets | "
                      f"正数 {len(positives):2d} | 负数 {len(negatives):2d}")

            stats["triplet_count"] += len(query_triplets)

        except Exception as e:
            print(f"  [FAIL] Query {i}: {e}")
            stats["failed_queries"] += 1
            continue

    print("-" * 70)

    # 步骤 8/9：输出结果
    print("\n[8/9] 输出结果...")

    save_jsonl(triplets, triplet_output_path)
    print(f"  ✓ Triplet 已保存: {triplet_output_path}")

    if top10_output_path:
        save_jsonl(top10_rows, top10_output_path)
        print(f"  ✓ Top10 已保存: {top10_output_path}")

    # 输出统计
    print("\n" + "=" * 70)
    print("【统计信息】")
    print("=" * 70)
    print(f"总 Query 数:         {stats['total_queries']}")
    print(f"有 Positive 的:      {stats['queries_with_positives']}")
    print(f"有 Negative 的:      {stats['queries_with_negatives']}")
    print(f"生成 Triplet 总数:   {stats['triplet_count']}")
    print(f"失败的 Query:        {stats['failed_queries']}")
    if stats["queries_with_positives"] > 0:
        avg_triplets = stats["triplet_count"] / stats["queries_with_positives"]
        print(f"平均每 query triplet: {avg_triplets:.2f}")
    print("=" * 70)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一 Triplet 生成管道 - 支持多版本 chunks"
    )
    parser.add_argument(
        "--query-path",
        required=True,
        help="query 文件路径 (txt/json/jsonl)"
    )
    parser.add_argument(
        "--chunk-version",
        required=True,
        choices=["semantic", "sliding"],
        help="Chunk 版本：'semantic' 或 'sliding'"
    )
    parser.add_argument(
        "--triplet-output",
        required=True,
        help="Triplet 输出路径"
    )
    parser.add_argument(
        "--chunk-file",
        default=None,
        help="Chunk 数据文件路径（可选，自动查找 data/processed/chunks_{version}_cleaned.json）"
    )
    parser.add_argument(
        "--top10-output",
        default=None,
        help="可选：Top10 reranked 结果输出路径"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-pos-score", type=float, default=0.8)
    parser.add_argument("--max-neg-score", type=float, default=0.7)
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--max-negatives-per-query", type=int, default=1)

    # chunk 相似度过滤参数
    parser.add_argument(
        "--use-chunk-similarity-filter",
        type=_parse_bool,
        default=True,
        help="是否使用 chunk 相似度过滤（默认：True）"
    )
    parser.add_argument(
        "--min-chunk-diff",
        type=float,
        default=0.05,
        help="Query-Positive 与 Query-Negative 相似度的最小差异（默认：0.05）"
    )
    parser.add_argument(
        "--max-chunk-similarity",
        type=float,
        default=0.95,
        help="Positive 和 Negative chunk 之间的最大允许相似度（默认：0.95）"
    )

    # ===== 预处理参数 =====
    parser.add_argument("--enable-preprocessing", type=_parse_bool, default=True, help="启用预处理")
    parser.add_argument("--clean-texts", type=_parse_bool, default=True, help="启用文本清洗")
    parser.add_argument("--min-query-length", type=int, default=5)
    parser.add_argument("--min-chunk-length", type=int, default=50)
    parser.add_argument("--max-chunk-length", type=int, default=1000)
    parser.add_argument("--min-info-density", type=float, default=0.1)
    parser.add_argument("--check-chunk-containment", type=_parse_bool, default=True)
    parser.add_argument("--deduplicate", type=_parse_bool, default=True)
    parser.add_argument("--dedup-similarity-threshold", type=float, default=0.95)
    parser.add_argument("--check-perplexity", type=_parse_bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_unified_pipeline(
        query_path=args.query_path,
        chunk_version=args.chunk_version,
        triplet_output_path=args.triplet_output,
        chunk_file=args.chunk_file,
        top10_output_path=args.top10_output,
        top_k=args.top_k,
        min_pos_score=args.min_pos_score,
        max_neg_score=args.max_neg_score,
        min_margin=args.min_margin,
        max_negatives_per_query=args.max_negatives_per_query,
        use_chunk_similarity_filter=args.use_chunk_similarity_filter,
        min_chunk_diff=args.min_chunk_diff,
        max_chunk_similarity=args.max_chunk_similarity,
        enable_preprocessing=getattr(args, 'enable_preprocessing', True),
        clean_texts=getattr(args, 'clean_texts', True),
        min_query_length=getattr(args, 'min_query_length', 5),
        min_chunk_length=getattr(args, 'min_chunk_length', 50),
        max_chunk_length=getattr(args, 'max_chunk_length', 1000),
        min_info_density=getattr(args, 'min_info_density', 0.1),
        check_chunk_containment=getattr(args, 'check_chunk_containment', True),
        deduplicate=getattr(args, 'deduplicate', True),
        dedup_similarity_threshold=getattr(args, 'dedup_similarity_threshold', 0.95),
        check_perplexity=getattr(args, 'check_perplexity', False),
    )
