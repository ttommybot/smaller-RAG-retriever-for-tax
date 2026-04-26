# -*- coding: utf-8 -*-
"""
直接生成检索结果脚本

跳过 criteria 加载，直接运行检索并保存结果文件。
用于生成 result_xxx.jsonl 文件。

使用方法：
    python scripts/crse.py --model BAAI--bge-large-zh-v1.5 --chunk semantic --reranker
    python scripts/crse.py --model sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 --chunk semantic
"""

import sys
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding.vectorstore import (
    load_vectorstore_for_custom_model,
    search_custom_vectorstore,
    get_vectorstore_model_dir,
)
from embedding.embedder import get_custom_embedder
from reranking.reranker import load_reranker, rerank_chunks


# ==========================================
# 配置
# ==========================================
CONFIG = {
    "eval_queries_file": PROJECT_ROOT / "data" / "query" / "evaluation_queries.txt",
    "eval_results_dir": PROJECT_ROOT / "data" / "evaluations" / "results",
    "top_k_retrieval": 20,
    "top_k_final": 10,
    "reranker_name": "BAAI/bge-reranker-v2-gemma",
}


def load_queries(query_file: Path) -> List[str]:
    """加载评估问题列表。"""
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    return queries


def run_retrieval_only(
    model_name: str,
    chunk_method: str,
    use_reranker: bool = False,
    use_cuda: bool = True,
) -> Dict[str, Any]:
    """
    运行检索（不评估），直接生成结果文件。

    参数
    ----------
    model_name : str
        模型名称。

    chunk_method : str
        chunk 方法 ('semantic' 或 'sliding')。

    use_reranker : bool
        是否使用 reranker。

    use_cuda : bool
        是否使用 GPU。

    返回
    -------
    Dict[str, Any]
        运行信息。
    """
    import torch

    result_info = {
        "model_name": model_name,
        "chunk_method": chunk_method,
        "use_reranker": use_reranker,
        "total_queries": 0,
        "success": False,
        "output_file": "",
        "avg_retrieval_ms": 0,
        "avg_rerank_ms": 0,
        "peak_memory_mb": 0,
    }

    # 加载 queries
    queries = load_queries(CONFIG["eval_queries_file"])
    result_info["total_queries"] = len(queries)

    print("\n" + "=" * 60)
    print(f"检索配置：{model_name} | {chunk_method} | reranker={use_reranker}")
    print("=" * 60)

    # ==================== 步骤 1: 加载向量库 ====================
    print("\n步骤 1: 加载向量库")

    vectorstore_dir = get_vectorstore_model_dir(model_name, chunk_method)
    embeddings_path = vectorstore_dir / "embeddings.npy"

    if not embeddings_path.exists():
        print(f"❌ 向量库不存在：{vectorstore_dir}")
        return result_info

    print(f"  加载向量库：{vectorstore_dir}")
    vectorstore = load_vectorstore_for_custom_model(model_name, chunk_method)
    print(f"  加载完成：{len(vectorstore['chunks'])} chunks")

    # ==================== 步骤 2: 加载模型 ====================
    print("\n步骤 2: 加载模型")

    embedder = get_custom_embedder(model_name)
    model = embedder['model']
    print(f"  Embedding 模型已加载")

    if use_reranker:
        reranker = load_reranker(CONFIG["reranker_name"])
        print(f"  Reranker 已加载")

    # ==================== 步骤 3: 运行检索 ====================
    print("\n步骤 3: 运行检索")

    # 创建结果目录
    CONFIG["eval_results_dir"].mkdir(parents=True, exist_ok=True)

    # 结果文件名
    result_suffix = "_reordered" if use_reranker else ""
    result_file_name = f"result_{model_name}_{chunk_method}{result_suffix}.jsonl"
    result_file_path = CONFIG["eval_results_dir"] / result_file_name
    result_info["output_file"] = str(result_file_path)

    # 重置 GPU 统计
    if use_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    all_retrieval_times = []
    all_rerank_times = []

    result_items = []

    for query in tqdm(queries, desc="检索查询"):
        query = query.strip()

        # 检索阶段
        retrieval_start = time.time()
        retrieved_results = search_custom_vectorstore(
            query=query,
            vectorstore=vectorstore,
            model=model,
            top_k=CONFIG["top_k_retrieval"]
        )
        retrieval_end = time.time()
        retrieval_time_ms = (retrieval_end - retrieval_start) * 1000
        all_retrieval_times.append(retrieval_time_ms)

        # 构建候选 chunks
        candidate_chunks = []
        for chunk, score in retrieved_results:
            candidate_chunks.append({
                "chunk_id": chunk.get("id", ""),
                "content": chunk.get("content", ""),
                "retrieval_score": score,
                "metadata": chunk.get("metadata", {})
            })

        # 重排阶段（如果启用）
        rerank_time_ms = 0.0
        if use_reranker and len(candidate_chunks) > 0:
            rerank_start = time.time()
            reranked = rerank_chunks(
                query=query,
                chunks=candidate_chunks,
                top_k=CONFIG["top_k_final"],
                text_key="content"
            )
            rerank_end = time.time()
            rerank_time_ms = (rerank_end - rerank_start) * 1000
            all_rerank_times.append(rerank_time_ms)

            # 构建最终结果
            final_chunks = []
            for item in reranked:
                raw = item.get("raw", {})
                final_chunks.append({
                    "chunk_id": raw.get("chunk_id", ""),
                    "content": item.get("text", ""),
                    "retrieval_score": raw.get("retrieval_score", 0),
                    "reranker_score": item.get("score", 0),
                    "metadata": raw.get("metadata", {})
                })
        else:
            final_chunks = candidate_chunks[:CONFIG["top_k_final"]]

        # 记录峰值显存
        if use_cuda and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if peak_memory > result_info["peak_memory_mb"]:
                result_info["peak_memory_mb"] = peak_memory

        # 保存结果
        result_item = {
            "query": query,
            "top_chunks": final_chunks,
            "efficiency": {
                "retrieval_latency_ms": retrieval_time_ms,
                "rerank_latency_ms": rerank_time_ms,
                "end_to_end_latency_ms": retrieval_time_ms + rerank_time_ms
            }
        }
        result_items.append(result_item)

    # ==================== 步骤 4: 保存结果 ====================
    print(f"\n步骤 4: 保存结果 ({len(result_items)} 条)")

    with open(result_file_path, 'w', encoding='utf-8') as f:
        for item in result_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  ✅ 结果已保存：{result_file_path}")

    # 计算平均延迟
    if all_retrieval_times:
        result_info["avg_retrieval_ms"] = sum(all_retrieval_times) / len(all_retrieval_times)
    if all_rerank_times and use_reranker:
        result_info["avg_rerank_ms"] = sum(all_rerank_times) / len(all_rerank_times)

    result_info["success"] = True

    print(f"\n⚡ 效率统计：")
    print(f"  平均检索延迟：{result_info['avg_retrieval_ms']:.2f} ms")
    if use_reranker:
        print(f"  平均重排延迟：{result_info['avg_rerank_ms']:.2f} ms")
    print(f"  峰值显存占用：{result_info['peak_memory_mb']:.2f} MB")

    return result_info


def main():
    import argparse

    parser = argparse.ArgumentParser(description="直接生成检索结果脚本")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称，如 BAAI--bge-large-zh-v1.5"
    )
    parser.add_argument(
        "--chunk",
        type=str,
        required=True,
        choices=["semantic", "sliding"],
        help="chunk 方法"
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="是否使用 reranker"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="不使用 GPU"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("直接检索结果生成")
    print("=" * 60)

    result = run_retrieval_only(
        model_name=args.model,
        chunk_method=args.chunk,
        use_reranker=args.reranker,
        use_cuda=not args.no_cuda,
    )

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)

    if result["success"]:
        print(f"✅ 成功")
        print(f"  输出文件：{result['output_file']}")
    else:
        print(f"❌ 失败")


if __name__ == "__main__":
    main()