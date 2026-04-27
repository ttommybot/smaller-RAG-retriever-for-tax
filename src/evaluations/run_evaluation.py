# -*- coding: utf-8 -*-
"""
RAG 评估模块

本模块提供完整的评估流程：
1. 加载/构建向量库
2. 运行检索（可选重排）
3. 与标准答案对比
4. 计算质量指标（Recall@k, HitRate@k, MRR@k, nDCG@k）
5. 记录效率指标（时间、显存）
"""

import json
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Set, Optional, Any
from pathlib import Path
from dataclasses import dataclass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys_path_added = False
if str(PROJECT_ROOT / "src") not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys_path_added = True

from embedding.vectorstore import (
    build_vectorstore_for_custom_model,
    load_vectorstore_for_custom_model,
    search_custom_vectorstore,
    get_vectorstore_model_dir,
    parse_model_name,
    get_chunk_methods_for_model,
    get_vectorstore_dir,
)
from embedding.embedder import load_custom_model, get_custom_embedder
from reranking.reranker import load_reranker, rerank_chunks


# ==========================================
# 配置区
# ==========================================
CONFIG = {
    "eval_data_dir": PROJECT_ROOT / "data" / "evaluations",
    "eval_queries_file": PROJECT_ROOT / "data" / "query" / "evaluation_queries.txt",
    "eval_results_dir": PROJECT_ROOT / "data" / "evaluations" / "results",
    "k_list": [10],
    "chunk_types": ["semantic", "sliding"],
    "gt_file_template": "evaluation_criteria_{chunk_type}.jsonl",
    "top_k_retrieval": 20,  # 检索数量（重排前）
    "top_k_final": 10,      # 最终输出数量（重排后）
    "reranker_name": "BAAI/bge-reranker-v2-gemma",
}


# ==========================================
# 数据类：用于存储完整的评估结果（质量+效率）
# ==========================================
@dataclass
class EvaluationResult:
    model_name: str
    chunk_type: str
    use_reranker: bool
    total_queries: int

    # 质量指标
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    hitrate_at_1: float = 0.0
    hitrate_at_3: float = 0.0
    hitrate_at_5: float = 0.0
    hitrate_at_10: float = 0.0
    mrr_at_1: float = 0.0
    mrr_at_3: float = 0.0
    mrr_at_5: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_1: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0

    # 效率指标
    avg_retrieval_latency_ms: float = 0.0
    avg_rerank_latency_ms: float = 0.0
    avg_end_to_end_latency_ms: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    vectordb_build_time_s: float = 0.0
    vectordb_build_memory_mb: float = 0.0


# ==========================================
# 核心质量指标计算函数
# ==========================================
def calculate_quality_metrics(
    gt_chunk_ids: Set[str],
    retrieved_chunk_ids: List[str],
    k_list: List[int]
) -> Dict:
    metrics = {}
    total_relevant = len(gt_chunk_ids)

    # 找到第一个相关chunk的排名
    first_rel_rank = None
    for idx, cid in enumerate(retrieved_chunk_ids):
        if cid in gt_chunk_ids:
            first_rel_rank = idx + 1
            break

    for k in k_list:
        topk_retrieved = retrieved_chunk_ids[:k]
        hit_count = sum(1 for cid in topk_retrieved if cid in gt_chunk_ids)

        # 1. Recall@k
        recall = hit_count / total_relevant if total_relevant > 0 else 0.0
        metrics[f"Recall@{k}"] = recall

        # 2. Hit Rate@k
        hit_rate = 1.0 if hit_count > 0 else 0.0
        metrics[f"HitRate@{k}"] = hit_rate

        # 3. MRR@k
        mrr = 0.0
        if first_rel_rank is not None and first_rel_rank <= k:
            mrr = 1.0 / first_rel_rank
        metrics[f"MRR@{k}"] = mrr

        # 4. nDCG@k
        dcg = 0.0
        for i, cid in enumerate(topk_retrieved):
            if cid in gt_chunk_ids:
                dcg += 1.0 / np.log2(i + 2)

        ideal_hit_count = min(total_relevant, k)
        idcg = 0.0
        for i in range(ideal_hit_count):
            idcg += 1.0 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"nDCG@{k}"] = ndcg

    return metrics, first_rel_rank is not None


# ==========================================
# 数据加载函数
# ==========================================
def load_jsonl(file_path: Path) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_queries(query_file: Path) -> List[str]:
    """加载评估问题列表。"""
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    return queries


def extract_chunk_ids_from_top_chunks(top_chunks: List[Dict]) -> List[str]:
    return [chunk["chunk_id"] for chunk in top_chunks]


# ==========================================
# 效率记录函数
# ==========================================
def get_gpu_memory_mb() -> float:
    """获取当前 GPU 显存使用量（MB）。"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def reset_gpu_memory_stats():
    """重置 GPU 显存统计。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ==========================================
# 单次评估运行函数
# ==========================================
def run_single_evaluation(
    model_name: str,
    chunk_method: str,
    use_reranker: bool = False,
    queries: List[str] = None,
    use_cuda: bool = True,
) -> EvaluationResult:
    """
    运行单个模型的评估。

    注意：向量库必须已存在（由 embed_to_vectordb.py 构建）。

    参数
    ----------
    model_name : str
        模型名称（如 "sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3"）。

    chunk_method : str
        chunk 方法 ('semantic' 或 'sliding')。

    use_reranker : bool
        是否使用 reranker。

    queries : List[str], optional
        评估问题列表，默认从配置加载。

    use_cuda : bool
        是否使用 GPU。

    返回
    -------
    EvaluationResult
        评估结果。
    """
    import torch

    # 初始化结果
    result = EvaluationResult(
        model_name=model_name,
        chunk_type=chunk_method,
        use_reranker=use_reranker,
        total_queries=0
    )

    # 加载 queries
    if queries is None:
        queries = load_queries(CONFIG["eval_queries_file"])

    # 加载标准答案
    gt_file_name = CONFIG["gt_file_template"].format(chunk_type=chunk_method)
    gt_file_path = CONFIG["eval_data_dir"] / gt_file_name
    if not gt_file_path.exists():
        print(f"❌ 标准答案文件不存在：{gt_file_path}")
        return result

    gt_data = load_jsonl(gt_file_path)
    gt_dict = {}
    for item in gt_data:
        query = item["query"].strip()
        gt_chunk_ids = set(extract_chunk_ids_from_top_chunks(item["top_chunks"]))
        gt_dict[query] = gt_chunk_ids

    print("\n" + "=" * 60)
    print(f"评估配置：{model_name} | chunk={chunk_method} | reranker={use_reranker}")
    print("=" * 60)

    # ==================== 步骤 1: 加载向量库（不构建） ====================
    print("\n步骤 1: 加载向量库")

    reset_gpu_memory_stats()
    load_start_time = time.time()

    vectorstore_dir = get_vectorstore_model_dir(model_name, chunk_method)
    embeddings_path = vectorstore_dir / "embeddings.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"向量库不存在：{vectorstore_dir}\n"
            f"请先运行：python scripts/embed_to_vectordb.py --models {model_name}"
        )

    print(f"  加载向量库：{vectorstore_dir}")
    vectorstore = load_vectorstore_for_custom_model(model_name, chunk_method)

    load_end_time = time.time()
    result.vectordb_build_time_s = 0  # 不构建，只加载
    result.vectordb_build_memory_mb = 0
    print(f"  向量库加载完成，耗时：{load_end_time - load_start_time:.2f}s")

    # ==================== 步骤 2: 加载模型 ====================
    print("\n步骤 2: 加载模型")

    embedder = get_custom_embedder(model_name)
    model = embedder['model']
    print(f"  Embedding 模型已加载：{model_name}")

    if use_reranker:
        reranker = load_reranker(CONFIG["reranker_name"])
        print(f"  Reranker 已加载：{CONFIG['reranker_name']}")

    # ==================== 步骤 3: 运行检索评估 ====================
    print("\n步骤 3: 运行检索评估")

    # 创建结果目录
    CONFIG["eval_results_dir"].mkdir(parents=True, exist_ok=True)

    # 结果文件名
    result_suffix = "_reordered" if use_reranker else ""
    result_file_name = f"result_{model_name}_{chunk_method}{result_suffix}.jsonl"
    result_file_path = CONFIG["eval_results_dir"] / result_file_name

    all_retrieval_times = []
    all_rerank_times = []
    all_end_to_end_times = []

    result_items = []
    matched_queries = 0

    for query in tqdm(queries, desc="评估查询"):
        query = query.strip()
        if query not in gt_dict:
            continue

        matched_queries += 1

        # 重置 GPU 统计
        reset_gpu_memory_stats()

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
                    "reranker_score": item.get("score", 0),  # rerank_chunks returns "score"
                    "metadata": raw.get("metadata", {})
                })
        else:
            final_chunks = candidate_chunks[:CONFIG["top_k_final"]]

        end_to_end_time_ms = retrieval_time_ms + rerank_time_ms
        all_end_to_end_times.append(end_to_end_time_ms)

        # 记录峰值显存
        peak_memory = get_gpu_memory_mb()
        if peak_memory > result.peak_gpu_memory_mb:
            result.peak_gpu_memory_mb = peak_memory

        # 保存结果
        result_item = {
            "query": query,
            "top_chunks": final_chunks,
            "efficiency": {
                "retrieval_latency_ms": retrieval_time_ms,
                "rerank_latency_ms": rerank_time_ms,
                "end_to_end_latency_ms": end_to_end_time_ms
            }
        }
        result_items.append(result_item)

    # ==================== 步骤 4: 保存检索结果 ====================
    print(f"\n步骤 4: 保存检索结果 ({len(result_items)} 条)")

    with open(result_file_path, 'w', encoding='utf-8') as f:
        for item in result_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  结果已保存：{result_file_path}")

    # ==================== 步骤 5: 计算质量指标 ====================
    print("\n步骤 5: 计算质量指标")

    result.total_queries = matched_queries

    all_quality_metrics = {
        f"{m}@{k}": []
        for m in ["Recall", "HitRate", "MRR", "nDCG"]
        for k in CONFIG["k_list"]
    }

    for item in result_items:
        query = item["query"]
        retrieved_ids = [chunk["chunk_id"] for chunk in item["top_chunks"]]
        gt_ids = gt_dict.get(query, set())

        metrics, _ = calculate_quality_metrics(
            gt_chunk_ids=gt_ids,
            retrieved_chunk_ids=retrieved_ids,
            k_list=CONFIG["k_list"]
        )

        for key, val in metrics.items():
            all_quality_metrics[key].append(val)

    # 计算平均值并填充结果
    print(f"\n📊 【{model_name} ({chunk_method}) | reranker={use_reranker}】质量指标：")
    for key in sorted(all_quality_metrics.keys()):
        avg_val = np.mean(all_quality_metrics[key])
        attr_name = key.replace("@", "_at_").lower()
        setattr(result, attr_name, round(avg_val, 4))
        print(f"  {key:15} : {avg_val:.4f}")

    # ==================== 步骤 6: 计算效率指标 ====================
    print(f"\n⚡ 效率指标：")
    if all_retrieval_times:
        result.avg_retrieval_latency_ms = round(np.mean(all_retrieval_times), 2)
        print(f"  平均检索延迟：{result.avg_retrieval_latency_ms:.2f} ms")
    if all_rerank_times and use_reranker:
        result.avg_rerank_latency_ms = round(np.mean(all_rerank_times), 2)
        print(f"  平均重排延迟：{result.avg_rerank_latency_ms:.2f} ms")
    if all_end_to_end_times:
        result.avg_end_to_end_latency_ms = round(np.mean(all_end_to_end_times), 2)
        print(f"  平均端到端延迟：{result.avg_end_to_end_latency_ms:.2f} ms")

    print(f"  峰值显存占用：{result.peak_gpu_memory_mb:.2f} MB")
    print(f"  向量库构建时间：{result.vectordb_build_time_s:.2f} s")

    return result


# ==========================================
# 批量评估函数
# ==========================================
def evaluate_model_all_configs(
    model_name: str,
    use_cuda: bool = True,
    force_rebuild_vectordb: bool = False
) -> List[EvaluationResult]:
    """
    评估单个模型的所有配置（chunk 方法 × 是否重排）。

    参数
    ----------
    model_name : str
        模型名称。

    use_cuda : bool
        是否使用 GPU。

    force_rebuild_vectordb : bool
        是否强制重建向量库。

    返回
    -------
    List[EvaluationResult]
        所有配置的评估结果。
    """
    results = []

    # 确定 chunk 方法
    chunk_methods = get_chunk_methods_for_model(model_name)

    # 加载问题
    queries = load_queries(CONFIG["eval_queries_file"])

    for chunk_method in chunk_methods:
        for use_reranker in [False, True]:
            try:
                result = run_single_evaluation(
                    model_name=model_name,
                    chunk_method=chunk_method,
                    use_reranker=use_reranker,
                    queries=queries,
                    force_rebuild_vectordb=force_rebuild_vectordb,
                    use_cuda=use_cuda
                )
                results.append(result)
            except Exception as e:
                print(f"❌ 评估失败：{model_name} ({chunk_method}) reranker={use_reranker}")
                print(f"  错误：{e}")

    return results


def batch_evaluate_models(
    model_names: List[str],
    use_cuda: bool = True,
    force_rebuild_vectordb: bool = False
) -> List[EvaluationResult]:
    """
    批量评估多个模型。

    参数
    ----------
    model_names : List[str]
        模型名称列表。

    use_cuda : bool
        是否使用 GPU。

    force_rebuild_vectordb : bool
        是否强制重建向量库。

    返回
    -------
    List[EvaluationResult]
        所有评估结果。
    """
    all_results = []

    print("=" * 60)
    print("批量评估开始")
    print("=" * 60)
    print(f"模型数量：{len(model_names)}")
    print(f"每个模型配置数：最多 4 种 (2 chunk × 2 reranker)")

    for i, model_name in enumerate(model_names):
        print(f"\n[{i+1}/{len(model_names)}] 评估模型：{model_name}")
        results = evaluate_model_all_configs(
            model_name=model_name,
            use_cuda=use_cuda,
            force_rebuild_vectordb=force_rebuild_vectordb
        )
        all_results.extend(results)

    print("\n" + "=" * 60)
    print("批量评估完成")
    print("=" * 60)

    return all_results


# ==========================================
# 结果对比函数
# ==========================================
def compare_results(results: List[EvaluationResult]) -> pd.DataFrame:
    """
    将评估结果转换为 DataFrame 并生成对比报告。
    """
    df = pd.DataFrame([vars(r) for r in results])
    return df


def save_results_table(results: List[EvaluationResult], output_path: Path) -> None:
    """
    保存评估结果对比表格。

    参数
    ----------
    results : List[EvaluationResult]
        评估结果列表。
    output_path : Path
        输出文件路径（不含扩展名）。
    """
    # 保存 JSON
    json_path = output_path.with_suffix('.json')
    results_dict = [vars(r) for r in results]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到：{json_path}")

    # 保存 Markdown 表格
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# RAG 评估结果对比\n\n")
        f.write(f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 质量指标\n\n")
        f.write("| Model | Chunk | Reranker | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |\n")
        f.write("|-------|-------|----------|----------|-----------|--------|----------|\n")

        for r in results:
            f.write(f"| {r.model_name} | {r.chunk_type} | {r.use_reranker} | ")
            f.write(f"{r.recall_at_5:.4f} | {r.recall_at_10:.4f} | {r.mrr_at_10:.4f} | {r.ndcg_at_10:.4f} |\n")

        f.write("\n## 效率指标\n\n")
        f.write("| Model | Chunk | Reranker | 检索(ms) | 重排(ms) | 端到端(ms) | 显存(MB) |\n")
        f.write("|-------|-------|----------|----------|----------|------------|----------|\n")

        for r in results:
            f.write(f"| {r.model_name} | {r.chunk_type} | {r.use_reranker} | ")
            f.write(f"{r.avg_retrieval_latency_ms:.2f} | {r.avg_rerank_latency_ms:.2f} | ")
            f.write(f"{r.avg_end_to_end_latency_ms:.2f} | {r.peak_gpu_memory_mb:.2f} |\n")

    print(f"Markdown 表格已保存到：{md_path}")


# ==========================================
# 主函数（用于单模块测试）
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 评估脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers--all-MiniLM-L6-v2",
        help="要评估的模型名称"
    )
    parser.add_argument(
        "--chunk",
        type=str,
        default="semantic",
        help="chunk 方法 (semantic/sliding)"
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

    result = run_single_evaluation(
        model_name=args.model,
        chunk_method=args.chunk,
        use_reranker=args.reranker,
        use_cuda=not args.no_cuda,
    )

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
    print(f"模型：{result.model_name}")
    print(f"Chunk：{result.chunk_type}")
    print(f"Reranker：{result.use_reranker}")
    print(f"Recall@5：{result.recall_at_5:.4f}")
    print(f"Recall@10：{result.recall_at_10:.4f}")
    print(f"MRR@10：{result.mrr_at_10:.4f}")
    print(f"nDCG@10：{result.ndcg_at_10:.4f}")