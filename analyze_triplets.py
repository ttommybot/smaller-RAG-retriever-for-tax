# -*- coding: utf-8 -*-
"""
Triplet 数据质量分析 - 查看 positive/negative chunk 的差异

用法：
  python scripts/analyze_triplets.py \
    --triplet-data data/training/semantic_triplets.json \
    --chunk-data data/processed/chunks_semantic_cleaned.json

   python scripts/analyze_triplets.py \
    --triplet-data data/training/sliding_triplets.json \
    --chunk-data data/processed/chunks_sliding_cleaned.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.embedder import get_embedder


def load_chunks(chunk_file: str) -> Dict[str, Dict[str, Any]]:
    """加载 chunks，按 id 索引。支持 JSON Lines 和 JSON 数组格式。"""
    chunks = {}
    with open(chunk_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # 尝试解析为 JSON 数组
        try:
            data = json.loads(content)
            if isinstance(data, list):
                print(f"  ✓ 检测到 JSON 数组格式，包含 {len(data)} 条 chunks")
                for chunk in data:
                    chunks[chunk['id']] = chunk
                return chunks
            elif isinstance(data, dict):
                print(f"  ✓ 检测到 JSON 对象格式，1 条 chunk")
                chunks[data['id']] = data
                return chunks
        except json.JSONDecodeError:
            pass
        
        # 如果不是有效的 JSON，尝试按行解析（JSON Lines 格式）
        f.seek(0)
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks[chunk['id']] = chunk
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第 {line_num} 行解析失败: {e}")
                print(f"     内容: {line[:100]}...")
                continue
    
    print(f"  ✓ 检测到 JSON Lines 格式，包含 {len(chunks)} 条 chunks")
    return chunks


def load_triplets(triplet_file: str) -> List[Dict[str, Any]]:
    """加载 triplets，支持 JSON Lines 和 JSON 数组格式。"""
    triplets = []
    with open(triplet_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # 尝试解析为 JSON 数组
        try:
            data = json.loads(content)
            if isinstance(data, list):
                triplets = data
                print(f"  ✓ 检测到 JSON 数组格式，包含 {len(triplets)} 条 triplets")
                return triplets
            elif isinstance(data, dict):
                triplets = [data]
                print(f"  ✓ 检测到 JSON 对象格式，1 条 triplet")
                return triplets
        except json.JSONDecodeError:
            pass
        
        # 如果不是有效的 JSON，尝试按行解析（JSON Lines 格式）
        f.seek(0)
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                triplets.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第 {line_num} 行解析失败: {e}")
                print(f"     内容: {line[:100]}...")
                continue
    
    print(f"  ✓ 检测到 JSON Lines 格式，包含 {len(triplets)} 条 triplets")
    return triplets


def compute_chunk_similarity(
    content1: str,
    content2: str,
    embedder_model,
) -> float:
    """计算两个 chunk 之间的 embedding 相似度。"""
    try:
        emb1 = embedder_model.encode([content1])[0]
        emb2 = embedder_model.encode([content2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        return 0.0


def analyze_triplets(
    triplet_file: str,
    chunk_file: str,
    model_type: str = "large",
) -> Dict[str, Any]:
    """
    分析 triplet 数据中 positive/negative chunk 的差异。

    返回
    -------
    Dict[str, Any]
        统计信息
    """
    print("=" * 80)
    print("【Triplet 质量分析】- Positive/Negative Chunk 差异统计")
    print("=" * 80)

    print("\n[1/4] 加载数据...")
    triplets = load_triplets(triplet_file)
    print(f"  ✓ 加载 {len(triplets)} 条 triplets")

    chunks = load_chunks(chunk_file)
    print(f"  ✓ 加载 {len(chunks)} 条 chunks")

    print("\n[2/4] 加载嵌入模型...")
    embedder_config = get_embedder(model_type=model_type)
    embedder_model = embedder_config["load"]()
    print(f"  ✓ 嵌入模型加载完成")

    print("\n[3/4] 计算差异...")
    stats = {
        "total_triplets": len(triplets),
        "valid_triplets": 0,
        "missing_chunks": 0,
        "chunk_similarities": [],
        "query_pos_similarities": [],
        "query_neg_similarities": [],
        "similarity_diffs": [],
        "score_margins": [],
    }

    for i, triplet in enumerate(triplets, 1):
        pos_id = triplet.get("positive_chunk_id")
        neg_id = triplet.get("negative_chunk_id")

        if pos_id not in chunks or neg_id not in chunks:
            stats["missing_chunks"] += 1
            continue

        pos_chunk = chunks[pos_id]
        neg_chunk = chunks[neg_id]

        pos_content = pos_chunk.get("content", "")
        neg_content = neg_chunk.get("content", "")
        query = triplet.get("query", "")

        if not (pos_content and neg_content and query):
            stats["missing_chunks"] += 1
            continue

        # 计算相似度
        pos_neg_similarity = compute_chunk_similarity(pos_content, neg_content, embedder_model)
        query_pos_similarity = compute_chunk_similarity(query, pos_content, embedder_model)
        query_neg_similarity = compute_chunk_similarity(query, neg_content, embedder_model)
        similarity_diff = query_pos_similarity - query_neg_similarity

        stats["chunk_similarities"].append(pos_neg_similarity)
        stats["query_pos_similarities"].append(query_pos_similarity)
        stats["query_neg_similarities"].append(query_neg_similarity)
        stats["similarity_diffs"].append(similarity_diff)
        stats["score_margins"].append(triplet.get("score_margin", 0))

        stats["valid_triplets"] += 1

        if i % max(1, len(triplets) // 10) == 0 or i == len(triplets):
            print(f"  [{i:4d}/{len(triplets):4d}] ✓")

    print("\n[4/4] 生成统计报告...")

    # 生成报告
    print("\n" + "=" * 80)
    print("【统计数据】")
    print("=" * 80)

    print(f"\n✓ 有效 triplet 数: {stats['valid_triplets']} / {stats['total_triplets']}")
    print(f"❌ 缺失数据: {stats['missing_chunks']}")

    if stats["valid_triplets"] == 0:
        print("❌ 没有有效的 triplet，无法进行分析")
        return stats

    # 关键指标
    chunk_sim_arr = np.array(stats["chunk_similarities"])
    query_pos_sim_arr = np.array(stats["query_pos_similarities"])
    query_neg_sim_arr = np.array(stats["query_neg_similarities"])
    sim_diff_arr = np.array(stats["similarity_diffs"])
    margin_arr = np.array(stats["score_margins"])

    print("\n" + "-" * 80)
    print("1️⃣  Positive 和 Negative Chunk 之间的相似度 (最关键)")
    print("-" * 80)
    print(f"   均值:       {chunk_sim_arr.mean():.4f}")
    print(f"   中位数:     {np.median(chunk_sim_arr):.4f}")
    print(f"   标准差:     {chunk_sim_arr.std():.4f}")
    print(f"   最小值:     {chunk_sim_arr.min():.4f}")
    print(f"   最大值:     {chunk_sim_arr.max():.4f}")
    print(f"   Q1 (25%):   {np.percentile(chunk_sim_arr, 25):.4f}")
    print(f"   Q3 (75%):   {np.percentile(chunk_sim_arr, 75):.4f}")

    print("\n" + "-" * 80)
    print("2️⃣  Query 与 Positive Chunk 相似度")
    print("-" * 80)
    print(f"   均值:       {query_pos_sim_arr.mean():.4f}")
    print(f"   中位数:     {np.median(query_pos_sim_arr):.4f}")
    print(f"   标准差:     {query_pos_sim_arr.std():.4f}")
    print(f"   最小值:     {query_pos_sim_arr.min():.4f}")
    print(f"   最大值:     {query_pos_sim_arr.max():.4f}")

    print("\n" + "-" * 80)
    print("3️⃣  Query 与 Negative Chunk 相似度")
    print("-" * 80)
    print(f"   均值:       {query_neg_sim_arr.mean():.4f}")
    print(f"   中位数:     {np.median(query_neg_sim_arr):.4f}")
    print(f"   标准差:     {query_neg_sim_arr.std():.4f}")
    print(f"   最小值:     {query_neg_sim_arr.min():.4f}")
    print(f"   最大值:     {query_neg_sim_arr.max():.4f}")

    print("\n" + "-" * 80)
    print("4️⃣  Query-Positive 与 Query-Negative 相似度差异 (越大越好)")
    print("-" * 80)
    print(f"   均值:       {sim_diff_arr.mean():.4f}  ← 平均区分度")
    print(f"   中位数:     {np.median(sim_diff_arr):.4f}")
    print(f"   标准差:     {sim_diff_arr.std():.4f}")
    print(f"   最小值:     {sim_diff_arr.min():.4f}")
    print(f"   最大值:     {sim_diff_arr.max():.4f}")
    print(f"   ≥ 0.05的:   {np.sum(sim_diff_arr >= 0.05)} ({100*np.sum(sim_diff_arr >= 0.05)/len(sim_diff_arr):.1f}%)")
    print(f"   ≥ 0.10的:   {np.sum(sim_diff_arr >= 0.10)} ({100*np.sum(sim_diff_arr >= 0.10)/len(sim_diff_arr):.1f}%)")
    print(f"   ≥ 0.15的:   {np.sum(sim_diff_arr >= 0.15)} ({100*np.sum(sim_diff_arr >= 0.15)/len(sim_diff_arr):.1f}%)")

    print("\n" + "-" * 80)
    print("5️⃣  Reranker Score Margin (Score 差异)")
    print("-" * 80)
    print(f"   均值:       {margin_arr.mean():.4f}")
    print(f"   中位数:     {np.median(margin_arr):.4f}")
    print(f"   标准差:     {margin_arr.std():.4f}")
    print(f"   最小值:     {margin_arr.min():.4f}")
    print(f"   最大值:     {margin_arr.max():.4f}")

    # 质量评分
    print("\n" + "=" * 80)
    print("【质量评估】")
    print("=" * 80)

    quality_score = 0
    max_score = 100

    # Criterion 1: Chunk similarity
    if chunk_sim_arr.mean() < 0.90:
        print(f"\n✅ Pos/Neg Chunk 差异度好")
        print(f"   平均相似度: {chunk_sim_arr.mean():.4f} (理想 < 0.90)")
        quality_score += 30
    else:
        print(f"\n⚠️  Pos/Neg Chunk 太相似")
        print(f"   平均相似度: {chunk_sim_arr.mean():.4f} (应该 < 0.90)")

    # Criterion 2: Similarity diff
    if sim_diff_arr.mean() > 0.10:
        print(f"\n✅ Query 区分度好")
        print(f"   平均差异: {sim_diff_arr.mean():.4f} (理想 > 0.10)")
        quality_score += 30
    elif sim_diff_arr.mean() > 0.05:
        print(f"\n⚠️  Query 区分度一般")
        print(f"   平均差异: {sim_diff_arr.mean():.4f} (应该 > 0.10)")
        quality_score += 15
    else:
        print(f"\n❌ Query 区分度差")
        print(f"   平均差异: {sim_diff_arr.mean():.4f} (应该 > 0.10)")

    # Criterion 3: Score margin
    if margin_arr.mean() > 0.15:
        print(f"\n✅ Reranker Score 差异足够大")
        print(f"   平均 margin: {margin_arr.mean():.4f} (理想 > 0.15)")
        quality_score += 25
    elif margin_arr.mean() > 0.10:
        print(f"\n⚠️  Reranker Score 差异一般")
        print(f"   平均 margin: {margin_arr.mean():.4f} (应该 > 0.15)")
        quality_score += 12
    else:
        print(f"\n⚠️  Reranker Score 差异较小")
        print(f"   平均 margin: {margin_arr.mean():.4f} (应该 > 0.15)")
        quality_score += 5

    # Criterion 4: Consistency
    if chunk_sim_arr.std() < 0.20:
        print(f"\n✅ 数据一致性好")
        print(f"   标准差: {chunk_sim_arr.std():.4f} (理想 < 0.20)")
        quality_score += 15
    else:
        print(f"\n⚠️  数据一致性差异大")
        print(f"   标准差: {chunk_sim_arr.std():.4f} (应该 < 0.20)")

    print("\n" + "-" * 80)
    print(f"【综合评分】{quality_score}/{max_score} 分")
    print("-" * 80)
    if quality_score >= 80:
        print("🌟 优秀 - 可直接用于微调")
    elif quality_score >= 60:
        print("✅ 良好 - 适合微调，但可考虑优化参数")
    elif quality_score >= 40:
        print("⚠️  一般 - 建议调整筛选参数")
    else:
        print("❌ 需改进 - 建议重新生成或调整参数")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Triplet 数据质量分析")
    parser.add_argument("--triplet-data", required=True, help="Triplet 数据文件")
    parser.add_argument("--chunk-data", required=True, help="Chunk 数据文件")
    parser.add_argument("--model-type", default="large", choices=["large", "small"])
    args = parser.parse_args()

    # 验证文件存在
    if not Path(args.triplet_data).exists():
        print(f"❌ Triplet 文件不存在: {args.triplet_data}")
        sys.exit(1)
    if not Path(args.chunk_data).exists():
        print(f"❌ Chunk 文件不存在: {args.chunk_data}")
        sys.exit(1)

    analyze_triplets(
        triplet_file=args.triplet_data,
        chunk_file=args.chunk_data,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()