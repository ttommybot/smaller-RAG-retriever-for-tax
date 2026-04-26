import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Set, Optional
from pathlib import Path
from dataclasses import dataclass

# ==========================================
# 配置区
# ==========================================
CONFIG = {
    "eval_data_dir": Path(__file__).parent.parent / "data" / "evaluations",
    "k_list": [1, 3, 5, 10],
    "chunk_types": ["semantic", "sliding"],
    "gt_file_template": "evaluation_criteria_{chunk_type}.jsonl",
    "result_file_pattern": "result_",
    # 基准模型名称（用于对比）
    "baseline_model": "large_reordered",
    # 显著提升阈值
    "significant_improvement_threshold": 0.05,  # 5%
}

# ==========================================
# 数据类：用于存储完整的评估结果（质量+效率）
# ==========================================
@dataclass
class EvaluationResult:
    model_name: str
    chunk_type: str
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
    
    # 效率指标（预留接口，需要在生成检索结果时记录）
    avg_retrieval_latency_ms: float = 0.0
    avg_rerank_latency_ms: float = 0.0
    avg_end_to_end_latency_ms: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    qps: float = 0.0

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

def extract_chunk_ids_from_top_chunks(top_chunks: List[Dict]) -> List[str]:
    return [chunk["chunk_id"] for chunk in top_chunks]

# ==========================================
# 单模型评估函数
# ==========================================
def evaluate_single_model(
    gt_file_path: Path,
    result_file_path: Path,
    model_name: str,
    chunk_type: str
) -> Optional[EvaluationResult]:
    # 1. 加载Ground Truth
    gt_data = load_jsonl(gt_file_path)
    gt_dict = {}
    for item in gt_data:
        query = item["query"].strip()
        gt_chunk_ids = set(extract_chunk_ids_from_top_chunks(item["top_chunks"]))
        gt_dict[query] = gt_chunk_ids

    # 2. 加载模型检索结果
    result_data = load_jsonl(result_file_path)
    result_list = []
    efficiency_data_available = False
    
    for item in result_data:
        query = item["query"].strip()
        if query in gt_dict:
            retrieved_chunk_ids = extract_chunk_ids_from_top_chunks(item["top_chunks"])
            result_item = {
                "query": query,
                "gt_chunk_ids": gt_dict[query],
                "retrieved_chunk_ids": retrieved_chunk_ids
            }
            # 尝试提取效率数据（如果结果文件里有）
            if "efficiency" in item:
                result_item["efficiency"] = item["efficiency"]
                efficiency_data_available = True
            result_list.append(result_item)

    if len(result_list) == 0:
        print(f"❌ 错误：{model_name} ({chunk_type}) 没有匹配到任何query")
        return None

    # 3. 计算质量指标
    all_quality_metrics = {f"{m}@{k}": [] for m in ["Recall", "HitRate", "MRR", "nDCG"] for k in CONFIG["k_list"]}
    all_efficiency_metrics = {"retrieval_latency": [], "rerank_latency": [], "end_to_end_latency": []}
    
    for item in tqdm(result_list, desc=f"评估 {model_name} ({chunk_type})"):
        metrics, _ = calculate_quality_metrics(
            gt_chunk_ids=item["gt_chunk_ids"],
            retrieved_chunk_ids=item["retrieved_chunk_ids"],
            k_list=CONFIG["k_list"]
        )
        for key, val in metrics.items():
            all_quality_metrics[key].append(val)
        
        # 收集效率数据（如果有）
        if efficiency_data_available and "efficiency" in item:
            eff = item["efficiency"]
            if "retrieval_latency_ms" in eff:
                all_efficiency_metrics["retrieval_latency"].append(eff["retrieval_latency_ms"])
            if "rerank_latency_ms" in eff:
                all_efficiency_metrics["rerank_latency"].append(eff["rerank_latency_ms"])
            if "end_to_end_latency_ms" in eff:
                all_efficiency_metrics["end_to_end_latency"].append(eff["end_to_end_latency_ms"])

    # 4. 构建结果对象
    result = EvaluationResult(
        model_name=model_name,
        chunk_type=chunk_type,
        total_queries=len(result_list)
    )

    # 填充质量指标
    print(f"\n📊 【{model_name} ({chunk_type})】质量指标报告：")
    for key in sorted(all_quality_metrics.keys()):
        avg_val = np.mean(all_quality_metrics[key])
        # 动态设置属性
        attr_name = key.replace("@", "_at_").lower()
        setattr(result, attr_name, round(avg_val, 4))
        print(f"  {key:15} : {avg_val:.4f}")

    # 填充效率指标（如果有）
    if efficiency_data_available and len(all_efficiency_metrics["retrieval_latency"]) > 0:
        print(f"\n⚡ 【{model_name} ({chunk_type})】效率指标报告：")
        result.avg_retrieval_latency_ms = round(np.mean(all_efficiency_metrics["retrieval_latency"]), 2)
        result.avg_rerank_latency_ms = round(np.mean(all_efficiency_metrics["rerank_latency"]), 2) if all_efficiency_metrics["rerank_latency"] else 0.0
        result.avg_end_to_end_latency_ms = round(np.mean(all_efficiency_metrics["end_to_end_latency"]), 2) if all_efficiency_metrics["end_to_end_latency"] else 0.0
        
        print(f"  平均检索延迟: {result.avg_retrieval_latency_ms} ms")
        if result.avg_rerank_latency_ms > 0:
            print(f"  平均重排序延迟: {result.avg_rerank_latency_ms} ms")
        if result.avg_end_to_end_latency_ms > 0:
            print(f"  平均端到端延迟: {result.avg_end_to_end_latency_ms} ms")

    return result

# ==========================================
# 模型对比与有效性判断函数
# ==========================================
def compare_and_decide(results: List[EvaluationResult]) -> pd.DataFrame:
    """
    对比所有模型，判断微调是否有效
    """
    df = pd.DataFrame([vars(r) for r in results])
    
    # 按chunk_type分组对比
    for chunk_type in CONFIG["chunk_types"]:
        chunk_df = df[df["chunk_type"] == chunk_type].copy()
        if chunk_df.empty:
            continue
            
        # 找到基准模型
        baseline_row = chunk_df[chunk_df["model_name"] == CONFIG["baseline_model"]]
        if baseline_row.empty:
            print(f"\n⚠️ 警告：在 {chunk_type} 分块中未找到基准模型 {CONFIG['baseline_model']}，跳过对比")
            continue
            
        baseline = baseline_row.iloc[0]
        
        print(f"\n{'='*60}")
        print(f"【{chunk_type.upper()}】分块 - 模型对比与有效性判断")
        print(f"{'='*60}")
        print(f"基准模型：{CONFIG['baseline_model']}")
        print(f"显著提升阈值：> {CONFIG['significant_improvement_threshold']*100}%\n")
        
        # 核心质量指标（关注Recall@5, Recall@10, MRR@10, nDCG@10）
        core_metrics = ["recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]
        
        for _, row in chunk_df.iterrows():
            if row["model_name"] == CONFIG["baseline_model"]:
                continue
                
            print(f"--- 对比模型：{row['model_name']} ---")
            
            improvements = []
            all_improved = True
            any_degraded = False
            
            for metric in core_metrics:
                baseline_val = baseline[metric]
                current_val = row[metric]
                diff = current_val - baseline_val
                pct_change = (diff / baseline_val) * 100 if baseline_val > 0 else 0
                
                status = "✅ 提升" if diff > 0 else "❌ 下降" if diff < 0 else "➖ 持平"
                print(f"  {metric.replace('_at_', '@').upper():15} : {baseline_val:.4f} → {current_val:.4f} ({status} {pct_change:+.2f}%)")
                
                if diff > CONFIG["significant_improvement_threshold"]:
                    improvements.append(metric)
                elif diff < 0:
                    any_degraded = True
                    all_improved = False
                else:
                    all_improved = False
            
            # 决策
            print(f"\n  📋 决策结论：", end="")
            if all_improved and len(improvements) >= 2:
                print("🎉 **微调有效！所有核心指标显著提升，建议部署**")
            elif len(improvements) >= 1 and not any_degraded:
                print("👍 **微调有帮助**，部分指标提升，无下降，可考虑部署")
            elif len(improvements) >= 1 and any_degraded:
                print("⚠️ **需要权衡**，部分指标提升但部分下降，建议检查调参")
            elif not any_degraded:
                print("➖ **微调效果有限**，指标基本持平，建议继续调参")
            else:
                print("❌ **微调无效/过拟合**，指标下降，建议回滚到基准模型")
            print()

    return df

# ==========================================
# 批量评估主函数
# ==========================================
def batch_evaluate_all_models():
    all_results = []

    for chunk_type in CONFIG["chunk_types"]:
        print(f"\n{'='*60}")
        print(f"开始评估【{chunk_type}】分块类型的所有模型")
        print(f"{'='*60}")

        gt_file_name = CONFIG["gt_file_template"].format(chunk_type=chunk_type)
        gt_file_path = CONFIG["eval_data_dir"] / gt_file_name

        if not gt_file_path.exists():
            print(f"⚠️ GT文件 {gt_file_path} 不存在，跳过")
            continue

        for file_path in CONFIG["eval_data_dir"].iterdir():
            if file_path.is_file() and file_path.name.startswith(CONFIG["result_file_pattern"]) and chunk_type in file_path.name:
                model_name = file_path.stem.replace("result_", "")
                result = evaluate_single_model(
                    gt_file_path=gt_file_path,
                    result_file_path=file_path,
                    model_name=model_name,
                    chunk_type=chunk_type
                )
                if result:
                    all_results.append(result)

    if all_results:
        # 生成对比报告
        df = compare_and_decide(all_results)
        
        # 保存完整报告
        output_path = CONFIG["eval_data_dir"] / "RAG完整评估报告.xlsx"
        df.to_excel(output_path, index=False)
        print(f"\n✅ 所有评估完成！完整报告已保存到：{output_path}")

    return all_results

# ==========================================
# 执行
# ==========================================
if __name__ == "__main__":
    batch_evaluate_all_models()