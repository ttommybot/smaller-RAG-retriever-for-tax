# -*- coding: utf-8 -*-
"""
批量评估结果生成脚本

本脚本：
1. 扫描 models/ 目录获取所有待评估模型
2. 对每个模型运行评估（所有 chunk 方法 × 是否重排）
3. 生成对比表格保存到项目根目录

注意：向量库必须已存在（由 embed_to_vectordb.py 构建）
如果向量库不存在，评估会失败。

评估对比目标：
- 微调是否有效？minilm vs 微调后minilm (有无reranker)
- 微调后小模型能否替代大模型？微调后minilm vs bge-large (有无reranker)
- 微调后小模型能否接近大模型+reranker？微调后minilm vs bge-large+reranker
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluations.run_evaluation import (
    run_single_evaluation,
    load_queries,
    EvaluationResult,
    save_results_table,
    CONFIG as EVAL_CONFIG,
)
from embedding.vectorstore import get_chunk_methods_for_model, get_vectorstore_model_dir


# ==========================================
# 配置
# ==========================================
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT
RESULTS_FILE_NAME = "evaluation_results"

# 排除的模型（reranker 不参与 embedding）
EXCLUDED_MODELS = ["BAAI--bge-reranker-v2-gemma"]


# ==========================================
# 获取待评估模型列表
# ==========================================
def get_models_to_evaluate() -> List[str]:
    """
    获取 models/ 目录下所有待评估的模型名称。
    排除 reranker 模型。

    返回
    -------
    List[str]
        模型名称列表。
    """
    models = []

    if not MODELS_DIR.exists():
        print(f"❌ 模型目录不存在：{MODELS_DIR}")
        return models

    for item in MODELS_DIR.iterdir():
        if item.is_dir() and item.name != ".gitkeep":
            # 排除 reranker
            if item.name in EXCLUDED_MODELS:
                print(f"  排除模型（reranker）：{item.name}")
                continue
            models.append(item.name)

    print(f"发现 {len(models)} 个待评估模型")
    for i, m in enumerate(models):
        print(f"  [{i+1}] {m}")

    return sorted(models)


def check_vectordb_exists(model_name: str, chunk_method: str) -> bool:
    """
    检查向量库是否存在。

    参数
    ----------
    model_name : str
        模型名称。
    chunk_method : str
        chunk 方法。

    返回
    -------
    bool
        是否存在。
    """
    vectorstore_dir = get_vectorstore_model_dir(model_name, chunk_method)
    embeddings_path = vectorstore_dir / "embeddings.npy"
    return embeddings_path.exists()


# ==========================================
# 运行所有评估
# ==========================================
def run_all_evaluations(
    models: List[str],
    use_cuda: bool = True,
    skip_missing_vectordb: bool = True
) -> List[EvaluationResult]:
    """
    对所有模型运行评估。
    假设向量库已存在（由 embed_to_vectordb.py 构建）。

    参数
    ----------
    models : List[str]
        模型名称列表。

    use_cuda : bool
        是否使用 GPU。

    skip_missing_vectordb : bool
        是否跳过向量库不存在的模型。

    返回
    -------
    List[EvaluationResult]
        所有评估结果。
    """
    all_results = []

    # 加载问题
    queries = load_queries(EVAL_CONFIG["eval_queries_file"])
    print(f"评估问题数量：{len(queries)}")

    # 计算总配置数
    total_configs = 0
    for model_name in models:
        chunk_methods = get_chunk_methods_for_model(model_name)
        for chunk_method in chunk_methods:
            if check_vectordb_exists(model_name, chunk_method) or not skip_missing_vectordb:
                total_configs += 2  # × 2 (有无 reranker)

    print(f"\n总评估配置数：{total_configs}")
    print("=" * 60)

    current_config = 0
    for model_name in models:
        chunk_methods = get_chunk_methods_for_model(model_name)

        for chunk_method in chunk_methods:
            # 检查向量库是否存在
            if skip_missing_vectordb and not check_vectordb_exists(model_name, chunk_method):
                print(f"\n⚠️ 向量库不存在，跳过：{model_name} ({chunk_method})")
                print(f"  请先运行：python scripts/embed_to_vectordb.py --models {model_name}")
                continue

            for use_reranker in [False, True]:
                current_config += 1
                print(f"\n[{current_config}/{total_configs}] {model_name} | {chunk_method} | reranker={use_reranker}")

                try:
                    result = run_single_evaluation(
                        model_name=model_name,
                        chunk_method=chunk_method,
                        use_reranker=use_reranker,
                        queries=queries,
                        use_cuda=use_cuda,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ 评估失败：{e}")
                    # 记录失败结果
                    failed_result = EvaluationResult(
                        model_name=model_name,
                        chunk_type=chunk_method,
                        use_reranker=use_reranker,
                        total_queries=0
                    )
                    all_results.append(failed_result)

    return all_results


# ==========================================
# 生成对比分析表格
# ==========================================
def generate_comparison_table(results: List[EvaluationResult]) -> str:
    """
    生成详细对比分析表格（Markdown 格式）。

    包含三个对比维度：
    1. 微调有效性对比
    2. 小模型替代大模型对比
    3. 小模型接近大模型+reranker对比
    """
    import pandas as pd

    df = pd.DataFrame([vars(r) for r in results])

    md_content = f"""# RAG 模型评估对比报告

**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、完整评估结果

### 质量指标

| Model | Chunk | Reranker | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|-------|-------|----------|----------|-----------|--------|----------|
"""

    # 添加所有结果
    for r in results:
        md_content += f"| {r.model_name} | {r.chunk_type} | {r.use_reranker} | "
        md_content += f"{r.recall_at_5:.4f} | {r.recall_at_10:.4f} | {r.mrr_at_10:.4f} | {r.ndcg_at_10:.4f} |\n"

    md_content += """
### 效率指标

| Model | Chunk | Reranker | 检索(ms) | 重排(ms) | 端到端(ms) | 显存(MB) |
|-------|-------|----------|----------|----------|------------|----------|
"""

    for r in results:
        md_content += f"| {r.model_name} | {r.chunk_type} | {r.use_reranker} | "
        md_content += f"{r.avg_retrieval_latency_ms:.2f} | {r.avg_rerank_latency_ms:.2f} | "
        md_content += f"{r.avg_end_to_end_latency_ms:.2f} | {r.peak_gpu_memory_mb:.2f} |\n"

    md_content += """
---

## 二、对比分析

### 1. 微调是否有效？

对比：MiniLM vs 微调后 MiniLM（有无 reranker）

"""

    # 提取 MiniLM 相关结果
    minilm_base = df[df['model_name'] == 'sentence-transformers--all-MiniLM-L6-v2']
    minilm_fft = df[df['model_name'].str.contains('FFT', na=False)]
    minilm_lora = df[df['model_name'].str.contains('LoRA', na=False)]

    # 按 chunk 和 reranker 分组对比
    for chunk in ['semantic', 'sliding']:
        md_content += f"\n#### {chunk.upper()} 分块\n\n"
        md_content += "| 模型 | Reranker | Recall@5 | Recall@10 | 相比基线提升 |\n"
        md_content += "|------|----------|----------|-----------|---------------|\n"

        # 基线（MiniLM base）
        for use_reranker in [False, True]:
            base_row = minilm_base[
                (minilm_base['chunk_type'] == chunk) &
                (minilm_base['use_reranker'] == use_reranker)
            ]
            if len(base_row) > 0:
                base_recall_5 = base_row.iloc[0]['recall_at_5']
                base_recall_10 = base_row.iloc[0]['recall_at_10']

                # FFT 模型
                fft_rows = minilm_fft[
                    (minilm_fft['chunk_type'] == chunk) &
                    (minilm_fft['use_reranker'] == use_reranker)
                ]
                for _, row in fft_rows.iterrows():
                    improvement = (row['recall_at_5'] - base_recall_5) / base_recall_5 * 100 if base_recall_5 > 0 else 0
                    md_content += f"| {row['model_name']} | {use_reranker} | "
                    md_content += f"{row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | "
                    md_content += f"{improvement:+.2f}% |\n"

                # LoRA 模型
                lora_rows = minilm_lora[
                    (minilm_lora['chunk_type'] == chunk) &
                    (minilm_lora['use_reranker'] == use_reranker)
                ]
                for _, row in lora_rows.iterrows():
                    improvement = (row['recall_at_5'] - base_recall_5) / base_recall_5 * 100 if base_recall_5 > 0 else 0
                    md_content += f"| {row['model_name']} | {use_reranker} | "
                    md_content += f"{row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | "
                    md_content += f"{improvement:+.2f}% |\n"

                # 基线
                md_content += f"| MiniLM (基线) | {use_reranker} | {base_recall_5:.4f} | {base_recall_10:.4f} | -- |\n"

    md_content += """
### 2. 微调后小模型能否替代大模型？

对比：微调后 MiniLM vs BGE-Large（有无 reranker）

"""

    # 提取 BGE-Large 结果
    bge_large = df[df['model_name'] == 'BAAI--bge-large-zh-v1.5']

    for chunk in ['semantic', 'sliding']:
        md_content += f"\n#### {chunk.upper()} 分块\n\n"
        md_content += "| 模型 | Reranker | Recall@5 | Recall@10 | 相比 BGE-Large |\n"
        md_content += "|------|----------|----------|-----------|----------------|\n"

        # BGE-Large 基线
        for use_reranker in [False, True]:
            bge_row = bge_large[
                (bge_large['chunk_type'] == chunk) &
                (bge_large['use_reranker'] == use_reranker)
            ]
            if len(bge_row) > 0:
                bge_recall_5 = bge_row.iloc[0]['recall_at_5']
                bge_recall_10 = bge_row.iloc[0]['recall_at_10']

                # 微调 MiniLM
                finetuned = df[
                    (df['model_name'].str.contains('FFT|LoRA', na=False)) &
                    (df['chunk_type'] == chunk) &
                    (df['use_reranker'] == use_reranker)
                ]
                for _, row in finetuned.iterrows():
                    diff = (row['recall_at_5'] - bge_recall_5) / bge_recall_5 * 100 if bge_recall_5 > 0 else 0
                    md_content += f"| {row['model_name']} | {use_reranker} | "
                    md_content += f"{row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | "
                    md_content += f"{diff:+.2f}% |\n"

                md_content += f"| BGE-Large (基线) | {use_reranker} | {bge_recall_5:.4f} | {bge_recall_10:.4f} | -- |\n"

    md_content += """
### 3. 微调后小模型能否接近大模型+reranker？

对比：微调后 MiniLM（无 reranker）vs BGE-Large + Reranker

"""

    bge_with_reranker = bge_large[bge_large['use_reranker'] == True]

    for chunk in ['semantic', 'sliding']:
        md_content += f"\n#### {chunk.upper()} 分块\n\n"
        md_content += "| 模型 | Reranker | Recall@5 | Recall@10 | 相比 BGE-Large+Reranker |\n"
        md_content += "|------|----------|----------|-----------|------------------------|\n"

        bge_rr_row = bge_with_reranker[bge_with_reranker['chunk_type'] == chunk]
        if len(bge_rr_row) > 0:
            bge_rr_recall_5 = bge_rr_row.iloc[0]['recall_at_5']
            bge_rr_recall_10 = bge_rr_row.iloc[0]['recall_at_10']

            # 微调 MiniLM（无 reranker）
            finetuned_no_rr = df[
                (df['model_name'].str.contains('FFT|LoRA', na=False)) &
                (df['chunk_type'] == chunk) &
                (df['use_reranker'] == False)
            ]
            for _, row in finetuned_no_rr.iterrows():
                diff = (row['recall_at_5'] - bge_rr_recall_5) / bge_rr_recall_5 * 100 if bge_rr_recall_5 > 0 else 0
                md_content += f"| {row['model_name']} | No | "
                md_content += f"{row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | "
                md_content += f"{diff:+.2f}% |\n"

            md_content += f"| BGE-Large | Yes (基准) | {bge_rr_recall_5:.4f} | {bge_rr_recall_10:.4f} | -- |\n"

    md_content += """
---

## 三、结论与建议

根据以上对比分析，可以得出以下结论：

1. **微调有效性**：观察 FFT/LoRA 模型相比 MiniLM 基线的提升幅度
2. **替代可能性**：观察微调后 MiniLM 与 BGE-Large 的差距
3. **效率权衡**：对比显存占用和推理延迟

建议选择综合表现最优的模型配置进行部署。
"""

    return md_content


# ==========================================
# 主函数
# ==========================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 模型批量评估脚本")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="指定要评估的模型名称（默认评估所有）"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="不使用 GPU"
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="跳过向量库不存在的模型（默认启用）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="输出文件名（保存到项目根目录）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAG 模型批量评估")
    print("=" * 60)
    print("注意：向量库必须已存在（由 embed_to_vectordb.py 构建）")

    # 获取模型列表
    if args.models:
        models = args.models
        print(f"指定评估模型：{models}")
    else:
        models = get_models_to_evaluate()

    if not models:
        print("❌ 没有可评估的模型")
        return

    # 运行评估
    print(f"\n开始评估...")
    results = run_all_evaluations(
        models=models,
        use_cuda=not args.no_cuda,
        skip_missing_vectordb=args.skip_missing
    )

    if not results:
        print("❌ 没有评估结果")
        return

    # 保存结果
    output_path = OUTPUT_DIR / args.output
    save_results_table(results, output_path)

    # 生成对比分析
    comparison_md = generate_comparison_table(results)
    comparison_path = OUTPUT_DIR / f"{args.output}_comparison.md"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(comparison_md)
    print(f"\n对比分析已保存到：{comparison_path}")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
    print(f"评估模型数：{len(models)}")
    print(f"评估配置数：{len(results)}")
    print(f"结果保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()