# -*- coding: utf-8 -*-
"""
批量向量化入库脚本

扫描 models/ 目录，为所有模型（排除 reranker）构建向量库。
根据模型名称自动识别需要的 chunk 方法：
- 模型名包含 'semantic' -> 只构建 semantic
- 模型名包含 'sliding' -> 只构建 sliding
- 其他 -> 构建两种

使用方法：
    # 为所有模型构建向量库
    python scripts/embed_to_vectordb.py

    # 指定模型
    python scripts/embed_to_vectordb.py --models sentence-transformers--all-MiniLM-L6-v2

    # 强制重建（即使已存在）
    python scripts/embed_to_vectordb.py --force-rebuild

    # 不使用 GPU
    python scripts/embed_to_vectordb.py --no-cuda
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding.vectorstore import (
    build_vectorstore_for_custom_model,
    get_vectorstore_model_dir,
    parse_model_name,
    get_chunk_methods_for_model,
)
from embedding.embedder import get_custom_embedder


# ==========================================
# 配置
# ==========================================
MODELS_DIR = PROJECT_ROOT / "models"

# 排除的模型（reranker 不参与 embedding）
EXCLUDED_MODELS = ["BAAI--bge-reranker-v2-gemma"]


# ==========================================
# 获取待入库模型列表
# ==========================================
def get_models_to_embed() -> List[str]:
    """
    获取 models/ 目录下所有需要构建向量库的模型名称。
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

    print(f"发现 {len(models)} 个待入库模型")
    for i, m in enumerate(models):
        print(f"  [{i+1}] {m}")

    return sorted(models)


# ==========================================
# 单模型向量库构建
# ==========================================
def embed_single_model(
    model_name: str,
    chunk_method: str,
    batch_size: int = 32,
    force_rebuild: bool = False,
    use_cuda: bool = True
) -> Dict[str, Any]:
    """
    为单个模型构建向量库。

    参数
    ----------
    model_name : str
        模型名称。

    chunk_method : str
        chunk 方法 ('semantic' 或 'sliding')。

    batch_size : int
        embedding 批量大小。

    force_rebuild : bool
        是否强制重建。

    use_cuda : bool
        是否使用 GPU。

    返回
    -------
    Dict[str, Any]
        构建结果信息。
    """
    import torch

    result = {
        "model_name": model_name,
        "chunk_method": chunk_method,
        "success": False,
        "vectordb_path": "",
        "build_time_s": 0,
        "peak_memory_mb": 0,
        "num_chunks": 0,
        "embedding_dim": 0,
        "error": None
    }

    # 重置显存统计
    if use_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    print("\n" + "=" * 60)
    print(f"构建向量库：{model_name} ({chunk_method})")
    print("=" * 60)

    try:
        # 检查是否已存在
        vectorstore_dir = get_vectorstore_model_dir(model_name, chunk_method)
        embeddings_path = vectorstore_dir / "embeddings.npy"

        if embeddings_path.exists() and not force_rebuild:
            print(f"向量库已存在：{vectorstore_dir}")
            result["success"] = True
            result["vectordb_path"] = str(vectorstore_dir)
            result["build_time_s"] = 0  # 跳过构建
            return result

        # 构建向量库
        vectorstore = build_vectorstore_for_custom_model(
            model_name=model_name,
            chunk_method=chunk_method,
            batch_size=batch_size,
            force_rebuild=force_rebuild
        )

        end_time = time.time()
        result["success"] = True
        result["vectordb_path"] = str(vectorstore_dir)
        result["build_time_s"] = end_time - start_time
        result["num_chunks"] = len(vectorstore['chunks'])
        result["embedding_dim"] = vectorstore['embedding_dim']

        if use_cuda and torch.cuda.is_available():
            result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"\n构建完成：")
        print(f"  - Chunk 数量：{result['num_chunks']}")
        print(f"  - 向量维度：{result['embedding_dim']}")
        print(f"  - 构建时间：{result['build_time_s']:.2f} s")
        print(f"  - 显存占用：{result['peak_memory_mb']:.2f} MB")
        print(f"  - 保存路径：{result['vectordb_path']}")

    except Exception as e:
        result["error"] = str(e)
        print(f"❌ 构建失败：{e}")

    return result


# ==========================================
# 批量构建
# ==========================================
def embed_all_models(
    models: List[str],
    batch_size: int = 32,
    force_rebuild: bool = False,
    use_cuda: bool = True
) -> List[Dict[str, Any]]:
    """
    为所有模型构建向量库。

    参数
    ----------
    models : List[str]
        模型名称列表。

    batch_size : int
        embedding 批量大小。

    force_rebuild : bool
        是否强制重建。

    use_cuda : bool
        是否使用 GPU。

    返回
    -------
    List[Dict[str, Any]]
        所有构建结果。
    """
    all_results = []

    # 计算总任务数
    total_tasks = 0
    for model_name in models:
        chunk_methods = get_chunk_methods_for_model(model_name)
        total_tasks += len(chunk_methods)

    print(f"\n总入库任务数：{total_tasks}")
    print("=" * 60)

    current_task = 0
    for model_name in models:
        chunk_methods = get_chunk_methods_for_model(model_name)

        for chunk_method in chunk_methods:
            current_task += 1
            print(f"\n[{current_task}/{total_tasks}] {model_name} | {chunk_method}")

            result = embed_single_model(
                model_name=model_name,
                chunk_method=chunk_method,
                batch_size=batch_size,
                force_rebuild=force_rebuild,
                use_cuda=use_cuda
            )
            all_results.append(result)

    return all_results


# ==========================================
# 生成入库报告
# ==========================================
def save_embed_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    保存入库结果报告。

    参数
    ----------
    results : List[Dict[str, Any]]
        构建结果列表。
    output_path : Path
        输出文件路径。
    """
    import json

    # 保存 JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"入库报告已保存到：{json_path}")

    # 保存 Markdown 表格
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 向量库构建报告\n\n")
        f.write(f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("| Model | Chunk | 状态 | Chunks | 维度 | 时间(s) | 显存(MB) | 路径 |\n")
        f.write("|-------|-------|------|--------|------|---------|----------|------|\n")

        for r in results:
            status = "✅" if r['success'] else "❌"
            if r['success'] and r['build_time_s'] == 0:
                status = "⏭️ (已存在)"

            f.write(f"| {r['model_name']} | {r['chunk_method']} | {status} | ")
            f.write(f"{r['num_chunks']} | {r['embedding_dim']} | ")
            f.write(f"{r['build_time_s']:.2f} | {r['peak_memory_mb']:.2f} | ")
            f.write(f"vectordb/ |\n")

        # 失败列表
        failed = [r for r in results if not r['success']]
        if failed:
            f.write("\n## 失败详情\n\n")
            for r in failed:
                f.write(f"- **{r['model_name']} ({r['chunk_method']})**: {r['error']}\n")

    print(f"Markdown 表格已保存到：{md_path}")


# ==========================================
# 主函数
# ==========================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="批量向量化入库脚本")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="指定要入库的模型名称（默认处理所有）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding 批量大小，默认 32"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重建向量库（即使已存在）"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="不使用 GPU"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vectordb_build_report",
        help="输出文件名（保存到项目根目录）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("批量向量化入库")
    print("=" * 60)

    # 获取模型列表
    if args.models:
        models = args.models
        print(f"指定入库模型：{models}")
    else:
        models = get_models_to_embed()

    if not models:
        print("❌ 没有可处理的模型")
        return

    # 执行入库
    print(f"\n开始构建向量库...")
    results = embed_all_models(
        models=models,
        batch_size=args.batch_size,
        force_rebuild=args.force_rebuild,
        use_cuda=not args.no_cuda
    )

    # 保存报告
    output_path = PROJECT_ROOT / args.output
    save_embed_report(results, output_path)

    # 统计
    success_count = sum(1 for r in results if r['success'])
    skipped_count = sum(1 for r in results if r['success'] and r['build_time_s'] == 0)
    failed_count = sum(1 for r in results if not r['success'])

    print("\n" + "=" * 60)
    print("入库完成")
    print("=" * 60)
    print(f"模型数：{len(models)}")
    print(f"总任务数：{len(results)}")
    print(f"成功构建：{success_count - skipped_count}")
    print(f"跳过（已存在）：{skipped_count}")
    print(f"失败：{failed_count}")
    print(f"报告保存到：{PROJECT_ROOT}")


if __name__ == "__main__":
    main()