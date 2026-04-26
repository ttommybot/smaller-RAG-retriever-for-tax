# -*- coding: utf-8 -*-
"""
补充入库脚本

为指定的 LoRA 模型构建向量库。

使用方法：
    python scripts/addition_ingestion.py
"""

import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding.vectorstore import (
    build_vectorstore_for_custom_model,
    get_vectorstore_model_dir,
    get_chunk_methods_for_model,
)

# 待入库的模型列表
MODELS_TO_INGEST = [
    "sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3",
    "sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4",
    "sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5",
    "sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3",
    "sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4",
    "sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="补充入库脚本 - LoRA 模型")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding 批量大小，默认 32"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重建向量库"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="不使用 GPU"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("补充入库 - LoRA 模型")
    print("=" * 60)
    print(f"模型数量：{len(MODELS_TO_INGEST)}")

    # 计算总任务数
    total_tasks = 0
    for model_name in MODELS_TO_INGEST:
        chunk_methods = get_chunk_methods_for_model(model_name)
        total_tasks += len(chunk_methods)

    print(f"总任务数：{total_tasks}")

    current_task = 0
    success_count = 0
    failed_count = 0

    for model_name in MODELS_TO_INGEST:
        chunk_methods = get_chunk_methods_for_model(model_name)

        for chunk_method in chunk_methods:
            current_task += 1
            print(f"\n[{current_task}/{total_tasks}] {model_name} | {chunk_method}")

            # 检查是否已存在
            vectorstore_dir = get_vectorstore_model_dir(model_name, chunk_method)
            embeddings_path = vectorstore_dir / "embeddings.npy"

            if embeddings_path.exists() and not args.force_rebuild:
                print(f"  ⏭️ 向量库已存在，跳过：{vectorstore_dir}")
                success_count += 1
                continue

            try:
                print(f"  构建向量库...")
                vectorstore = build_vectorstore_for_custom_model(
                    model_name=model_name,
                    chunk_method=chunk_method,
                    batch_size=args.batch_size,
                    force_rebuild=args.force_rebuild
                )
                print(f"  ✅ 成功：{len(vectorstore['chunks'])} chunks, {vectorstore['embedding_dim']} 维")
                success_count += 1
            except Exception as e:
                print(f"  ❌ 失败：{e}")
                failed_count += 1

    print("\n" + "=" * 60)
    print("补充入库完成")
    print("=" * 60)
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")

    if failed_count > 0:
        print("\n⚠️ 失败的模型需要检查 LoRA 权重是否正确合并")


if __name__ == "__main__":
    main()