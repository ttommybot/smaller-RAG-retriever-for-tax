# -*- coding: utf-8 -*-
"""
文档分块脚本

执行两种分块方法（语义分块 + 滑动窗口），结果保存到 data/processed 目录。

使用方法：
    # 在项目根目录运行
    python scripts/chunk_to_processed.py
"""

import sys
from pathlib import Path
from typing import Optional

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loading.loader import load_documents_from_dir
from chunking.chunker import sliding_window_chunking, raw_data_semantic_chunking, get_chunking_config
from chunking.preprocess import preprocess_chunks


def chunk_to_processed(data_dir: Optional[str] = None) -> None:
    """
    执行完整的分块流程：loading -> chunking -> preprocessing。

    两种分块方法的结果分别保存到 data/processed：
    - chunks_semantic.json / chunks_semantic_cleaned.json
    - chunks_sliding.json / chunks_sliding_cleaned.json

    参数
    ----------
    data_dir : str, optional
        原始数据目录，默认为配置中的 data/raw。
    """
    import yaml

    # 加载配置
    config_path = PROJECT_ROOT / "configs" / "configs.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolved_data_dir: str = data_dir or config.get("paths", {}).get("raw_data_dir", "data/raw")

    print("=" * 60)
    print("文档分块流程")
    print("=" * 60)
    print(f"\n原始数据目录：{resolved_data_dir}")

    # ==================== 步骤 1: Loading ====================
    print("\n" + "=" * 60)
    print("步骤 1: Loading - 加载文档")
    print("=" * 60)
    documents = load_documents_from_dir(directory=str(PROJECT_ROOT / resolved_data_dir))

    if not documents:
        print("错误：未加载到任何文档")
        return

    print(f"成功加载 {len(documents)} 个文档")

    # ==================== 步骤 2: Chunking ====================
    print("\n" + "=" * 60)
    print("步骤 2: Chunking - 文档分块")
    print("=" * 60)

    # 语义分块
    print("使用语义分块...")
    chunks_semantic = raw_data_semantic_chunking(
        documents,
        save_to_file=True,
        output_file="chunks_semantic.json"
    )

    # 滑动窗口分块
    print("\n使用滑动窗口分块...")
    chunk_config = get_chunking_config()
    chunks_sliding = sliding_window_chunking(
        documents,
        window_size=chunk_config["chunk_size"],
        step_size=chunk_config["chunk_size"] - chunk_config["chunk_overlap"],
        min_chunk=chunk_config["min_chunk"],
        save_to_file=True,
        output_file="chunks_sliding.json"
    )

    print(f"\n语义分块：{len(chunks_semantic)} 个块")
    print(f"滑动窗口：{len(chunks_sliding)} 个块")

    # ==================== 步骤 3: Preprocessing ====================
    print("\n" + "=" * 60)
    print("步骤 3: Preprocessing - 文本预处理")
    print("=" * 60)

    # 预处理语义分块结果
    print("预处理语义分块结果...")
    clean_chunks_semantic = preprocess_chunks(
        chunks_semantic,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True,
        save_to_file=True,
        output_file="chunks_semantic_cleaned.json"
    )

    # 预处理滑动窗口分块结果
    print("\n预处理滑动窗口分块结果...")
    clean_chunks_sliding = preprocess_chunks(
        chunks_sliding,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True,
        save_to_file=True,
        output_file="chunks_sliding_cleaned.json"
    )

    print(f"\n语义分块预处理后：{len(clean_chunks_semantic)} 个块")
    print(f"滑动窗口预处理后：{len(clean_chunks_sliding)} 个块")

    print("\n" + "=" * 60)
    print("分块流程完成")
    print("=" * 60)
    print("输出文件:")
    print("  - data/processed/chunks_semantic.json")
    print("  - data/processed/chunks_semantic_cleaned.json")
    print("  - data/processed/chunks_sliding.json")
    print("  - data/processed/chunks_sliding_cleaned.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文档分块脚本")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="原始数据目录，默认为配置中的 data/raw"
    )

    args = parser.parse_args()

    chunk_to_processed(data_dir=args.data_dir)