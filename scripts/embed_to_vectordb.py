# -*- coding: utf-8 -*-
"""
向量化入库脚本

从 data/processed 目录读取已分块并清洗好的 JSON 文件，进行向量化并存入向量库。
支持三种模型（student、small、large），每次运行只入库一个模型。

使用方法：
    # 使用 large 模型，语义分块
    python scripts/embed_to_vectordb.py --model large --chunk-method semantic

    # 使用 small 模型，滑动窗口分块
    python scripts/embed_to_vectordb.py --model small --chunk-method sliding_window
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding.vectorstore import build_vectorstore


def load_chunks_from_json(chunk_method: str) -> List[Dict[str, Any]]:
    """
    从 data/processed 目录加载已清洗的 chunk 数据。

    参数
    ----------
    chunk_method : str
        分块方法，'semantic' 或 'sliding_window'。

    返回
    -------
    List[Dict[str, Any]]
        chunk 列表。
    """
    import json

    # chunk_method 到文件名的映射
    file_map = {
        "semantic": "semantic",
        "sliding_window": "sliding"
    }

    if chunk_method not in file_map:
        raise ValueError(f"未知的 chunk_method: {chunk_method}，可选值：{list(file_map.keys())}")

    chunks_file = f"chunks_{file_map[chunk_method]}_cleaned.json"
    chunks_path = PROJECT_ROOT / "data" / "processed" / chunks_file

    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_to_vectordb(
    chunk_method: str = "semantic",
    model_type: str = "large",
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> None:
    """
    执行向量化入库流程：embedding -> vectorstore。

    参数
    ----------
    chunk_method : str, optional
        分块方法，'semantic' 或 'sliding_window'，默认 'semantic'。
    model_type : str, optional
        embedding 模型类型，'large'、'small' 或 'student'，默认 'large'。
    batch_size : int, optional
        embedding 批量大小，默认 32。
    save_path : str, optional
        向量库保存路径，默认为配置中的 vectordb。
    """
    import yaml

    # 加载配置
    config_path = PROJECT_ROOT / "configs" / "configs.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolved_save_path: str = save_path or config.get("paths", {}).get("vector_db_dir", "vectordb")
    resolved_save_dir = PROJECT_ROOT / resolved_save_path

    print("=" * 60)
    print("向量化入库流程")
    print("=" * 60)
    print("\n配置:")
    print(f"  分块方法：{chunk_method}")
    print(f"  Embedding 模型：{model_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  向量库保存路径：{resolved_save_path}")

    # ==================== 步骤 1: 加载已处理的 chunk ====================
    print("\n" + "=" * 60)
    print("步骤 1: 加载已处理的 chunk")
    print("=" * 60)

    chunks = load_chunks_from_json(chunk_method)

    if not chunks:
        print("错误：未加载到任何 chunk")
        return

    print(f"成功加载 {len(chunks)} 个文本块")

    # ==================== 步骤 2: Embedding + Vectorstore ====================
    print("\n" + "=" * 60)
    print("步骤 2: Embedding + Vectorstore - 生成向量并入库")
    print("=" * 60)

    vectorstore = build_vectorstore(
        chunks,
        chunk_method=chunk_method,
        model_type=model_type,
        batch_size=batch_size,
        save_path=str(resolved_save_dir)
    )

    print("\n" + "=" * 60)
    print("入库流程完成")
    print("=" * 60)
    print("结果:")
    print(f"  - Chunk 数量：{len(chunks)}")
    print(f"  - 向量维度：{vectorstore['embedding_dim']}")
    print(f"  - 保存路径：{resolved_save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="向量化入库脚本")
    parser.add_argument(
        "--chunk-method",
        type=str,
        choices=["semantic", "sliding_window"],
        default="semantic",
        help="分块方法，默认 semantic"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["large", "small", "student"],
        default="large",
        help="Embedding 模型类型，默认 large"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding 批量大小，默认 32"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="向量数据库保存路径，默认为配置中的 vectordb"
    )

    args = parser.parse_args()

    embed_to_vectordb(
        chunk_method=args.chunk_method,
        model_type=args.model,
        batch_size=args.batch_size,
        save_path=args.save_path
    )