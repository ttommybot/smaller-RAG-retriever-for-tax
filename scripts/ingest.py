# -*- coding: utf-8 -*-
"""
数据入库脚本

本脚本将完整的 RAG 数据处理管道：
1. 从 data/raw 加载文档
2. 使用 chunker 进行分块
3. 使用 preprocess 进行文本清洗
4. 使用 embedder 生成向量
5. 将向量保存到向量数据库

使用方法：
    # 在项目根目录运行
    python scripts/ingest.py
"""

import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loading.loader import load_documents_from_dir
from chunking.chunker import sliding_window_chunking, get_chunking_config
from chunking.preprocess import preprocess_chunks
from embedding.embedder import get_embedder


def ingest(
    data_dir: str = None,
    chunking_strategy: str = "sliding_window",
    model_type: str = "large",
    batch_size: int = 32,
    save_path: str = None
):
    """
    执行完整的 RAG 数据入库流程。

    参数
    ----------
    data_dir : str, optional
        原始数据目录，默认为配置中的 raw_data_dir。
    chunking_strategy : str, optional
        分块策略，可选 "sliding_window" 或 "semantic"，默认为 "sliding_window"。
    model_type : str, optional
        embedding 模型类型，可选 "large"、"small"、"student"，默认为 "large"。
    batch_size : int, optional
        embedding 批量大小，默认为 32。
    save_path : str, optional
        向量数据库保存路径，默认为配置中的 vector_db_dir。
    """
    import yaml

    # 加载配置
    config_path = PROJECT_ROOT / "configs" / "configs.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 使用配置中的默认路径
    if data_dir is None:
        data_dir = config.get('paths', {}).get('raw_data_dir', 'data/raw')
    if save_path is None:
        save_path = config.get('paths', {}).get('vector_db_dir', 'vectordb')

    print("=" * 60)
    print("RAG 数据入库流程")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  原始数据目录：{data_dir}")
    print(f"  向量数据库路径：{save_path}")
    print(f"  分块策略：{chunking_strategy}")
    print(f"  Embedding 模型：{model_type}")
    print(f"  Batch size: {batch_size}")

    # ==================== 步骤 1: 加载文档 ====================
    print("\n" + "=" * 60)
    print("步骤 1: 加载文档")
    print("=" * 60)
    documents = load_documents_from_dir(directory=str(PROJECT_ROOT / data_dir))

    if not documents:
        print("错误：未加载到任何文档")
        return

    print(f"成功加载 {len(documents)} 个文档")

    # ==================== 步骤 2: 分块 ====================
    print("\n" + "=" * 60)
    print("步骤 2: 文档分块")
    print("=" * 60)

    if chunking_strategy == "sliding_window":
        chunk_config = get_chunking_config()
        chunks = sliding_window_chunking(
            documents,
            window_size=chunk_config['chunk_size'],
            step_size=chunk_config['chunk_size'] - chunk_config['chunk_overlap'],
            min_chunk=chunk_config['min_chunk']
        )
    elif chunking_strategy == "semantic":
        from chunking.chunker import raw_data_semantic_chunking
        chunks = raw_data_semantic_chunking(documents)
    else:
        raise ValueError(f"未知的分块策略：{chunking_strategy}")

    print(f"分块后共 {len(chunks)} 个文本块")

    # ==================== 步骤 3: 文本预处理 ====================
    print("\n" + "=" * 60)
    print("步骤 3: 文本预处理")
    print("=" * 60)
    clean_chunks = preprocess_chunks(
        chunks,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True
    )
    print(f"预处理后剩余 {len(clean_chunks)} 个文本块")

    # ==================== 步骤 4: 生成 Embedding ====================
    print("\n" + "=" * 60)
    print("步骤 4: 生成 Embedding 向量")
    print("=" * 60)
    embedder = get_embedder(model_type)
    model = embedder['load']()

    # 批量生成 embedding
    all_embeddings = []
    total_batches = (len(clean_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(clean_chunks), batch_size):
        batch_chunks = clean_chunks[i:i + batch_size]
        batch_texts = [chunk['content'] for chunk in batch_chunks]
        batch_num = i // batch_size + 1

        print(f"  处理批次 {batch_num}/{total_batches}...", end="\r")
        embeddings = embedder['embed_texts'](batch_texts, normalize=True)
        all_embeddings.append(embeddings)

    import numpy as np
    all_embeddings = np.vstack(all_embeddings)
    print(f"Embedding 生成完成！形状：{all_embeddings.shape}")

    # ==================== 步骤 5: 保存向量数据库 ====================
    print("\n" + "=" * 60)
    print("步骤 5: 保存向量数据库")
    print("=" * 60)

    # 创建保存目录
    vector_db_path = PROJECT_ROOT / save_path
    vector_db_path.mkdir(parents=True, exist_ok=True)

    # 保存向量和元数据
    import pickle

    # 保存向量
    embeddings_path = vector_db_path / f"embeddings_{model_type}.npy"
    np.save(embeddings_path, all_embeddings)
    print(f"向量已保存：{embeddings_path}")

    # 保存 chunk 元数据
    metadata_path = vector_db_path / f"metadata_{model_type}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(clean_chunks, f)
    print(f"元数据已保存：{metadata_path}")

    # 保存配置信息
    info = {
        'model_type': model_type,
        'chunking_strategy': chunking_strategy,
        'chunk_config': get_chunking_config() if chunking_strategy == 'sliding_window' else None,
        'num_chunks': len(clean_chunks),
        'embedding_dim': all_embeddings.shape[1],
        'data_dir': data_dir
    }
    info_path = vector_db_path / f"info_{model_type}.json"
    import json
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"配置信息已保存：{info_path}")

    print("\n" + "=" * 60)
    print("入库流程完成")
    print("=" * 60)
    print(f"最终结果:")
    print(f"  - 文档数量：{len(documents)}")
    print(f"  - Chunk 数量：{len(clean_chunks)}")
    print(f"  - 向量维度：{all_embeddings.shape[1]}")
    print(f"  - 保存路径：{vector_db_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 数据入库脚本")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="原始数据目录，默认为配置中的 data/raw"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["sliding_window", "semantic"],
        default="sliding_window",
        help="分块策略"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["large", "small", "student"],
        default="large",
        help="Embedding 模型类型"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding 批量大小"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="向量数据库保存路径，默认为配置中的 vectordb"
    )

    args = parser.parse_args()

    ingest(
        data_dir=args.data_dir,
        chunking_strategy=args.strategy,
        model_type=args.model,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
