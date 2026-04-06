# -*- coding: utf-8 -*-
"""
入库流水线模块

本模块执行完整的 RAG 数据入库流程：
1. 从 raw_data_dir 加载文档
2. 使用 chunker 进行分块
3. 使用 preprocess 进行文本清洗
4. 使用 embedder 生成向量
5. 构建并保存向量库
"""

from pathlib import Path
import sys

# 添加 src 目录到导入路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from loading.loader import load_documents_from_dir
from chunking.chunker import sliding_window_chunking, get_chunking_config
from chunking.preprocess import preprocess_chunks
from embedding.embedder import get_embedder
from embedding.vectorstore import build_vectorstore
from utils.config_loader import load_config


def run_ingestion_pipeline(config_path: str) -> int:
    """
    执行完整的数据入库流水线。

    参数
    ----------
    config_path : str
        配置文件路径。

    返回
    -------
    int
        处理的文档数量。
    """
    # 加载配置
    config = load_config(config_path)

    print(f"从以下路径加载配置：{config_path}")
    print(f"原始数据目录：{config['paths']['raw_data_dir']}")
    print(f"处理数据目录：{config['paths']['processed_data_dir']}")
    print(f"向量数据库目录：{config['paths']['vector_db_dir']}")

    # 步骤 1: 从原始数据目录加载文档
    print("\n[1/5] 从原始数据目录加载文档...")
    raw_dir = Path(current_dir.parent.parent) / config['paths']['raw_data_dir']
    documents = load_documents_from_dir(str(raw_dir))
    print(f"已加载 {len(documents)} 个文档")

    # 步骤 2: 文档分块
    print("\n[2/5] 文档分块...")
    chunk_config = get_chunking_config()
    chunks = sliding_window_chunking(
        documents,
        window_size=chunk_config['chunk_size'],
        step_size=chunk_config['chunk_size'] - chunk_config['chunk_overlap'],
        min_chunk=chunk_config['min_chunk']
    )
    print(f"已创建 {len(chunks)} 个文本块")

    # 步骤 3: 文本预处理
    print("\n[3/5] 文本预处理...")
    clean_chunks = preprocess_chunks(
        chunks,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True
    )
    print(f"预处理后剩余 {len(clean_chunks)} 个文本块")

    # 步骤 4: 加载 embedding 模型并生成向量
    print("\n[4/5] 生成 embedding 向量...")
    model_type = "large"  # 默认使用 large 模型
    embedder = get_embedder(model_type)
    model = embedder['load']()
    print(f"已加载 embedding 模型")

    # 步骤 5: 构建并保存向量库
    print("\n[5/5] 构建并保存向量库...")
    vector_db_dir = Path(current_dir.parent.parent) / config['paths']['vector_db_dir']

    vectorstore = build_vectorstore(
        clean_chunks,
        model_type=model_type,
        batch_size=32,
        save_path=str(vector_db_dir)
    )
    print(f"向量库已保存到：{vector_db_dir}")

    print(f"\n✓ 入库流水线完成!")
    print(f"  - 文档数：{len(documents)}")
    print(f"  - 文本块数：{len(clean_chunks)}")
    print(f"  - 向量维度：{vectorstore['embedding_dim']}")

    return len(documents)


if __name__ == "__main__":
    # 测试运行
    config_path = str(Path(__file__).parent.parent.parent / "configs" / "configs.yaml")
    run_ingestion_pipeline(config_path)
