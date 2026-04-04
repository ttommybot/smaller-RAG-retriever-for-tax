# -*- coding: utf-8 -*-
"""
向量数据库模块

本模块提供向量库的构建、加载和检索功能。
- build_vectorstore(): 从原始文档构建向量库
- load_vectorstore(): 从磁盘加载向量库
- search(): 基于查询向量检索相似文档

向量库存储在 vectordb 目录下。
"""

import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from embedder import get_embedder, _load_config


# ==================== 向量库目录管理 ====================

def get_vectorstore_dir() -> Path:
    """
    获取向量库存储目录。

    从 configs/configs.yaml 的 paths.vector_db_dir 读取路径。

    返回
    ----
    Path
        向量库目录的 Path 对象。
    """
    config = _load_config()
    vector_db_dir = config.get('paths', {}).get('vector_db_dir', 'vectordb')

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dir_path = project_root / vector_db_dir

    return dir_path


# ==================== 构建向量库 ====================

def build_vectorstore(
    chunks: List[Dict[str, Any]],
    model_type: str = "large",
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    从 chunk 列表构建向量库。

    使用指定的 embedding 模型将文本块转换为向量，并保存到 vectordb 目录。

    参数
    ----------
    chunks : List[Dict[str, Any]]
        由 chunker 和 preprocess 处理后的 chunk 列表。
        每个 chunk 应包含 'id'、'content' 和 'metadata' 字段。

    model_type : str, optional
        embedding 模型类型，可选 'large'、'small'、'student'，默认为 'large'。

    batch_size : int, optional
        embedding 批量大小，默认为 32。

    save_path : str, optional
        向量库保存路径，默认为配置中的 vectordb 目录。

    返回
    -------
    Dict[str, Any]
        向量库信息，包括：
        - 'embeddings': np.ndarray, 向量矩阵
        - 'chunks': List[Dict], chunk 元数据列表
        - 'model_type': str, 使用的模型类型
        - 'embedding_dim': int, 向量维度
    """
    print("=" * 60)
    print("构建向量库")
    print("=" * 60)

    # 获取 embedder
    embedder = get_embedder(model_type)
    model = embedder['load']()

    # 提取文本内容
    texts = [chunk['content'] for chunk in chunks]

    # 批量生成 embedding
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"\n开始生成 embedding，共 {len(texts)} 个文本块...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"  处理批次 {batch_num}/{total_batches}...", end="\r")
        embeddings = embedder['embed_texts'](batch_texts, normalize=True)
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    print(f"\nEmbedding 生成完成！形状：{all_embeddings.shape}")

    # 确定保存路径
    if save_path is None:
        save_dir = get_vectorstore_dir()
    else:
        save_dir = Path(save_path)

    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存向量
    embeddings_path = save_dir / f"embeddings_{model_type}.npy"
    np.save(embeddings_path, all_embeddings)
    print(f"向量已保存：{embeddings_path}")

    # 保存 chunk 元数据
    metadata_path = save_dir / f"metadata_{model_type}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"元数据已保存：{metadata_path}")

    # 保存配置信息
    info = {
        'model_type': model_type,
        'num_chunks': len(chunks),
        'embedding_dim': all_embeddings.shape[1],
        'batch_size': batch_size
    }
    info_path = save_dir / f"info_{model_type}.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"配置信息已保存：{info_path}")

    print("\n" + "=" * 60)
    print("向量库构建完成")
    print("=" * 60)
    print(f"  - Chunk 数量：{len(chunks)}")
    print(f"  - 向量维度：{all_embeddings.shape[1]}")
    print(f"  - 保存路径：{save_dir}")

    return {
        'embeddings': all_embeddings,
        'chunks': chunks,
        'model_type': model_type,
        'embedding_dim': all_embeddings.shape[1],
        'save_dir': save_dir
    }


# ==================== 加载向量库 ====================

def load_vectorstore(
    model_type: str = "large",
    vectorstore_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    从磁盘加载向量库。

    从 vectordb 目录加载已保存的向量和元数据。

    参数
    ----------
    model_type : str, optional
        embedding 模型类型，可选 'large'、'small'、'student'，默认为 'large'。

    vectorstore_dir : str, optional
        向量库目录路径，默认为配置中的 vectordb 目录。

    返回
    -------
    Dict[str, Any]
        向量库信息，包括：
        - 'embeddings': np.ndarray, 向量矩阵
        - 'chunks': List[Dict], chunk 元数据列表
        - 'info': Dict, 配置信息
        - 'model_type': str, 使用的模型类型

    Raises
    ------
    FileNotFoundError
        如果指定的向量库文件不存在。
    """
    print("=" * 60)
    print("加载向量库")
    print("=" * 60)

    # 确定加载路径
    if vectorstore_dir is None:
        load_dir = get_vectorstore_dir()
    else:
        load_dir = Path(vectorstore_dir)

    # 加载向量
    embeddings_path = load_dir / f"embeddings_{model_type}.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"向量文件不存在：{embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"向量已加载：{embeddings_path}, 形状：{embeddings.shape}")

    # 加载元数据
    metadata_path = load_dir / f"metadata_{model_type}.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"元数据文件不存在：{metadata_path}")
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"元数据已加载：{metadata_path}, 共 {len(chunks)} 个 chunk")

    # 加载配置信息
    info_path = load_dir / f"info_{model_type}.json"
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        print(f"配置信息已加载：{info_path}")
    else:
        info = {
            'model_type': model_type,
            'num_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1]
        }
        print("未找到配置文件，使用默认信息")

    print("\n" + "=" * 60)
    print("向量库加载完成")
    print("=" * 60)
    print(f"  - Chunk 数量：{len(chunks)}")
    print(f"  - 向量维度：{embeddings.shape[1]}")
    print(f"  - 模型类型：{model_type}")

    return {
        'embeddings': embeddings,
        'chunks': chunks,
        'info': info,
        'model_type': model_type,
        'load_dir': load_dir
    }


# ==================== 向量检索 ====================

def search(
    query: str,
    vectorstore: Dict[str, Any],
    top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """
    基于查询文本检索最相似的 chunk。

    使用余弦相似度计算查询向量与向量库中所有向量的相似度。

    参数
    ----------
    query : str
        查询文本。

    vectorstore : Dict[str, Any]
        向量库信息，由 load_vectorstore() 返回。
        应包含 'embeddings' 和 'chunks' 字段。

    top_k : int, optional
        返回最相似的 k 个结果，默认为 5。

    返回
    -------
    List[Tuple[Dict[str, Any], float]]
        (chunk, similarity_score) 列表，按相似度降序排列。
        每个 chunk 包含原始的 id、content 和 metadata。
        相似度分数范围 [-1, 1]，越大越相似。
    """
    # 获取 embedding 模型
    model_type = vectorstore.get('model_type', 'large')
    embedder = get_embedder(model_type)

    # 生成查询向量
    query_vector = embedder['embed_query'](query, normalize=True)
    query_vector = query_vector.reshape(1, -1)

    # 获取向量库中的向量
    embeddings = vectorstore['embeddings']
    chunks = vectorstore['chunks']

    # 计算余弦相似度（向量已归一化，点积即余弦相似度）
    similarities = np.dot(embeddings, query_vector.T).flatten()

    # 获取 top_k 索引
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 返回结果
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        score = float(similarities[idx])
        results.append((chunk, score))

    return results


def search_by_vector(
    query_vector: np.ndarray,
    vectorstore: Dict[str, Any],
    top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """
    基于查询向量检索最相似的 chunk。

    参数
    ----------
    query_vector : np.ndarray
        查询向量，形状应为 (embedding_dim,) 或 (1, embedding_dim)。

    vectorstore : Dict[str, Any]
        向量库信息，由 load_vectorstore() 返回。

    top_k : int, optional
        返回最相似的 k 个结果，默认为 5。

    返回
    -------
    List[Tuple[Dict[str, Any], float]]
        (chunk, similarity_score) 列表，按相似度降序排列。
    """
    # 确保向量形状正确
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # 获取向量库中的向量
    embeddings = vectorstore['embeddings']
    chunks = vectorstore['chunks']

    # 计算余弦相似度
    similarities = np.dot(embeddings, query_vector.T).flatten()

    # 获取 top_k 索引
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 返回结果
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        score = float(similarities[idx])
        results.append((chunk, score))

    return results


if __name__ == "__main__":
    import sys

    # 添加 src 目录到路径
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from loading.loader import load_documents_from_dir
    from chunking.chunker import sliding_window_chunking, get_chunking_config
    from chunking.preprocess import preprocess_chunks

    print("=" * 60)
    print("向量库模块测试")
    print("=" * 60)

    # 测试：从原始文档构建向量库
    print("\n" + "=" * 60)
    print("测试 1: 构建向量库（使用前 10 个文档）")
    print("=" * 60)

    # 1. 加载少量文档测试
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents_from_dir(directory=str(data_dir))[:10]
    print(f"加载 {len(documents)} 个文档")

    # 2. 分块
    config = get_chunking_config()
    chunks = sliding_window_chunking(
        documents,
        window_size=config['chunk_size'],
        step_size=config['chunk_size'] - config['chunk_overlap'],
        min_chunk=config['min_chunk']
    )
    print(f"分块后 {len(chunks)} 个块")

    # 3. 预处理
    clean_chunks = preprocess_chunks(
        chunks,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True
    )
    print(f"预处理后 {len(clean_chunks)} 个块")

    # 4. 构建向量库（使用 small 模型加快测试）
    vectorstore = build_vectorstore(
        clean_chunks,
        model_type="small",
        batch_size=16
    )

    # 测试：加载向量库
    print("\n" + "=" * 60)
    print("测试 2: 加载向量库")
    print("=" * 60)
    loaded_vectorstore = load_vectorstore(model_type="small")

    # 测试：检索
    print("\n" + "=" * 60)
    print("测试 3: 检索功能")
    print("=" * 60)

    test_queries = [
        "增值税是什么？",
        "企业所得税如何计算？",
        "个人所得税有哪些扣除项目？"
    ]

    for query in test_queries:
        print(f"\n查询：{query}")
        results = search(query, loaded_vectorstore, top_k=3)

        for i, (chunk, score) in enumerate(results, 1):
            print(f"  [{i}] 分数：{score:.4f}")
            print(f"      ID: {chunk['id']}")
            print(f"      内容：{chunk['content'][:80]}...")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
