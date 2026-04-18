# -*- coding: utf-8 -*-
"""
向量数据库模块

本模块提供向量库的构建、加载和检索功能。

六种组合方案 (chunk_method × model_type):
- chunk_method: 'semantic' | 'sliding_window' (2 种)
- model_type: 'large' | 'small' | 'student' (3 种)

组合结果 (6 种):
1. semantic + large      (默认推荐)
2. semantic + small
3. semantic + student
4. sliding_window + large
5. sliding_window + small
6. sliding_window + student

每套向量独立存储，文件名格式：{chunk_method}_{model_type}.*
"""

import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from .embedder import get_embedder, _load_config


# ==================== 向量库目录管理 ====================

def get_vectorstore_dir() -> Path:
    """
    获取向量库存储目录。

    返回
    ----
    Path
        向量库目录的 Path 对象。
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    config = _load_config()
    vector_db_dir = config.get('paths', {}).get('vector_db_dir', 'vectordb')

    return project_root / vector_db_dir


# ==================== 构建向量库 ====================

def build_vectorstore(
    chunks: List[Dict[str, Any]],
    chunk_method: str = "semantic",
    model_type: str = "large",
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    从 chunk 列表构建向量库。

    参数
    ----------
    chunks : List[Dict[str, Any]]
        由 chunker 处理后的 chunk 列表。

    chunk_method : str, optional
        分块方法，可选 'semantic' | 'sliding_window'，默认为 'semantic'。

    model_type : str, optional
        embedding 模型类型，可选 'large' | 'small' | 'student'，默认为 'large'。
        - 'large': BGE-large-zh-v1.5, 1024 维，高质量检索（默认）
        - 'small': all-MiniLM-L6-v2, 384 维，平衡性能
        - 'student': all-MiniLM-L6-v2, 384 维，快速推理

    batch_size : int, optional
        embedding 批量大小，默认为 32。

    save_path : str, optional
        向量库保存路径，默认为配置中的 vectordb 目录。

    返回
    -------
    Dict[str, Any]
        向量库信息。
    """
    print("=" * 60)
    print("构建向量库")
    print("=" * 60)

    # 验证参数
    if chunk_method not in ['semantic', 'sliding_window']:
        raise ValueError(f"不支持的分块方法：{chunk_method}，可选值：['semantic', 'sliding_window']")
    if model_type not in ['large', 'small', 'student']:
        raise ValueError(f"不支持的模型类型：{model_type}，可选值：['large', 'small', 'student']")

    # 获取 embedder
    embedder = get_embedder(model_type)

    # 提取文本内容
    texts = [chunk['content'] for chunk in chunks]

    # 批量生成 embedding
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"\n开始生成 embedding ({chunk_method} + {model_type})，共 {len(texts)} 个文本块...")
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
    prefix = f"{chunk_method}_{model_type}"
    embeddings_path = save_dir / f"embeddings_{prefix}.npy"
    np.save(embeddings_path, all_embeddings)
    print(f"向量已保存：{embeddings_path}")

    # 保存 chunk 元数据
    metadata_path = save_dir / f"metadata_{prefix}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"元数据已保存：{metadata_path}")

    # 保存配置信息
    info = {
        'chunk_method': chunk_method,
        'model_type': model_type,
        'num_chunks': len(chunks),
        'embedding_dim': all_embeddings.shape[1],
        'batch_size': batch_size
    }
    info_path = save_dir / f"info_{prefix}.json"
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
        'chunk_method': chunk_method,
        'model_type': model_type,
        'embedding_dim': all_embeddings.shape[1],
        'save_dir': save_dir
    }


def build_all_vectorstores(
    chunks_semantic: List[Dict[str, Any]],
    chunks_sliding: List[Dict[str, Any]],
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    构建全部 6 套向量库（2 种 chunk × 3 种 model）。

    返回
    -------
    Dict[str, Dict[str, Any]]
        嵌套字典：
        {
            'semantic': {'large': {...}, 'small': {...}, 'student': {...}},
            'sliding_window': {'large': {...}, 'small': {...}, 'student': {...}}
        }
    """
    print("=" * 60)
    print("构建全部 6 套向量库")
    print("=" * 60)

    results = {
        'semantic': {},
        'sliding_window': {}
    }

    total = 0
    for chunk_method in ['semantic', 'sliding_window']:
        chunks = chunks_semantic if chunk_method == 'semantic' else chunks_sliding
        for model_type in ['large', 'small', 'student']:
            total += 1
            print(f"\n[{total}/6] {chunk_method} + {model_type}...")
            results[chunk_method][model_type] = build_vectorstore(
                chunks, chunk_method=chunk_method, model_type=model_type,
                batch_size=batch_size, save_path=save_path
            )

    print("\n" + "=" * 60)
    print("全部 6 套向量库构建完成")
    print("=" * 60)

    return results


# ==================== 加载向量库 ====================

def load_vectorstore(
    chunk_method: str = "semantic",
    model_type: str = "large",
    vectorstore_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    从磁盘加载向量库。

    参数
    ----------
    chunk_method : str, optional
        分块方法，可选 'semantic' | 'sliding_window'，默认为 'semantic'。

    model_type : str, optional
        embedding 模型类型，可选 'large' | 'small' | 'student'，默认为 'large'。

    vectorstore_dir : str, optional
        向量库目录路径。

    返回
    -------
    Dict[str, Any]
        向量库信息。

    Raises
    ------
    FileNotFoundError
        如果指定的向量库文件不存在。
    """
    if chunk_method not in ['semantic', 'sliding_window']:
        raise ValueError(f"不支持的分块方法：{chunk_method}")
    if model_type not in ['large', 'small', 'student']:
        raise ValueError(f"不支持的模型类型：{model_type}")

    print("=" * 60)
    print("加载向量库")
    print("=" * 60)

    if vectorstore_dir is None:
        load_dir = get_vectorstore_dir()
    else:
        load_dir = Path(vectorstore_dir)

    prefix = f"{chunk_method}_{model_type}"

    # 加载向量
    embeddings_path = load_dir / f"embeddings_{prefix}.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"向量文件不存在：{embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"向量已加载：{embeddings_path}, 形状：{embeddings.shape}")

    # 加载元数据
    metadata_path = load_dir / f"metadata_{prefix}.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"元数据文件不存在：{metadata_path}")
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"元数据已加载：{metadata_path}, 共 {len(chunks)} 个 chunk")

    # 加载配置信息
    info_path = load_dir / f"info_{prefix}.json"
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    else:
        info = {
            'chunk_method': chunk_method,
            'model_type': model_type,
            'num_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1]
        }

    print("\n" + "=" * 60)
    print("向量库加载完成")
    print("=" * 60)
    print(f"  - Chunk 数量：{len(chunks)}")
    print(f"  - 向量维度：{embeddings.shape[1]}")
    print(f"  - 组合方案：{chunk_method} + {model_type}")

    return {
        'embeddings': embeddings,
        'chunks': chunks,
        'chunk_method': chunk_method,
        'model_type': model_type,
        'embedding_dim': embeddings.shape[1],
        'load_dir': load_dir
    }


def load_all_vectorstores(
    vectorstore_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    加载全部 6 套向量库。

    返回
    -------
    Dict[str, Dict[str, Any]]
        包含 6 套向量库的嵌套字典。
    """
    print("=" * 60)
    print("加载全部 6 套向量库")
    print("=" * 60)

    results = {
        'semantic': {},
        'sliding_window': {}
    }

    for chunk_method in ['semantic', 'sliding_window']:
        for model_type in ['large', 'small', 'student']:
            print(f"\n加载 {chunk_method} + {model_type}...")
            results[chunk_method][model_type] = load_vectorstore(
                chunk_method=chunk_method, model_type=model_type, vectorstore_dir=vectorstore_dir
            )

    return results


# ==================== 向量检索 ====================

def search(
    query: str,
    vectorstore: Dict[str, Any],
    top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """
    基于查询文本检索最相似的 chunk。

    参数
    ----------
    query : str
        查询文本。

    vectorstore : Dict[str, Any]
        向量库信息。

    top_k : int, optional
        返回最相似的 k 个结果，默认为 5。

    返回
    -------
    List[Tuple[Dict[str, Any], float]]
        (chunk, similarity_score) 列表。
    """
    model_type = vectorstore.get('model_type', 'large')
    embedder = get_embedder(model_type)

    query = str(query).strip()
    query_vector = embedder['embed_query'](query, normalize=True)
    query_vector = query_vector.reshape(1, -1)

    embeddings = vectorstore['embeddings']
    chunks = vectorstore['chunks']

    # 余弦相似度（向量已归一化）
    similarities = np.dot(embeddings, query_vector.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        score = float(similarities[idx])
        results.append((chunk, score))

    return results


def search_from_config(
    query: str,
    chunk_method: str = "semantic",
    model_type: str = "large",
    top_k: int = 5,
    vectorstore_dir: Optional[str] = None
) -> List[Tuple[Dict[str, Any], float]]:
    """
    便捷检索：自动加载向量库并检索。

    参数
    ----------
    query : str
        查询文本。

    chunk_method : str, optional
        分块方法，默认为 'semantic'。

    model_type : str, optional
        模型类型，默认为 'large'。

    top_k : int, optional
        返回结果数量，默认为 5。

    vectorstore_dir : str, optional
        向量库目录。

    返回
    -------
    List[Tuple[Dict[str, Any], float]]
        (chunk, score) 列表。
    """
    vectorstore = load_vectorstore(
        chunk_method=chunk_method, model_type=model_type, vectorstore_dir=vectorstore_dir
    )
    return search(query, vectorstore, top_k)


# ==================== 检索器接口 ====================

def get_searcher(
    chunk_method: str = "semantic",
    model_type: str = "large",
    vectorstore: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    获取检索器接口。

    参数
    ----------
    chunk_method : str, optional
        分块方法，默认为 'semantic'。

    model_type : str, optional
        模型类型，默认为 'large'。

    vectorstore : Optional[Dict[str, Any]], optional
        预加载的向量库。

    返回
    -------
    Dict[str, Any]
        包含 search 函数的字典。
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(chunk_method=chunk_method, model_type=model_type)

    return {
        'search': lambda q, k=5: search(q, vectorstore, k),
        'vectorstore': vectorstore,
        'chunk_method': chunk_method,
        'model_type': model_type
    }


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from loading.loader import load_documents_from_dir
    from chunking.chunker import sliding_window_chunking, raw_data_semantic_chunking, get_chunking_config
    from chunking.preprocess import preprocess_chunks

    print("=" * 60)
    print("向量库模块测试 - 6 种组合方案")
    print("=" * 60)

    # 加载少量文档测试
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents_from_dir(directory=str(data_dir))[:3]
    print(f"加载 {len(documents)} 个文档")

    # 语义分块
    print("\n" + "=" * 60)
    print("语义分块...")
    print("=" * 60)
    chunks_semantic = raw_data_semantic_chunking(documents)
    chunks_semantic = preprocess_chunks(chunks_semantic)

    # 滑动窗口分块
    print("\n" + "=" * 60)
    print("滑动窗口分块...")
    print("=" * 60)
    config = get_chunking_config()
    chunks_sliding = sliding_window_chunking(
        documents,
        window_size=config['chunk_size'],
        step_size=config['chunk_size'] - config['chunk_overlap'],
        min_chunk=config['min_chunk']
    )
    chunks_sliding = preprocess_chunks(chunks_sliding)

    print(f"\nsemantic: {len(chunks_semantic)} chunks")
    print(f"sliding_window: {len(chunks_sliding)} chunks")

    # 构建 6 套向量库
    all_stores = build_all_vectorstores(chunks_semantic, chunks_sliding, batch_size=8)

    # 测试检索
    print("\n" + "=" * 60)
    print("测试 6 种组合方案的检索")
    print("=" * 60)

    test_query = "增值税是什么？"
    print(f"\n查询：{test_query}\n")

    for chunk_method in ['semantic', 'sliding_window']:
        for model_type in ['large', 'small', 'student']:
            store = all_stores[chunk_method][model_type]
            results = search(test_query, store, top_k=2)

            print(f"[{chunk_method} + {model_type}]")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"  [{i}] 分数：{score:.4f} | 内容：{chunk['content'][:50]}...")
            print()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
