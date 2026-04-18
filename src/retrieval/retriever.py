# -*- coding: utf-8 -*-
"""
检索模块

本模块提供基于向量库的语义检索功能。

六种组合方案 (chunk_method × model_type):
- chunk_method: 'semantic' | 'sliding_window' (2 种)
- model_type: 'large' | 'small' | 'student' (3 种)

默认配置：chunk_method='semantic', model_type='large'
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.embedder import _load_config
from embedding.vectorstore import (
    load_vectorstore,
    load_all_vectorstores,
    search,
    search_from_config,
    get_searcher
)


def retrieve_top_k(
    query: str,
    top_k: int = 5,
    chunk_method: str = "semantic",
    model_type: str = "large",
    vectorstore: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    基于查询文本检索 Top-K 相似文档。

    参数
    ----------
    query : str
        查询文本。

    top_k : int, optional
        返回结果数量，默认为 5。

    chunk_method : str, optional
        分块方法，可选 'semantic' | 'sliding_window'，默认为 'semantic'。

    model_type : str, optional
        embedding 模型类型，可选 'large' | 'small' | 'student'，默认为 'large'。

    vectorstore : Optional[Dict[str, Any]], optional
        预加载的向量库。

    返回
    -------
    List[Dict[str, Any]]
        检索结果列表。
    """
    if vectorstore is None:
        print(f"正在加载向量库 ({chunk_method} + {model_type})...")
        vectorstore = load_vectorstore(
            chunk_method=chunk_method, model_type=model_type
        )

    results = search(query, vectorstore, top_k=top_k)

    formatted_results = []
    for chunk, score in results:
        formatted_results.append({
            'text': chunk.get('content', ''),
            'source': chunk.get('metadata', {}).get('file_name', '未知来源'),
            'score': round(score, 4),
            'metadata': chunk.get('metadata', {})
        })

    return formatted_results


def retrieve_quick(
    query: str,
    top_k: int = 5,
    chunk_method: str = "semantic",
    model_type: str = "large"
) -> List[Dict[str, Any]]:
    """
    便捷检索接口：自动加载向量库并返回结果。

    参数
    ----------
    query : str
        查询文本。

    top_k : int, optional
        返回结果数量，默认为 5。

    chunk_method : str, optional
        分块方法，默认为 'semantic'。

    model_type : str, optional
        模型类型，默认为 'large'。

    返回
    -------
    List[Dict[str, Any]]
        检索结果列表。
    """
    results = search_from_config(
        query, chunk_method=chunk_method, model_type=model_type, top_k=top_k
    )

    formatted_results = []
    for chunk, score in results:
        formatted_results.append({
            'text': chunk.get('content', ''),
            'source': chunk.get('metadata', {}).get('file_name', '未知来源'),
            'score': round(score, 4),
            'metadata': chunk.get('metadata', {})
        })

    return formatted_results


def format_retrieved_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    将检索到的文档格式化为带来源标注的纯文本字符串。

    参数
    ----------
    retrieved_docs : List[Dict[str, Any]]
        检索结果列表。

    返回
    -------
    str
        格式化后的上下文文本。
    """
    if not retrieved_docs:
        return "未找到相关参考资料。"

    formatted_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        text = doc.get("text", "")
        source = doc.get("source", "未知来源")
        score = doc.get("score", 0)
        formatted_parts.append(f"[资料{i} - 来源：{source} - 相似度：{score:.4f}]\n{text}")

    return "\n\n".join(formatted_parts)


def get_retriever(
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
        包含 retrieve 和 format 函数的字典。
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(
            chunk_method=chunk_method, model_type=model_type
        )

    return {
        'retrieve': lambda q, k=5: retrieve_top_k(q, k, chunk_method, model_type, vectorstore),
        'format': format_retrieved_context,
        'vectorstore': vectorstore,
        'chunk_method': chunk_method,
        'model_type': model_type
    }


def get_all_retrievers(
    vectorstores: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    获取全部 6 种组合方案的检索器。

    返回
    -------
    Dict[str, Dict[str, Dict[str, Any]]]
        嵌套字典结构：
        {
            'semantic': {'large': retriever, 'small': retriever, 'student': retriever},
            'sliding_window': {'large': retriever, 'small': retriever, 'student': retriever}
        }
    """
    if vectorstores is None:
        vectorstores = load_all_vectorstores()

    retrievers = {
        'semantic': {},
        'sliding_window': {}
    }

    for chunk_method in ['semantic', 'sliding_window']:
        for model_type in ['large', 'small', 'student']:
            retrievers[chunk_method][model_type] = get_searcher(
                chunk_method=chunk_method,
                model_type=model_type,
                vectorstore=vectorstores[chunk_method][model_type]
            )

    return retrievers


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Retriever 模块测试 - 6 种组合方案")
    print("=" * 60)

    config = _load_config()
    top_k = config.get('retrieval', {}).get('top_k', 5)
    print(f"\n配置：top_k = {top_k}")

    test_queries = [
        "增值税是什么？",
        "企业所得税如何计算？",
        "个人所得税专项附加扣除有哪些？"
    ]

    # 测试 6 种组合
    print("\n" + "=" * 60)
    print("测试 6 种组合方案")
    print("=" * 60)

    for chunk_method in ['semantic', 'sliding_window']:
        for model_type in ['large', 'small', 'student']:
            print(f"\n{'=' * 40}")
            print(f"[{chunk_method} + {model_type}]")
            print(f"{'=' * 40}")

            try:
                for query in test_queries[:1]:
                    print(f"  查询：{query}")
                    docs = retrieve_top_k(
                        query, top_k=top_k,
                        chunk_method=chunk_method, model_type=model_type
                    )
                    for i, doc in enumerate(docs[:2], 1):
                        print(f"    [{i}] 分数：{doc['score']:.4f} | 来源：{doc['source']}")
            except FileNotFoundError as e:
                print(f"  向量库未找到：{e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
