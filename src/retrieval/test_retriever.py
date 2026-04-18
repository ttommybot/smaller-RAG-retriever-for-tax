# -*- coding: utf-8 -*-
"""
Retriever 模块测试脚本

测试 6 种组合方案 (chunk_method × model_type) 的检索功能。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.retriever import (
    retrieve_top_k,
    retrieve_quick,
    format_retrieved_context,
    get_retriever,
    _load_config
)
from embedding.vectorstore import load_vectorstore


def test_single_config(
    chunk_method: str = "semantic",
    model_type: str = "large",
    top_k: int = 5
):
    """测试单一组合方案"""
    print("=" * 60)
    print(f"Retriever 模块测试 ({chunk_method} + {model_type})")
    print("=" * 60)

    test_queries = [
        "增值税是什么？",
        "企业所得税如何计算？",
        "个人所得税专项附加扣除有哪些？"
    ]

    print(f"\n正在加载向量库 ({chunk_method} + {model_type})...")
    try:
        vectorstore = load_vectorstore(chunk_method=chunk_method, model_type=model_type)
        print(f"向量库加载成功：{vectorstore['embeddings'].shape}")
    except FileNotFoundError as e:
        print(f"错误：{e}")
        print(f"\n请先构建向量库")
        return

    print("\n执行检索测试:")
    for query in test_queries:
        print(f"\n查询：{query}")
        docs = retrieve_top_k(
            query, top_k=top_k,
            chunk_method=chunk_method, model_type=model_type,
            vectorstore=vectorstore
        )
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] 分数：{doc['score']:.4f} | 来源：{doc['source']}")
            print(f"      内容：{doc['text'][:80]}...")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def test_all_configs():
    """对比测试全部 6 种组合方案"""
    print("=" * 60)
    print("6 种组合方案对比测试")
    print("=" * 60)

    config = _load_config()
    top_k = config.get('retrieval', {}).get('top_k', 5)

    test_query = "增值税优惠政策有哪些？"
    print(f"\n测试查询：{test_query}")
    print(f"检索数量：Top-{top_k}\n")

    results_summary = {}

    for chunk_method in ['semantic', 'sliding_window']:
        for model_type in ['large', 'small', 'student']:
            key = f"{chunk_method} + {model_type}"
            print(f"{'=' * 40}")
            print(f"[{key}]")
            print(f"{'=' * 40}")

            try:
                docs = retrieve_top_k(
                    test_query, top_k=3,
                    chunk_method=chunk_method, model_type=model_type
                )
                results_summary[key] = docs

                for i, doc in enumerate(docs, 1):
                    print(f"  [{i}] 分数：{doc['score']:.4f} | 来源：{doc['source']}")

            except FileNotFoundError as e:
                print(f"  向量库未找到")
                results_summary[key] = None
            print()

    # 汇总对比
    print("\n" + "=" * 60)
    print("结果汇总 (第一条结果对比)")
    print("=" * 60)

    for key, docs in results_summary.items():
        if docs:
            top_doc = docs[0]
            print(f"{key}: 分数={top_doc['score']:.4f} | 来源={top_doc['source']}")
        else:
            print(f"{key}: 未找到向量库")


def test_retriever_interface():
    """测试 retriever 接口"""
    print("=" * 60)
    print("Retriever 接口测试")
    print("=" * 60)

    try:
        retriever = get_retriever(chunk_method='semantic', model_type='large')
        print(f"检索器已加载：{retriever['chunk_method']} + {retriever['model_type']}")

        query = "税收政策"
        print(f"\n查询：{query}")

        docs = retriever['retrieve'](query, k=3)
        print(f"\n检索结果 ({len(docs)} 条):")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] 分数：{doc['score']:.4f} | 来源：{doc['source']}")

        context = retriever['format'](docs)
        print(f"\n格式化上下文:\n{context[:300]}...")

    except FileNotFoundError as e:
        print(f"错误：{e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retriever 模块测试")
    parser.add_argument(
        '--chunk-method',
        choices=['semantic', 'sliding_window'],
        default='semantic',
        help='分块方法'
    )
    parser.add_argument(
        '--model',
        choices=['large', 'small', 'student'],
        default='large',
        help='embedding 模型'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'all', 'interface'],
        default='single',
        help='测试模式'
    )
    parser.add_argument('--top-k', type=int, default=None, help='覆盖配置的 top_k 值')

    args = parser.parse_args()

    if args.top_k is not None:
        config = _load_config()
        config['retrieval']['top_k'] = args.top_k

    if args.mode == 'all':
        test_all_configs()
    elif args.mode == 'interface':
        test_retriever_interface()
    else:
        config = _load_config()
        top_k = config.get('retrieval', {}).get('top_k', 5)
        test_single_config(
            chunk_method=args.chunk_method,
            model_type=args.model,
            top_k=top_k
        )
