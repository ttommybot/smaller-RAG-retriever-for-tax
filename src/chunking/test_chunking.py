# -*- coding: utf-8 -*-
"""
分块与预处理测试脚本

测试 chunker 和 preprocess 管道是否正确处理文档。
选取 5 个文件进行测试，显示最终输出结果。

使用方法：
    # 在项目根目录运行
    python src/chunking/test_chunking.py
"""

import sys
from pathlib import Path

# 获取项目根目录（当前文件的父目录的父目录的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loading.loader import load_documents_from_dir
from chunking.chunker import sliding_window_chunking, get_chunking_config
from chunking.preprocess import preprocess_chunks, get_chunk_stats


def test_chunking_pipeline():
    """测试 chunker + preprocess 管道"""
    print("=" * 60)
    print("分块与预处理测试")
    print("=" * 60)

    # 加载配置
    config = get_chunking_config()
    print(f"\n配置：chunk_size={config['chunk_size']}, "
          f"chunk_overlap={config['chunk_overlap']}, "
          f"min_chunk={config['min_chunk']}")

    # 1. 加载文档（使用项目根目录的相对路径）
    print("\n" + "-" * 60)
    print("步骤 1: 加载文档")
    print("-" * 60)
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents_from_dir(directory=str(data_dir))

    if not documents:
        print("错误：未加载到任何文档")
        return

    # 选取前 5 个文档进行测试
    test_docs = documents[:5]
    print(f"选取 {len(test_docs)} 个文档进行测试")

    # 2. 分块
    print("\n" + "-" * 60)
    print("步骤 2: 滑动窗口分块")
    print("-" * 60)
    chunks = sliding_window_chunking(
        test_docs,
        window_size=config['chunk_size'],
        step_size=config['chunk_size'] - config['chunk_overlap'],
        min_chunk=config['min_chunk']
    )
    print(f"分块后统计：{get_chunk_stats(chunks)}")

    # 3. 预处理
    print("\n" + "-" * 60)
    print("步骤 3: 预处理")
    print("-" * 60)
    clean_chunks = preprocess_chunks(
        chunks,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True
    )
    print(f"预处理后统计：{get_chunk_stats(clean_chunks)}")

    # 4. 显示结果
    print("\n" + "=" * 60)
    print("最终结果预览")
    print("=" * 60)

    for i, chunk in enumerate(clean_chunks[:10], start=1):
        print(f"\n[Chunk {i}]")
        print(f"  ID: {chunk['id']}")
        print(f"  长度：{len(chunk['content'])} 字符")
        print(f"  内容：{chunk['content'][:150]}...")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"总 chunk 数：{len(clean_chunks)}")
    print(f"平均长度：{get_chunk_stats(clean_chunks)['avg_length']:.1f} 字符")


if __name__ == "__main__":
    test_chunking_pipeline()
