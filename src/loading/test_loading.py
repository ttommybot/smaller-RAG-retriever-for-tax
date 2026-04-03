# -*- coding: utf-8 -*-
"""
加载模块测试脚本

测试 load_documents_from_dir() 函数是否正确加载文档。
选取 5 个文件进行测试，显示加载结果。

使用方法：
    # 在项目根目录运行
    python src/loading/test_loading.py
"""

import sys
from pathlib import Path

# 获取项目根目录（当前文件的父目录的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loading.loader import load_documents_from_dir


def test_load_documents():
    """测试加载 5 个文档并显示内容"""
    print("=" * 60)
    print("加载模块测试")
    print("=" * 60)

    # 使用相对于项目根目录的路径
    data_dir = PROJECT_ROOT / "data" / "raw"
    documents = load_documents_from_dir(directory=str(data_dir))

    if not documents:
        print("错误：未加载到任何文档")
        return

    # 选取前 5 个文档进行测试
    test_docs = documents[:5]

    print(f"\n从 {len(documents)} 个文档中选取 {len(test_docs)} 个进行测试")
    print("=" * 60)

    for i, doc in enumerate(test_docs, start=1):
        print(f"\n[文档 {i}/5]")
        print("-" * 40)
        print(f"文件类型：{doc['file_type']}")
        print(f"文件名称：{doc['file_name']}")
        print(f"文件路径：{doc['file_path']}")
        print(f"段落数量：{len(doc['paragraphs'])}")
        print(f"文本长度：{len(doc['full_text'])} 字符")
        print(f"\n内容预览（前 200 字符）:")
        print(doc['full_text'][:200])
        print("..." if len(doc['full_text']) > 200 else "")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_load_documents()
