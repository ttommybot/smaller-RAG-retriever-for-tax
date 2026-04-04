# -*- coding: utf-8 -*-
"""
Embedding 模块测试脚本

测试三种模型（large、small、student）的加载和 embedding 功能。

使用方法：
    # 在项目根目录运行
    python src/embedding/test_embedding.py
"""

import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding.embedder import (
    get_embedder,
    embed_texts_large, embed_query_large,
    embed_texts_small, embed_query_small,
    embed_texts_student, embed_query_student,
    _load_config
)


def test_embedding_models():
    """测试三种 embedding 模型"""
    print("=" * 60)
    print("Embedding 模块测试")
    print("=" * 60)

    # 显示配置
    config = _load_config()
    print(f"\n当前配置:")
    print(f"  large 模型：{config['embedding']['model_large_name']}")
    print(f"  small 模型：{config['embedding']['model_small_name']}")
    print(f"  student 模型：{config['embedding']['model_student_name']}")

    # 测试文本
    test_texts = [
        "什么是增值税？",
        "企业所得税如何计算？",
        "个人所得税专项附加扣除有哪些？"
    ]
    test_query = "税收政策"

    # 测试 Large 模型
    print("\n" + "=" * 60)
    print("测试 Large 模型")
    print("=" * 60)
    large_embedder = get_embedder('large')
    large_model = large_embedder['load']()
    large_embeddings = large_embedder['embed_texts'](test_texts)
    large_query_vec = large_embedder['embed_query'](test_query)
    print(f"Large 模型输出形状：{large_embeddings.shape}")
    print(f"Query 向量形状：{large_query_vec.shape}")
    print(f"前 5 个维度示例：{large_query_vec[:5]}")

    # 测试 Small 模型
    print("\n" + "=" * 60)
    print("测试 Small 模型")
    print("=" * 60)
    small_embedder = get_embedder('small')
    small_model = small_embedder['load']()
    small_embeddings = small_embedder['embed_texts'](test_texts)
    small_query_vec = small_embedder['embed_query'](test_query)
    print(f"Small 模型输出形状：{small_embeddings.shape}")
    print(f"Query 向量形状：{small_query_vec.shape}")
    print(f"前 5 个维度示例：{small_query_vec[:5]}")

    # 测试 Student 模型
    print("\n" + "=" * 60)
    print("测试 Student 模型")
    print("=" * 60)
    student_embedder = get_embedder('student')
    student_model = student_embedder['load']()
    student_embeddings = student_embedder['embed_texts'](test_texts)
    student_query_vec = student_embedder['embed_query'](test_query)
    print(f"Student 模型输出形状：{student_embeddings.shape}")
    print(f"Query 向量形状：{student_query_vec.shape}")
    print(f"前 5 个维度示例：{student_query_vec[:5]}")

    # 验证缓存机制（第二次加载应该瞬间完成）
    print("\n" + "=" * 60)
    print("验证模型缓存机制")
    print("=" * 60)
    print("再次加载 large 模型（应使用缓存，不重新加载）...")
    large_model_cached = get_embedder('large')['load']()
    print(f"缓存验证成功：{large_model is large_model_cached}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_embedding_models()
