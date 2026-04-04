# -*- coding: utf-8 -*-
"""
文本嵌入模块

本模块提供多模型 embedding 支持，可加载 large、small、student 三种模型。
每种模型提供三种 embedding 函数：
- load_embedding_model(): 加载 embedding 模型
- embed_texts(): 批量文本转向量
- embed_query(): 单条 query 转向量

模型配置从 configs/configs.yaml 读取。
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import yaml

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "请安装 sentence-transformers 库：pip install sentence-transformers\n"
        "该库用于加载和运行文本 embedding 模型"
    )


# ==================== 配置加载 ====================

def _load_config() -> Dict[str, Any]:
    """
    从 configs/configs.yaml 加载配置。

    返回
    ----
    Dict[str, Any]
        配置字典，包含 embedding 相关的模型名称。
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_path = project_root / "configs" / "configs.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ==================== Large 模型 ====================

_large_model: Optional[SentenceTransformer] = None


def load_large_model() -> SentenceTransformer:
    """
    加载 large 规模的 embedding 模型。

    模型名称从 configs 的 embedding.model_large_name 读取。
    使用单例模式，首次调用时加载，后续调用返回缓存的模型。

    返回
    ----
    SentenceTransformer
        加载后的 large 模型实例。
    """
    global _large_model

    if _large_model is None:
        config = _load_config()
        model_name = config.get('embedding', {}).get('model_large_name', 'BAAI/bge-large-zh-v1.5')
        print(f"正在加载 large 模型：{model_name}")
        _large_model = SentenceTransformer(model_name)
        print(f"Large 模型加载完成")

    return _large_model


def embed_texts_large(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    使用 large 模型批量编码文本列表。

    参数
    ----------
    texts : List[str]
        待编码的文本列表。
    normalize : bool, optional
        是否对输出向量进行归一化，默认为 True。
        归一化后向量长度为 1，便于计算余弦相似度。

    返回
    -------
    np.ndarray
        形状为 (n_texts, embedding_dim) 的向量矩阵。
    """
    model = load_large_model()
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings


def embed_query_large(text: str, normalize: bool = True) -> np.ndarray:
    """
    使用 large 模型编码单条查询文本。

    参数
    ----------
    text : str
        待编码的查询文本。
    normalize : bool, optional
        是否对输出向量进行归一化，默认为 True。

    返回
    -------
    np.ndarray
        形状为 (embedding_dim,) 的向量。
    """
    embeddings = embed_texts_large([text], normalize=normalize)
    return embeddings[0]


# ==================== Small 模型 ====================

_small_model: Optional[SentenceTransformer] = None


def load_small_model() -> SentenceTransformer:
    """
    加载 small 规模的 embedding 模型。

    模型名称从 configs 的 embedding.model_small_name 读取。
    使用单例模式，首次调用时加载，后续调用返回缓存的模型。

    返回
    ----
    SentenceTransformer
        加载后的 small 模型实例。
    """
    global _small_model

    if _small_model is None:
        config = _load_config()
        model_name = config.get('embedding', {}).get('model_small_name', 'sentence-transformers/all-MiniLM-L6-v2')
        print(f"正在加载 small 模型：{model_name}")
        _small_model = SentenceTransformer(model_name)
        print(f"Small 模型加载完成")

    return _small_model


def embed_texts_small(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    使用 small 模型批量编码文本列表。

    参数
    ----------
    texts : List[str]
        待编码的文本列表。
    normalize : bool, optional
        是否对输出向量进行归一化，默认为 True。

    返回
    -------
    np.ndarray
        形状为 (n_texts, embedding_dim) 的向量矩阵。
    """
    model = load_small_model()
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings


def embed_query_small(text: str, normalize: bool = True) -> np.ndarray:
    """
    使用 small 模型编码单条查询文本。

    参数
    ----------
    text : str
        待编码的查询文本。
    normalize : bool, optional
        是否对输出向量进行归一化，默认为 True。

    返回
    -------
    np.ndarray
        形状为 (embedding_dim,) 的向量。
    """
    embeddings = embed_texts_small([text], normalize=normalize)
    return embeddings[0]


# ==================== Student 模型 ====================

_student_model: Optional[SentenceTransformer] = None


def load_student_model() -> SentenceTransformer:
    """
    加载 student 规模的 embedding 模型。

    模型名称从 configs 的 embedding.model_student_name 读取。
    使用单例模式，首次调用时加载，后续调用返回缓存的模型。

    返回
    ----
    SentenceTransformer
        加载后的 student 模型实例。
    """
    global _student_model

    if _student_model is None:
        config = _load_config()
        model_name = config.get('embedding', {}).get('model_student_name', 'sentence-transformers/all-MiniLM-L6-v2')
        print(f"正在加载 student 模型：{model_name}")
        _student_model = SentenceTransformer(model_name)
        print(f"Student 模型加载完成")

    return _student_model


def embed_texts_student(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    使用 student 模型批量编码文本列表。

    参数
    ----------
    texts : List[str]
        待编码的文本列表。
    normalize : bool, optional
        是否对输出向量进行归一化，默认为 True。

    返回
    -------
    np.ndarray
        形状为 (n_texts, embedding_dim) 的向量矩阵。
    """
    model = load_student_model()
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings


def embed_query_student(text: str, normalize: bool = True) -> np.ndarray:
    """
    使用 student 模型编码单条查询文本。

    参数
    ----------
    text : str
        待编码的查询文本。
    normalize : bool, optional
        是否对输出向量进行归一化，默认为 True。

    返回
    -------
    np.ndarray
        形状为 (embedding_dim,) 的向量。
    """
    embeddings = embed_texts_student([text], normalize=normalize)
    return embeddings[0]


# ==================== 统一接口 ====================

# 模型类型映射
MODEL_TYPES = {
    'large': {
        'load': load_large_model,
        'embed_texts': embed_texts_large,
        'embed_query': embed_query_large
    },
    'small': {
        'load': load_small_model,
        'embed_texts': embed_texts_small,
        'embed_query': embed_query_small
    },
    'student': {
        'load': load_student_model,
        'embed_texts': embed_texts_student,
        'embed_query': embed_query_student
    }
}


def get_embedder(model_type: str = 'large') -> Dict[str, Any]:
    """
    获取指定模型的 embedder 接口。

    参数
    ----------
    model_type : str, optional
        模型类型，可选 'large'、'small'、'student'，默认为 'large'。

    返回
    -------
    Dict[str, Any]
        包含 load、embed_texts、embed_query 函数的字典。

    示例
    ----
    >>> embedder = get_embedder('large')
    >>> model = embedder['load']()
    >>> embeddings = embedder['embed_texts'](['文本 1', '文本 2'])
    >>> query_vec = embedder['embed_query']('查询文本')
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"未知的模型类型：{model_type}，可选值：{list(MODEL_TYPES.keys())}")

    return MODEL_TYPES[model_type]


if __name__ == "__main__":
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

    print("\n" + "=" * 60)
    print("测试 Large 模型")
    print("=" * 60)
    large_embedder = get_embedder('large')
    large_model = large_embedder['load']()
    large_embeddings = large_embedder['embed_texts'](test_texts)
    large_query_vec = large_embedder['embed_query']("税收政策")
    print(f"Large 模型输出形状：{large_embeddings.shape}")
    print(f"Query 向量形状：{large_query_vec.shape}")

    print("\n" + "=" * 60)
    print("测试 Small 模型")
    print("=" * 60)
    small_embedder = get_embedder('small')
    small_model = small_embedder['load']()
    small_embeddings = small_embedder['embed_texts'](test_texts)
    small_query_vec = small_embedder['embed_query']("税收政策")
    print(f"Small 模型输出形状：{small_embeddings.shape}")
    print(f"Query 向量形状：{small_query_vec.shape}")

    print("\n" + "=" * 60)
    print("测试 Student 模型")
    print("=" * 60)
    student_embedder = get_embedder('student')
    student_model = student_embedder['load']()
    student_embeddings = student_embedder['embed_texts'](test_texts)
    student_query_vec = student_embedder['embed_query']("税收政策")
    print(f"Student 模型输出形状：{student_embeddings.shape}")
    print(f"Query 向量形状：{student_query_vec.shape}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
