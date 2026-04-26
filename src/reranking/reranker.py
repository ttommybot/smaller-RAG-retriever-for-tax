# -*- coding: utf-8 -*-
"""
重排序（Reranker）模块

本模块提供 teacher reranker 模型加载、打分与重排能力。
默认使用配置中的 embedding.model_teacher_name（如 BAAI/bge-reranker-v2-gemma）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# 兼容导入：支持从 src 目录或项目根目录运行
try:
    from src.utils.config_loader import load_config
except ImportError:
    from utils.config_loader import load_config

try:
    from sentence_transformers import CrossEncoder
except ImportError as exc:
    raise ImportError(
        "请先安装 sentence-transformers：pip install sentence-transformers"
    ) from exc


_RERANKER_CACHE: Dict[str, CrossEncoder] = {}


def _resolve_model_name(model_name: Optional[str] = None) -> str:
    if model_name:
        return model_name
    config = load_config()
    return config.get("embedding", {}).get(
        "model_teacher_name", "BAAI/bge-reranker-v2-gemma"
    )


def _resolve_local_model_path(model_name: str) -> Optional[Path]:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    local_dir = project_root / "models" / model_name.replace("/", "--")
    if local_dir.exists():
        return local_dir
    return None


def load_reranker(model_name: Optional[str] = None, device: Optional[str] = None) -> CrossEncoder:
    """
    加载并缓存 reranker 模型。

    参数
    ----------
    model_name : Optional[str]
        模型名或本地路径。若为空，使用 configs 中的 teacher 模型名。
    device : Optional[str]
        设备，例："cuda" / "cpu"。为空时由 sentence-transformers 自动选择。
    """
    resolved_name = _resolve_model_name(model_name)
    cache_key = f"{resolved_name}|{device or 'auto'}"
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    local_path = _resolve_local_model_path(resolved_name)
    model_source = str(local_path) if local_path else resolved_name
    model = CrossEncoder(model_source, device=device)
    _RERANKER_CACHE[cache_key] = model
    return model


def score_query_chunks(
    query: str,
    chunks: Sequence[str],
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> List[float]:
    """
    对单个 query 与多个 chunk 进行打分。
    """
    if not chunks:
        return []

    model = load_reranker(model_name=model_name, device=device)
    pairs = [[query, chunk] for chunk in chunks]
    scores = model.predict(pairs, batch_size=batch_size)
    return [float(score) for score in scores]


def rerank_chunks(
    query: str,
    chunks: Sequence[Union[str, Dict[str, Any]]],
    top_k: Optional[int] = None,
    text_key: str = "content",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    对候选 chunk 重排，返回按分数降序排列的结果。

    返回元素结构：
    {
      "text": str,
      "score": float,
      "raw": 原始输入（str 或 Dict）
    }
    """
    texts: List[str] = []
    raws: List[Union[str, Dict[str, Any]]] = []
    for item in chunks:
        if isinstance(item, str):
            text = item
        else:
            text = str(item.get(text_key, ""))
        text = text.strip()
        if not text:
            continue
        texts.append(text)
        raws.append(item)

    if not texts:
        return []

    scores = score_query_chunks(
        query=query,
        chunks=texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    reranked = [
        {"text": text, "score": score, "raw": raw}
        for text, score, raw in zip(texts, scores, raws)
    ]
    reranked.sort(key=lambda x: x["score"], reverse=True)

    if top_k is not None:
        return reranked[:top_k]
    return reranked