# -*- coding: utf-8 -*-
"""
重排程序模块

本模块基于 reranker 打分结果进行重排，主要用于 RAG 推理流程。
与 reranker.py 的职责分离：
- reranker.py: 负责打分
- reorder.py: 负责根据分数重排
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# 兼容导入：支持从 src 目录或项目根目录运行
try:
    from src.reranking.reranker import score_query_chunks
except ImportError:
    from reranking.reranker import score_query_chunks


DocLike = Union[str, Dict[str, Any]]


def _extract_text(doc: DocLike, text_keys: Sequence[str]) -> str:
    """从字符串或字典文档中提取文本。"""
    if isinstance(doc, str):
        return doc.strip()
    for key in text_keys:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def rerank_documents(
    query: str,
    documents: Sequence[DocLike],
    top_k: Optional[int] = None,
    text_keys: Sequence[str] = ("text", "content", "chunk"),
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    根据 reranker 分数对检索结果进行重排。

    返回格式：
    {
      "text": str,
      "reranker_score": float,
      "raw": 原始文档对象
    }
    """
    valid_texts: List[str] = []
    valid_docs: List[DocLike] = []

    for doc in documents:
        text = _extract_text(doc, text_keys)
        if not text:
            continue
        valid_texts.append(text)
        valid_docs.append(doc)

    if not valid_texts:
        return []

    scores = score_query_chunks(
        query=query,
        chunks=valid_texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    ranked = [
        {
            "text": text,
            "reranker_score": float(score),
            "raw": doc,
        }
        for text, score, doc in zip(valid_texts, scores, valid_docs)
    ]
    ranked.sort(key=lambda x: x["reranker_score"], reverse=True)

    if top_k is not None:
        return ranked[:top_k]
    return ranked


def rerank_with_pairs(
    query: str,
    documents: Sequence[DocLike],
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    简化输出：仅返回 (text, score) 列表，便于快速拼接上下文。
    """
    ranked = rerank_documents(query=query, documents=documents, top_k=top_k)
    return [(item["text"], item["reranker_score"]) for item in ranked]
