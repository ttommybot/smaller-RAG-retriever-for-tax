# -*- coding: utf-8 -*-
"""
RAG 查询流水线模块

运行完整的 RAG 查询流程：
1. 从向量库加载已嵌入的文档
2. 将用户查询转换为向量
3. 检索最相关的文档
4. 格式化上下文
5. 构建提示词
6. 生成答案
"""

import json
import sys
import os
from typing import Optional, Dict, List, TypedDict

# 添加 src 目录到导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from embedding.vectorstore import load_vectorstore, search
from embedding.embedder import get_embedder
from generation.prompt_builder import build_rag_prompt
from generation.generator import generate_answer


class RAGResult(TypedDict):
    """RAG 流水线返回结果类型"""
    answer: str
    sources: List[str]


def run_rag_pipeline(
    query: str,
    config_path: Optional[str] = "configs/configs.yaml",
    model_type: str = "small"
) -> RAGResult:
    """
    运行完整的 RAG 查询流水线。

    流程：
    1. 加载向量库
    2. 将查询转换为向量
    3. 检索 top_k 个相关文档
    4. 格式化检索到的上下文
    5. 构造 RAG 提示词
    6. 生成答案
    7. 提取来源列表

    参数
    ----------
    query : str
        用户的问题。
    config_path : str, optional
        配置文件路径，默认为 None。
    model_type : str, optional
        embedding 模型类型，可选 "large"、"small"、"student"，默认为 "small"。
        应与构建向量库时使用的模型一致。

    返回
    -------
    RAGResult
        包含 'answer' 和 'sources' 的字典。
    """
    # 步骤 1: 加载向量库
    vectorstore = load_vectorstore(model_type=model_type)

    # 步骤 2: 检索相关文档
    results = search(query, vectorstore, top_k=5)

    # 从结果中提取 chunk 和相似度
    retrieved_chunks = []
    for chunk, score in results:
        retrieved_chunks.append({
            "content": chunk["content"],
            "source": chunk["metadata"].get("file_name", "未知来源"),
            "score": score
        })

    # 步骤 3: 格式化检索到的上下文
    retrieved_context = _format_retrieved_context(retrieved_chunks)

    # 步骤 4: 构造 RAG 提示词
    prompt = build_rag_prompt(query, retrieved_context)

    # 步骤 5: 生成答案
    answer = generate_answer(prompt, config_path)

    # 步骤 6: 提取来源列表
    sources = [chunk["source"] for chunk in retrieved_chunks]

    # 返回指定格式的字典
    return {
        "answer": answer,
        "sources": sources
    }


def _format_retrieved_context(chunks: List[Dict]) -> str:
    """
    格式化检索到的文档为上下文字符串。

    参数
    ----------
    chunks : List[Dict]
        检索到的 chunk 列表，每个包含 'content'、'source'、'score'。

    返回
    -------
    str
        格式化后的上下文字符串。
    """
    if not chunks:
        return "未找到相关参考资料。"

    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        source = chunk.get("source", "未知来源")
        score = chunk.get("score", 0)
        formatted_parts.append(
            f"[资料{i} - 来源：{source} - 相似度：{score:.4f}]\n{content}"
        )

    return "\n\n".join(formatted_parts)


if __name__ == "__main__":
    # 测试问题
    test_query = "什么是增值税？"

    # 运行流水线（使用 small 模型）
    result = run_rag_pipeline(test_query, model_type="small")

    # 以 JSON 格式美观打印结果
    print("RAG Pipeline 测试结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
