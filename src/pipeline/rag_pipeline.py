# src/pipeline/rag_pipeline.py

import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval.retriever import retrieve_top_k, format_retrieved_context
from generation.prompt_builder import build_rag_prompt
from generation.generator import generate_answer


def run_rag_pipeline(query: str, config_path: str = None, **kwargs) -> dict:
    """
    核心串联函数：运行完整的 RAG 流水线。
    接收 query -> 调用 retrieve_top_k -> 调用 format_retrieved_context 格式化 ->
    调用 build_rag_prompt 构造提示词 -> 调用 generate_answer 生成答案。

    Args:
        query: 用户的问题

    Returns:
        dict: 包含 'answer' 和 'sources' 的字典
    """
    # 步骤1: 检索相关文档（Mock 数据）
    retrieved_docs = retrieve_top_k(query, top_k=5)

    # 步骤2: 格式化检索到的上下文
    retrieved_context = format_retrieved_context(retrieved_docs)

    # 步骤3: 构造 RAG 提示词
    prompt = build_rag_prompt(query, retrieved_context)

    # 步骤4: 生成答案
    answer = generate_answer(prompt)

    # 步骤5: 提取来源列表
    sources = [doc["source"] for doc in retrieved_docs]

    # 返回指定格式的字典
    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    # 测试问题
    test_query = "什么是增值税？"

    # 运行流水线
    result = run_rag_pipeline(test_query)

    # 以 JSON 格式美观打印结果
    print("RAG Pipeline 测试结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))