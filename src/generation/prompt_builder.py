# src/generation/prompt_builder.py

def build_rag_prompt(query: str, retrieved_context: str) -> str:
    """
    构造发给大模型的 RAG 专属 Prompt。
    """
    prompt = f"""
你是一个专业的税务问答助手。请严格根据以下【参考资料】来回答用户的【问题】。
要求：
1. 回答要清晰、准确、有条理。
2. 如果在参考资料中找不到答案，请诚实地回答“根据提供的资料，我无法回答该问题”，绝不能捏造事实。

【参考资料】：
{retrieved_context}

【问题】：
{query}

请给出你的回答：
"""
    return prompt.strip()

# 测试一下
if __name__ == "__main__":
    test_query = "个人所得税起征点是多少？"
    test_context = "根据国家税务总局规定，目前个人所得税的起征点为每月 5000 元人民币。"
    print(build_rag_prompt(test_query, test_context))