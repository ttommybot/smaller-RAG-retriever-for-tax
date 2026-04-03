# src/retrieval/retriever.py

def retrieve_top_k(query: str, top_k: int = 5) -> list[dict]:
    """
    模拟检索功能，返回包含两条 Mock 数据的列表。
    每条数据是一个字典，包含 'text' (税务相关假文本) 和 'source' (假文件名) 两个 key。
    """
    # Mock 数据：两条税务相关的假文本
    mock_docs = [
        {
            "text": "增值税是以商品在流转过程中产生的增值额作为计税依据而征收的一种流转税。增值税的纳税人是在中国境内销售货物或者提供加工、修理修配劳务以及进口货物的单位和个人。",
            "source": "tax_law_basics.pdf"
        },
        {
            "text": "个人所得税是对个人取得的各项所得征收的一种税种。个人所得税的计税依据是纳税人取得的各项应税所得，包括工资、薪金所得，个体工商户的生产、经营所得等。",
            "source": "income_tax_guide.pdf"
        }
    ]

    # 由于是 Mock，直接返回前两条（或根据 top_k 截取，但这里固定两条）
    return mock_docs[:min(top_k, len(mock_docs))]


def format_retrieved_context(retrieved_docs: list[dict]) -> str:
    """
    遍历提取检索到的文档数据，拼接成带有来源标注的纯文本字符串。
    """
    if not retrieved_docs:
        return "未找到相关参考资料。"

    formatted_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        text = doc.get("text", "")
        source = doc.get("source", "未知来源")
        formatted_parts.append(f"[资料{i} - 来源: {source}]\n{text}")

    return "\n\n".join(formatted_parts)


# 测试一下
if __name__ == "__main__":
    query = "什么是增值税？"
    docs = retrieve_top_k(query, top_k=2)
    print("检索结果:")
    for doc in docs:
        print(f"- 文本: {doc['text'][:50]}...")
        print(f"- 来源: {doc['source']}")

    context = format_retrieved_context(docs)
    print("\n格式化上下文:")
    print(context)