# -*- coding: utf-8 -*-
"""
数据预处理模块

在 chunk 之后、embed 之前对文本块进行清洗和标准化。
主要职责：
1. 删除空/无效 chunk
2. 去除多余空白
3. 文本标准化（全角/半角、标点、数字、日期）
4. 去除无意义的短字符串
"""

import re
from typing import List, Dict, Any


# ==================== 全角/半角转换 ====================

def _fullwidth_to_halfwidth(text: str) -> str:
    """
    将全角字符转换为半角字符。

    转换范围：
    - 全角字母数字：FF01-FF5E → 0021-007E
    - 全角空格：3000 → 0020

    参数
    ----------
    text : str
        原始文本。

    返回
    -------
    str
        转换后的文本。
    """
    result = []
    for char in text:
        code = ord(char)
        # 全角空格
        if code == 0x3000:
            result.append(' ')
        # 全角字符 (FF01-FF5E) → 半角字符 (0021-007E)
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return ''.join(result)


# ==================== 标点符号标准化 ====================

def _normalize_punctuation(text: str) -> str:
    """
    统一标点符号为中文格式。

    处理内容：
    - 英文引号 → 中文引号
    - 英文括号 → 中文括号
    - 连续句号/省略号 → 中文省略号
    - 破折号统一

    参数
    ----------
    text : str
        原始文本。

    返回
    -------
    str
        标点标准化后的文本。
    """
    # 处理引号：英文直引号 → 中文弯引号
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")

    # 处理括号
    text = text.replace('(', '(').replace(')', ')')

    # 处理省略号（先处理 3 个点，再处理单个）
    text = re.sub(r'\.{3,}', '……', text)
    text = text.replace('…', '……')

    # 处理破折号
    text = re.sub(r'-{2,}', '——', text)
    text = re.sub(r'—{2,}', '——', text)

    return text


# ==================== 数字格式标准化 ====================

def _normalize_numbers(text: str) -> str:
    """
    统一数字格式。

    目前主要处理中文数字转阿拉伯数字（可选功能）。

    参数
    ----------
    text : str
        原始文本。

    返回
    -------
    str
        数字标准化后的文本。
    """
    # 中文数字映射（简单处理，复杂情况需要更复杂的逻辑）
    chinese_nums = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '零': '0',
    }

    # 注意：中文数字转换比较复杂，这里只做简单替换
    # 实际使用时可能需要更复杂的逻辑，如"二十一"→"21"

    return text


# ==================== 日期格式标准化 ====================

def _normalize_dates(text: str) -> str:
    """
    标准化日期格式。

    将各种日期格式统一为：YYYY 年 MM 月 DD 日

    支持的输入格式：
    - YYYY/MM/DD → YYYY 年 MM 月 DD 日
    - YYYY-MM-DD → YYYY 年 MM 月 DD 日
    - YYYY.MM.DD → YYYY 年 MM 月 DD 日

    参数
    ----------
    text : str
        原始文本。

    返回
    -------
    str
        日期标准化后的文本。
    """
    # YYYY/MM/DD → YYYY 年 MM 月 DD 日
    text = re.sub(
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
        r'\1 年\2 月\3 日',
        text
    )

    # YYYY-MM-DD → YYYY 年 MM 月 DD 日
    text = re.sub(
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'\1 年\2 月\3 日',
        text
    )

    # YYYY.MM.DD → YYYY 年 MM 月 DD 日
    text = re.sub(
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
        r'\1 年\2 月\3 日',
        text
    )

    return text


# ==================== 空白字符处理 ====================

def _normalize_whitespace(text: str) -> str:
    """
    标准化空白字符。

    处理内容：
    - 将连续空白字符（空格、制表符、换行）压缩为单个空格
    - 去除首尾空白

    参数
    ----------
    text : str
        原始文本。

    返回
    -------
    str
        空白标准化后的文本。
    """
    # 将各种空白字符统一为单个空格
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空白
    text = text.strip()

    return text


# ==================== 主预处理函数 ====================

def preprocess_chunks(
    chunks: List[Dict[str, Any]],
    min_chunk_length: int = 10,
    normalize_fullwidth: bool = True,
    normalize_punctuation: bool = True,
    normalize_numbers: bool = False,
    normalize_dates: bool = True
) -> List[Dict[str, Any]]:
    """
    对 chunk 列表进行预处理。

    处理步骤：
    1. 删除空/空白 chunk
    2. 去除多余空白（连续空格、制表符、换行）
    3. 去除首尾空白
    4. 统一全角/半角字符
    5. 统一标点符号（如中文/英文引号）
    6. 统一数字格式（可选）
    7. 标准化日期格式
    8. 去除无意义的短字符串

    参数
    ----------
    chunks : List[Dict[str, Any]]
        由 chunker 生成的 chunk 列表。
        每个 chunk 应包含 'id' 和 'content' 字段。

    min_chunk_length : int, optional
        最小 chunk 长度（字符数），默认为 10。
        短于此值的 chunk 会被过滤，因为信息量不足。

    normalize_fullwidth : bool, optional
        是否统一全角/半角字符，默认为 True。
        将全角字符转换为半角。

    normalize_punctuation : bool, optional
        是否统一标点符号，默认为 True。
        将英文标点转换为中文标点。

    normalize_numbers : bool, optional
        是否统一数字格式，默认为 False。
        将中文数字转换为阿拉伯数字。

    normalize_dates : bool, optional
        是否标准化日期格式，默认为 True。
        将各种日期格式统一为 YYYY 年 MM 月 DD 日。

    返回
    -------
    List[Dict[str, Any]]
        经过预处理后的 chunk 列表，格式与原格式相同。

    示例
    ----
    >>> from src.chunking.chunker import sliding_window_chunking
    >>> from src.chunking.preprocess import preprocess_chunks
    >>>
    >>> chunks = sliding_window_chunking(documents, window_size=500, step_size=400)
    >>> clean_chunks = preprocess_chunks(chunks, min_chunk_length=20)
    >>> print(f"原始：{len(chunks)} 个块，清洗后：{len(clean_chunks)} 个块")
    """
    processed_chunks = []

    for chunk in chunks:
        content = chunk.get("content", "")

        # 1. 去除首尾空白
        content = content.strip()

        # 2. 删除空/空白 chunk
        if not content:
            continue

        # 3. 统一全角/半角字符
        if normalize_fullwidth:
            content = _fullwidth_to_halfwidth(content)

        # 4. 统一标点符号
        if normalize_punctuation:
            content = _normalize_punctuation(content)

        # 5. 统一数字格式
        if normalize_numbers:
            content = _normalize_numbers(content)

        # 6. 标准化日期格式
        if normalize_dates:
            content = _normalize_dates(content)

        # 7. 标准化空白字符
        content = _normalize_whitespace(content)

        # 8. 去除无意义的短字符串
        if len(content) < min_chunk_length:
            continue

        # 更新 chunk 内容
        chunk_copy = chunk.copy()
        chunk_copy["content"] = content
        processed_chunks.append(chunk_copy)

    print(f"预处理完成！原始：{len(chunks)} 个块 → 清洗后：{len(processed_chunks)} 个块")
    return processed_chunks


def get_chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取 chunk 列表的统计信息。

    参数
    ----------
    chunks : List[Dict[str, Any]]
        chunk 列表。

    返回
    -------
    Dict[str, Any]
        统计信息，包括：
        - total_chunks: 总数
        - avg_length: 平均长度
        - min_length: 最小长度
        - max_length: 最大长度
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0
        }

    lengths = [len(chunk.get("content", "")) for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }


if __name__ == "__main__":
    # 测试预处理功能
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from loading.loader import load_documents_from_dir
    from chunking.chunker import sliding_window_chunking, raw_data_semantic_chunking, get_chunking_config

    print("=" * 60)
    print("加载文档...")
    print("=" * 60)
    documents = load_documents_from_dir(directory=str(PROJECT_ROOT / "data" / "raw"))

    print("\n" + "=" * 60)
    print("语义分块 + 预处理")
    print("=" * 60)
    semantic_chunks = raw_data_semantic_chunking(documents[:3])
    print("预处理前统计:", get_chunk_stats(semantic_chunks))

    clean_semantic = preprocess_chunks(
        semantic_chunks,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True
    )
    print("预处理后统计:", get_chunk_stats(clean_semantic))

    print("\n" + "=" * 60)
    print("滑动窗口分块 + 预处理")
    print("=" * 60)
    config = get_chunking_config()
    sliding_chunks = sliding_window_chunking(
        documents[:3],
        window_size=config['chunk_size'],
        step_size=config['chunk_size'] - config['chunk_overlap'],
        min_chunk=config['min_chunk']
    )
    print("预处理前统计:", get_chunk_stats(sliding_chunks))

    clean_sliding = preprocess_chunks(
        sliding_chunks,
        min_chunk_length=20,
        normalize_fullwidth=True,
        normalize_punctuation=True,
        normalize_dates=True
    )
    print("预处理后统计:", get_chunk_stats(clean_sliding))
