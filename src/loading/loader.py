# -*- coding: utf-8 -*-
"""
文档加载模块

本模块提供从指定目录批量加载 Word 文档的功能。
只负责读取文档内容，不进行任何分块处理。
"""

import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

try:
    from docx import Document
except ImportError:
    raise ImportError(
        "请安装 python-docx 库：pip install python-docx\n"
        "该库用于读取 Microsoft Word (.docx) 格式文档"
    )


def load_documents_from_dir(
    directory: str = "data/raw",
    file_extension: str = ".docx"
) -> List[Dict[str, Any]]:
    """
    从指定目录批量加载 Word 文档，返回每个文档的完整文本。

    本函数遍历指定目录下的所有 .docx 文件，读取每个文档的完整文本内容，
    并提取文件名中包含的文档类型信息。

    参数
    ----------
    directory : str, optional
        文档目录的路径，默认为 "data/raw"。
        可以是绝对路径或相对路径（相对于当前工作目录）。

    file_extension : str, optional
        要读取的文件扩展名，默认为 ".docx"。

    返回
    -------
    List[Dict[str, Any]]
        包含所有文档的列表，每个元素是一个字典：
        - 'full_text' (str): 文档的完整文本内容
        - 'file_name' (str): 原始文件名（不含扩展名）
        - 'file_type' (str): 从文件名解析出的文档类型
        - 'file_path' (str): 文档的完整路径
        - 'paragraphs' (List[str]): 原始段落列表

    示例
    ----
    >>> documents = load_documents_from_dir()
    >>> print(f"共加载 {len(documents)} 个文档")
    >>> for doc in documents[:3]:
    ...     print(f"{doc['file_type']} - {doc['file_name']}")
    """
    dir_path = Path(directory)

    # 检查目录是否存在
    if not dir_path.exists():
        raise FileNotFoundError(f"指定的文档目录不存在：{directory}")

    # 检查路径是否为目录
    if not dir_path.is_dir():
        raise NotADirectoryError(f"指定的路径不是目录：{directory}")

    all_documents: List[Dict[str, Any]] = []

    # 获取目录下所有 .docx 文件并排序
    files = sorted([
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix == file_extension
    ])

    file_count = len(files)
    print(f"在目录 '{directory}' 中找到 {file_count} 个 '{file_extension}' 格式的文件")

    for file_idx, file_path in enumerate(files, start=1):
        try:
            doc = Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs]
            full_text = "\n".join(paragraphs)

            file_type, file_name = parse_file_name(file_path.stem)

            document_obj = {
                "full_text": full_text,
                "file_name": file_name,
                "file_type": file_type,
                "file_path": str(file_path),
                "paragraphs": paragraphs
            }
            all_documents.append(document_obj)

        except Exception as e:
            print(f"  [{file_idx}/{file_count}] {file_path.name}: 读取失败 - {e}")
            continue

    print(f"加载完成！共处理 {len(all_documents)} 个文档")
    return all_documents


def parse_file_name(file_stem: str) -> Tuple[str, str]:
    """
    从文件名字符串中解析文档类型和文件名称。

    文件名格式："文件类型 - 文件名字"，使用第一个连字符分隔。

    参数
    ----------
    file_stem : str
        不含扩展名的文件名字符串。

    返回
    -------
    Tuple[str, str]
        (file_type, file_name) 元组
    """
    # 支持中文连字符（‐）和英文连字符（-）
    match = re.match(r"^([^‑-]+)[‑-](.*)$", file_stem)

    if match:
        file_type = match.group(1).strip()
        file_name = match.group(2).strip()
        return file_type, file_name
    else:
        return "未知", file_stem.strip()


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent

    documents = load_documents_from_dir(directory=str(PROJECT_ROOT / "data" / "raw"))
    print(f"\n共加载 {len(documents)} 个文档")
    if documents:
        first = documents[0]
        print(f"\n第一个文档：{first['file_type']} - {first['file_name']}")
        print(f"文本长度：{len(first['full_text'])} 字符")
