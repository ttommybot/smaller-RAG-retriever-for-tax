# -*- coding: utf-8 -*-
"""
文档分块模块

本模块提供两种分块策略：
1. 原始语义分块（raw_data_semantic_chunking）：基于分隔符进行分块
2. 滑动窗口分块（sliding_window_chunking）：按固定字符数滑动窗口进行分块

分块超参数从 configs/configs.yaml 读取。
分块结果可保存到 data/processed 目录。
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import yaml


def _load_config() -> Dict[str, Any]:
    """
    从 configs/configs.yaml 加载配置。

    返回
    ----
    Dict[str, Any]
        配置字典，包含 chunking 相关的超参数。
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_path = project_root / "configs" / "configs.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _generate_chunk_id(file_type: str, file_name: str, chunk_index: int) -> str:
    """
    生成 chunk 的唯一标识符。

    格式：{file_type}_{file_name}_chunk_{chunk_index:04d}

    参数
    ----------
    file_type : str
        文件类型，如"法律"、"行政法规"等。
    file_name : str
        文件名（不含扩展名）。
    chunk_index : int
        块索引（从 0 开始）。

    返回
    ----
    str
        格式化的 chunk ID。
    """
    # 清理 file_type 和 file_name 中可能影响文件名的字符
    safe_type = file_type.strip().replace("/", "_").replace("\\", "_")
    safe_name = file_name.strip().replace("/", "_").replace("\\", "_")
    return f"{safe_type}_{safe_name}_chunk_{chunk_index:04d}"


def raw_data_semantic_chunking(
    documents: List[Dict[str, Any]],
    chunk_separator: str = "\n\n",
    strip_whitespace: bool = True,
    save_to_file: bool = False,
    output_file: str = "chunks_semantic.json"
) -> List[Dict[str, Any]]:
    """
    对文档列表进行原始语义分块（基于分隔符）。原文档是已经经过语义分块的文档。

    使用指定的分隔符（默认为双换行符）将每个文档切分成多个文本块。

    参数
    ----------
    documents : List[Dict[str, Any]]
        由 load_documents_from_dir() 返回的文档列表。
        每个文档应包含 'full_text'、'file_name'、'file_type' 等字段。

    chunk_separator : str, optional
        用于分割文档内容的分隔符，默认为 "\\n\\n"（双换行符）。

    strip_whitespace : bool, optional
        是否去除每个文本块前后的空白字符，默认为 True。

    save_to_file : bool, optional
        是否将结果保存到 data/processed 目录，默认为 False。

    output_file : str, optional
        输出文件名，默认为 "chunks_semantic.json"。

    返回
    -------
    List[Dict[str, Any]]
        包含所有文本块的列表，每个元素格式为：
        {
            "id": str,                 # {file_type}_{file_name}_chunk_{chunk_index:04d}
            "content": str,            # 文本块内容
            "metadata": {
                "file_name": str,      # 文件名
                "file_type": str,      # 文档类型
                "chunk_index": int,    # 块索引
                "total_chunks": int,   # 总块数
                "file_path": str       # 文件路径
            }
        }

    示例
    ----
    >>> from src.loading.loader import load_documents_from_dir
    >>> from src.chunking.chunker import raw_data_semantic_chunking
    >>>
    >>> documents = load_documents_from_dir()
    >>> chunks = raw_data_semantic_chunking(documents)
    >>> print(chunks[0]['id'])
    '法律_中华人民共和国会计法_chunk_0000'
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        full_text = doc.get("full_text", "")

        file_type = doc.get("file_type", "未知")
        file_name = doc.get("file_name", "未知")
        file_path = doc.get("file_path", "")

        # 按分隔符切分
        raw_chunks = full_text.split(chunk_separator)

        # 过滤空块
        chunks = []
        for chunk in raw_chunks:
            if strip_whitespace:
                chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

        total_chunks = len(chunks)

        # 为每个块添加元数据
        for chunk_idx, chunk_content in enumerate(chunks):
            chunk_obj = {
                "id": _generate_chunk_id(file_type, file_name, chunk_idx),
                "content": chunk_content,
                "metadata": {
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_index": chunk_idx,
                    "total_chunks": total_chunks,
                    "file_path": file_path
                }
            }
            all_chunks.append(chunk_obj)

    print(f"语义分块完成！共生成 {len(all_chunks)} 个文本块")

    # 保存文件
    if save_to_file:
        save_chunks(all_chunks, output_file)

    return all_chunks


def sliding_window_chunking(
    documents: List[Dict[str, Any]],
    window_size: int,
    step_size: int,
    min_chunk: int,
    strip_whitespace: bool = True,
    save_to_file: bool = False,
    output_file: str = "chunks_sliding.json"
) -> List[Dict[str, Any]]:
    """
    对文档列表进行滑动窗口分块。

    按固定字符数进行滑动窗口切分，窗口之间可以有重叠。

    文末小块处理规则：
    - 如果最后一个块长度 < step_size + min_chunk，则合并到倒数第二个块
    - 避免产生过碎的末尾小块

    参数
    ----------
    documents : List[Dict[str, Any]]
        由 load_documents_from_dir() 返回的文档列表。

    window_size : int
        每个文本块的字符数。

    step_size : int
        滑动步长。
        例如：chunk_size=500, chunk_overlap=100 → step_size=400

    min_chunk : int
        允许保留的最小块额外长度。
        文末允许保留的最小长度 = step_size + min_chunk
        建议从 config 读取：min_chunk=config['min_chunk']

    strip_whitespace : bool, optional
        是否去除每个文本块前后的空白字符，默认为 True。

    save_to_file : bool, optional
        是否将结果保存到 data/processed 目录，默认为 False。

    output_file : str, optional
        输出文件名，默认为 "chunks_sliding.json"。

    返回
    -------
    List[Dict[str, Any]]
        包含所有文本块的列表，每个元素格式为：
        {
            "id": str,                 # {file_type}_{file_name}_chunk_{chunk_index:04d}
            "content": str,            # 文本块内容
            "metadata": {
                "file_name": str,      # 文件名
                "file_type": str,      # 文档类型
                "chunk_index": int,    # 块索引
                "total_chunks": int,   # 总块数
                "file_path": str,      # 文件路径
                "start_char": int,     # 在原文中的起始字符位置
                "end_char": int        # 在原文中的结束字符位置
            }
        }

    示例
    ----
    >>> from src.loading.loader import load_documents_from_dir
    >>> from src.chunking.chunker import sliding_window_chunking, get_chunking_config
    >>>
    >>> documents = load_documents_from_dir()
    >>> config = get_chunking_config()
    >>> chunks = sliding_window_chunking(
    ...     documents,
    ...     window_size=config['chunk_size'],
    ...     step_size=config['chunk_size'] - config['chunk_overlap'],
    ...     min_chunk=config['min_chunk']
    ... )
    >>> print(chunks[0]['id'])
    '法律_中华人民共和国会计法_chunk_0000'
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        full_text = doc.get("full_text", "")

        file_type = doc.get("file_type", "未知")
        file_name = doc.get("file_name", "未知")
        file_path = doc.get("file_path", "")

        text_length = len(full_text)
        chunks = []
        start_pos = 0

        # 滑动窗口切分
        while start_pos < text_length:
            end_pos = min(start_pos + window_size, text_length)
            chunk = full_text[start_pos:end_pos]

            if strip_whitespace:
                chunk = chunk.strip()

            if chunk:
                chunks.append({
                    "content": chunk,
                    "start_char": start_pos,
                    "end_char": end_pos
                })

            start_pos += step_size

            # 已到文末，退出
            if end_pos >= text_length:
                break

        # 文末小块处理：如果最后一个块太小，合并到倒数第二个块
        if len(chunks) >= 2:
            last_chunk_len = chunks[-1]["end_char"] - chunks[-1]["start_char"]
            min_allowed_len = step_size + min_chunk  # 允许的最小长度

            if last_chunk_len < min_allowed_len:
                # 合并最后一个块到倒数第二个块
                second_last = chunks[-2]
                # 扩展倒数第二个块到文末
                chunks[-2] = {
                    "content": full_text[second_last["start_char"]:text_length],
                    "start_char": second_last["start_char"],
                    "end_char": text_length
                }
                # 移除最后一个块
                chunks.pop()

        total_chunks = len(chunks)

        # 为每个块添加元数据
        for i, chunk_data in enumerate(chunks):
            chunk_obj = {
                "id": _generate_chunk_id(file_type, file_name, i),
                "content": chunk_data["content"],
                "metadata": {
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "file_path": file_path,
                    "start_char": chunk_data["start_char"],
                    "end_char": chunk_data["end_char"]
                }
            }
            all_chunks.append(chunk_obj)

    print(f"滑动窗口分块完成！共生成 {len(all_chunks)} 个文本块 "
          f"(window_size={window_size}, step_size={step_size})")

    # 保存文件
    if save_to_file:
        save_chunks(all_chunks, output_file)

    return all_chunks


def get_chunking_config() -> Dict[str, Any]:
    """
    获取当前的分块配置。

    返回
    ----
    Dict[str, Any]
        包含 chunk_size、chunk_overlap 和 min_chunk 的配置字典。
    """
    config = _load_config()
    chunking = config.get("chunking", {})
    return {
        "chunk_size": chunking.get("chunk_size", 500),
        "chunk_overlap": chunking.get("chunk_overlap", 100),
        "min_chunk": chunking.get("min_chunk", 100)
    }


def get_processed_data_dir() -> Path:
    """
    获取 processed_data 目录的路径。

    从 configs/configs.yaml 的 paths.processed_data_dir 读取路径。
    如果目录不存在则创建。

    返回
    ----
    Path
        processed_data 目录的 Path 对象。
    """
    config = _load_config()
    processed_data_dir = config.get('paths', {}).get('processed_data_dir', 'data/processed')

    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    dir_path = project_root / processed_data_dir

    # 如果目录不存在则创建
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


def save_chunks(
    chunks: List[Dict[str, Any]],
    output_file: str = "chunks.json",
    save_dir: Optional[Path] = None
) -> str:
    """
    将 chunk 列表保存为 JSON 文件。

    参数
    ----------
    chunks : List[Dict[str, Any]]
        由 chunker 生成的 chunk 列表。
    output_file : str, optional
        输出文件名，默认为 "chunks.json"。
    save_dir : Optional[Path], optional
        保存目录，默认为配置中的 processed_data_dir。

    返回
    -------
    str
        保存文件的完整路径。
    """
    if save_dir is None:
        save_dir = get_processed_data_dir()

    output_path = save_dir / output_file

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunk 数据已保存到：{output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from loading.loader import load_documents_from_dir

    # 显示当前配置
    print("=" * 60)
    print("当前分块配置")
    print("=" * 60)
    config = get_chunking_config()
    print(f"  chunk_size: {config['chunk_size']}")
    print(f"  chunk_overlap: {config['chunk_overlap']}")
    print(f"  min_chunk: {config['min_chunk']}")
    print(f"  → step_size = {config['chunk_size'] - config['chunk_overlap']}")
    print(f"  → 文末最小保留长度 = {config['chunk_size'] - config['chunk_overlap']} + {config['min_chunk']}")

    print("\n" + "=" * 60)
    print("加载文档...")
    print("=" * 60)
    documents = load_documents_from_dir(directory=str(PROJECT_ROOT / "data" / "raw"))

    print("\n" + "=" * 60)
    print("测试 1: 语义分块（前 3 个文档）")
    print("=" * 60)
    semantic_chunks = raw_data_semantic_chunking(documents[:3])

    if semantic_chunks:
        print(f"\n第一个语义块示例：")
        c = semantic_chunks[0]
        print(f"  id: {c['id']}")
        print(f"  content: {c['content'][:60]}...")
        print(f"  metadata: {c['metadata']}")

    print("\n" + "=" * 60)
    print("测试 2: 滑动窗口分块（前 3 个文档，使用 config 配置）")
    print("=" * 60)
    config = get_chunking_config()
    sliding_chunks = sliding_window_chunking(
        documents[:3],
        window_size=config['chunk_size'],
        step_size=config['chunk_size'] - config['chunk_overlap'],
        min_chunk=config['min_chunk']
    )

    print(f"\n第一个滑动窗口块示例：")
    c = sliding_chunks[0]
    print(f"  id: {c['id']}")
    print(f"  content: {c['content'][:60]}...")
    print(f"  metadata: {c['metadata']}")

    print("\n" + "=" * 60)
    print("对比两种分块方法")
    print("=" * 60)
    print(f"语义分块：{len(semantic_chunks)} 个块")
    print(f"滑动窗口：{len(sliding_chunks)} 个块")
