# -*- coding: utf-8 -*-
"""
问题清洗模块

从原始问题文件中清理数据，移除日期等无关内容，输出干净的问题列表。
"""

import re
from pathlib import Path
import sys

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config


def clean_questions(
    input_file: str = None,
    output_file: str = None,
    config_path: str = "configs/configs.yaml"
) -> tuple[int, str]:
    """
    清理问题文件中的日期等无关内容。

    参数
    ----------
    input_file : str, optional
        输入文件路径（含日期的原始文件），默认为配置中的 `query.input_file`。
    output_file : str, optional
        输出文件路径（干净问题文件），默认为配置中的 `query.output_file`。
    config_path : str, optional
        配置文件路径，默认为 "configs/configs.yaml"。

    返回
    -------
    tuple[int, str]
        (清理的问题数量，输出文件路径)
    """
    # 加载配置
    config = load_config(config_path)
    query_config = config.get("query", {})

    # 确定输入输出路径
    if input_file is None:
        input_file = query_config.get("input_file", "seed_questions.txt")
    if output_file is None:
        output_file = query_config.get("output_file", "data/query/seed_questions_cleaned.txt")

    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取并处理文件
    cleaned_lines = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            # 核心逻辑：用正则删除行尾的 YYYY-MM-DD 格式日期，包括日期前的空格
            # 正则解释：\\s* 匹配任意空格，\\d{4}-\\d{2}-\\d{2} 匹配日期格式，$ 匹配行尾
            cleaned_line = re.sub(r"\s*\d{4}-\d{2}-\d{2}$", "", line.strip())
            # 只保留非空行
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

    # 保存处理后的干净文件
    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\n")

    return len(cleaned_lines), str(output_path)


def main():
    """主函数"""
    print("=" * 60)
    print("问题清洗工具")
    print("=" * 60)

    count, output_path = clean_questions()

    print(f"[OK] 处理完成！共清理 {count} 条问题")
    print(f"[FILE] 干净文件已保存到：{output_path}")


if __name__ == "__main__":
    main()
