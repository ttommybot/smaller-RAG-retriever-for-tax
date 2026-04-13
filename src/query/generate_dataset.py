# -*- coding: utf-8 -*-
"""
问题数据集生成模块

基于种子问题，调用 LLM API 批量生成更多税务咨询问题，生成不同规模的数据集。
"""

import asyncio
import aiohttp
from aiohttp import ClientTimeout
import re
import random
import os
import shutil
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List, cast

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config


# ========================== 配置区 ==========================

# 优化后的 Prompt
GENERATION_PROMPT = """
你是一位专业的中国税务咨询问题生成专家，擅长基于真实的税务咨询场景，举一反三地生成合理的新问题。

你的任务是：基于给定的【原始税务咨询问题】，生成{generate_per_seed}个全新的、独立的、符合真实用户咨询习惯的税务问题。

生成要求：
1. 灵活调整地域、纳税主体类型、具体金额、涉及的税种，保证每个新问题的场景有细微差异。
2. 生成的问题必须是真实用户会问的，口语化但逻辑严谨，符合 2025-2026 年现行中国税法。
3. 只输出纯问题文本，每行 1 个，不要序号、不要解释、不要任何多余内容，不要重复原始问题。

原始问题：{seed_question}
"""


async def generate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    seed: str,
    idx: int,
    api_url: str,
    api_key: str,
    model: str,
    generate_per_seed: int
) -> List[str]:
    """单条种子的异步生成"""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": GENERATION_PROMPT.format(
                generate_per_seed=generate_per_seed,
                seed_question=seed
            )
        }],
        "temperature": 0.8
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 设置超时配置
    timeout = ClientTimeout(total=30)

    async with semaphore:
        for retry in range(3):
            try:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        new_questions = [
                            re.sub(r"^\d+[.、) ]*|^[①②③④⑤⑥⑦⑧⑨⑩][.、) ]+", "", q.strip())
                            for q in content.split("\n")
                            if q.strip() and "？" in q
                        ]
                        print(f"[OK] 第{idx+1}条种子生成成功，产出{len(new_questions)}条")
                        return new_questions
            except Exception as e:
                print(f"[WARN] 第{idx+1}条种子请求失败：{e}")
                pass
            await asyncio.sleep(1)
        print(f"[FAIL] 第{idx+1}条种子生成失败，跳过")
        return []


async def generate_dataset_async(
    seed_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_path: str = "configs/configs.yaml"
) -> Dict[str, Any]:
    """
    异步批量生成问题数据集。

    参数
    ----------
    seed_file : str, optional
        种子问题文件路径，默认为配置中的 `query.seed_file`。
    output_dir : str, optional
        输出目录，默认为配置中的 `query.output_dir`。
    config_path : str, optional
        配置文件路径，默认为 "configs/configs.yaml"。

    返回
    -------
    Dict[str, Any]
        生成的数据集信息，包括各规模数据集的问题数量和路径。
    """
    # 加载配置
    config = load_config(config_path)
    query_config = config.get("query", {})
    api_config = config.get("api", {})

    # 确定路径（使用 cast 告诉 Pylance 这些变量不会是 None）
    if seed_file is None:
        seed_file = cast(str, query_config.get("seed_file", "data/query/seed_questions_cleaned.txt"))
    if output_dir is None:
        output_dir = cast(str, query_config.get("output_dir", "data/query"))

    # API 配置
    api_url = cast(str, api_config.get("generation_url", "https://api.closeai-asia.com/v1/chat/completions"))
    api_key = cast(str, api_config.get("generation_api_key", ""))
    model = cast(str, api_config.get("generation_model", "deepseek-chat"))

    # 生成参数
    generate_per_seed = query_config.get("generate_per_seed", 6)
    concurrent_limit = query_config.get("concurrent_limit", 15)
    dataset_sizes = query_config.get("dataset_sizes", {
        "small": 200,
        "medium": 500,
        "large": 1000
    })

    print("=" * 60)
    print("步骤 1/4：正在读取种子问题...")
    with open(seed_file, "r", encoding="utf-8") as f:
        seed_questions = [line.strip() for line in f if line.strip()]
    print(f"[OK] 成功读取 {len(seed_questions)} 条种子")

    print("\n" + "=" * 60)
    print("步骤 2/4：正在异步并发批量生成...")
    semaphore = asyncio.Semaphore(concurrent_limit)
    all_augmented = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_one(
                session, semaphore, seed, idx,
                api_url, api_key, model, generate_per_seed
            )
            for idx, seed in enumerate(seed_questions)
        ]
        results = await asyncio.gather(*tasks)

    for res in results:
        all_augmented.extend(res)
    print(f"\n[OK] 批量生成完成！原始数据量：{len(all_augmented)} 条")

    print("\n" + "=" * 60)
    print("步骤 3/4：正在清洗去重、打乱顺序...")
    seen = set()
    cleaned_questions = []
    for q in all_augmented:
        q = re.sub(r"\s+", " ", q.strip())
        if (
            len(q) >= 10
            and "？" in q
            and not q.endswith("...")
            and q not in seen
        ):
            seen.add(q)
            cleaned_questions.append(q)

    # 打乱顺序
    random.shuffle(cleaned_questions)
    print(f"[OK] 清洗完成！有效数据量：{len(cleaned_questions)} 条")

    print("\n" + "=" * 60)
    print("步骤 4/4：正在生成数据集...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_datasets = {}
    for name, size in dataset_sizes.items():
        dataset = cleaned_questions[:size]
        file_path = output_path / f"tax_queries_{name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for q in dataset:
                f.write(q + "\n")
        generated_datasets[name] = {
            "count": len(dataset),
            "path": str(file_path)
        }
        print(f"[OK] {name}数据集：{len(dataset)} 条，已保存到 {file_path}")

    # 备份种子文件
    if os.path.exists(seed_file):
        backup_path = output_path / "seed_questions_source.txt"
        shutil.copy(seed_file, backup_path)
        print("\n[OK] 种子源文件已备份")

    # 生成 README
    readme_content = f"""# 税务咨询 RAG 项目 - Query 数据集

## 数据集说明
本数据集为高等机器学习项目专用的税务咨询用户 Query 数据集。

## 版本说明
| 版本 | 规模 | 文件 |
|------|------|------|
| small | {generated_datasets.get('small', {}).get('count', 0)}条 | tax_queries_small.txt |
| medium | {generated_datasets.get('medium', {}).get('count', 0)}条 | tax_queries_medium.txt |
| large | {generated_datasets.get('large', {}).get('count', 0)}条 | tax_queries_large.txt |

## 生成方式
- 种子来源：国家税务总局 12366 官方热点问答
- 扩充模型：{model}
- 处理方式：已完成去重、清洗、随机打乱
"""
    readme_path = output_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("[OK] README.md 已生成")

    print("\n" + "=" * 60)
    print("[DONE] 所有任务全部完成！")
    print(f"[DIR] 最终交付文件已保存在：{output_path}")

    return generated_datasets


def generate_dataset(
    seed_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_path: str = "configs/configs.yaml"
) -> Dict[str, Any]:
    """
    生成问题数据集的同步接口。

    参数
    ----------
    seed_file : str, optional
        种子问题文件路径。
    output_dir : str, optional
        输出目录。
    config_path : str, optional
        配置文件路径。

    返回
    -------
    Dict[str, Any]
        生成的数据集信息。
    """
    return asyncio.run(generate_dataset_async(seed_file, output_dir, config_path))


def main():
    """主函数"""
    generate_dataset()


if __name__ == "__main__":
    # 自动安装 aiohttp（如果没装）
    try:
        import aiohttp
    except ImportError:
        print("正在安装依赖 aiohttp...")
        import subprocess
        subprocess.check_call(["python", "-m", "pip", "install", "aiohttp"])

    main()
