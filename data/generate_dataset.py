import asyncio
import aiohttp
import re
import random
import os
import shutil

# ========================== 核心配置区（仅需改API Key） ==========================
API_KEY = "sk-u235i9PfGBJDtTDm705FcqB7dgRTMbIezjPz8twlWOeV5MgY"
SEED_FILE = "seed_questions_cleaned.txt"  # 你的200条纯问题种子文件
OUTPUT_DIR = "./tax_query_datasets"
GENERATE_PER_SEED = 6  # 每条种子生成6条，200*6=1200条原始数据，清洗后肯定超1000
CONCURRENT_LIMIT = 15
# 新的数据集规模定义
DATASET_SIZES = {
    "small": 200,
    "medium": 500,
    "large": 1000
}
# ================================================================================

# 优化后的Prompt
OPTIMIZED_PROMPT = """
你是一位专业的中国税务咨询问题生成专家，擅长基于真实的税务咨询场景，举一反三地生成合理的新问题。

你的任务是：基于给定的【原始税务咨询问题】，生成{GENERATE_PER_SEED}个全新的、独立的、符合真实用户咨询习惯的税务问题。

生成要求：
1.  灵活调整地域、纳税主体类型、具体金额、涉及的税种，保证每个新问题的场景有细微差异。
2.  生成的问题必须是真实用户会问的，口语化但逻辑严谨，符合2025-2026年现行中国税法。
3.  只输出纯问题文本，每行1个，不要序号、不要解释、不要任何多余内容，不要重复原始问题。

原始问题：{SEED_QUESTION}
"""

async def generate_one(session, semaphore, seed, idx):
    """单条种子的异步生成"""
    url = "https://api.closeai-asia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",  
        "messages": [{"role": "user", "content": OPTIMIZED_PROMPT.format(GENERATE_PER_SEED=GENERATE_PER_SEED, SEED_QUESTION=seed)}],
        "temperature": 0.8
    }

    async with semaphore:
        for retry in range(3):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        new_questions = [
                            re.sub(r"^\d+[.、) ]*|^[①②③④⑤⑥⑦⑧⑨⑩][.、) ]+", "", q.strip())
                            for q in content.split("\n")
                            if q.strip() and "？" in q
                        ]
                        print(f"✅ 第{idx+1}/200条种子生成成功，产出{len(new_questions)}条")
                        return new_questions
            except Exception as e:
                pass
            await asyncio.sleep(1)
        print(f"❌ 第{idx+1}/200条种子生成失败，跳过")
        return []

async def main():
    print("="*60)
    print("步骤1/4：正在读取200条种子问题...")
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        seed_questions = [line.strip() for line in f if line.strip()]
    print(f"✅ 成功读取 {len(seed_questions)} 条种子")

    print("\n" + "="*60)
    print("步骤2/4：正在异步并发批量生成...（速度极快）")
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    all_augmented = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_one(session, semaphore, seed, idx)
            for idx, seed in enumerate(seed_questions)
        ]
        results = await asyncio.gather(*tasks)

    for res in results:
        all_augmented.extend(res)
    print(f"\n✅ 批量生成完成！原始数据量：{len(all_augmented)} 条")

    print("\n" + "="*60)
    print("步骤3/4：正在清洗去重、打乱顺序...")
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
    
    # 打乱顺序，保证随机性
    random.shuffle(cleaned_questions)
    print(f"✅ 清洗完成！有效数据量：{len(cleaned_questions)} 条（目标1000条）")

    # 无论多少都继续生成文件

    print("\n" + "="*60)
    print("步骤4/4：正在生成200/500/1000三个数据集...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, size in DATASET_SIZES.items():
        # 有多少生成多少，不中断
        dataset = cleaned_questions[:size]
        output_path = os.path.join(OUTPUT_DIR, f"tax_queries_{name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for q in dataset:
                f.write(q + "\n")
        print(f"✅ {name}数据集：{len(dataset)} 条，已保存到 {output_path}")

    # 备份种子文件
    if os.path.exists(SEED_FILE):
        shutil.copy(SEED_FILE, os.path.join(OUTPUT_DIR, "seed_questions_source.txt"))
        print("\n✅ 种子源集已备份")

    # 生成极简README
    readme_content = """# 税务咨询RAG项目-Query数据集
## 数据集说明
本数据集为高等机器学习项目专用的税务咨询用户Query数据集。

## 版本说明
| 版本 | 规模 | 用途 |
|------|------|------|
| tax_queries_small.txt | 200条 | 最小可行性验证 |
| tax_queries_medium.txt | 500条 | 中等规模测试 |
| tax_queries_large.txt | 1000条 | 完整实验 |

## 生成方式
- 种子数据：200条国家税务总局12366官方热点问答
- 扩充模型：deepseek-chat (CloseAI)
- 处理方式：已完成去重、清洗、随机打乱
"""
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✅ README.md已生成")

    print("\n" + "="*60)
    print("🎉 所有任务全部完成！")
    print(f"📁 最终交付文件已保存在：{OUTPUT_DIR}")

if __name__ == "__main__":
    # 自动安装aiohttp（如果没装）
    try:
        import aiohttp
    except ImportError:
        print("正在安装依赖aiohttp...")
        import subprocess
        subprocess.check_call(["python", "-m", "pip", "install", "aiohttp"])
        import aiohttp
    
    asyncio.run(main())