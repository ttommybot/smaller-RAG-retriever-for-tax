import re

# -------------------------- 仅需修改这里2个路径 --------------------------
INPUT_FILE = "seed_questions.txt"    # 你的原始带日期的txt文件路径
OUTPUT_FILE = "seed_questions_cleaned.txt"  # 处理后生成的干净文件路径
# -------------------------------------------------------------------------

# 读取并处理文件
cleaned_lines = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        # 核心逻辑：用正则删除行尾的 YYYY-MM-DD 格式日期，包括日期前的空格
        # 正则解释：\s* 匹配任意空格，\d{4}-\d{2}-\d{2} 匹配日期格式，$ 匹配行尾
        cleaned_line = re.sub(r"\s*\d{4}-\d{2}-\d{2}$", "", line.strip())
        # 只保留非空行
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

# 保存处理后的干净文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        f.write(line + "\n")

print(f"✅ 处理完成！共清理 {len(cleaned_lines)} 条问题")
print(f"📄 原始文件：{INPUT_FILE}（未修改，保留备份）")
print(f"📄 干净文件已生成：{OUTPUT_FILE}")





