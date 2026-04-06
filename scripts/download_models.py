# -*- coding: utf-8 -*-
"""
模型下载脚本

下载 MiniLM 等 embedding 模型到本地 models 目录。
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path


def download_model(model_name: str, save_dir: str = "models"):
    """
    下载模型到本地目录。

    参数
    ----------
    model_name : str
        模型名称，如 "sentence-transformers/all-MiniLM-L6-v2"。
    save_dir : str
        保存目录，默认为 "models"。
    """
    save_path = Path(save_dir) / model_name.replace("/", "--")

    print(f"正在下载模型：{model_name}")
    print(f"保存路径：{save_path}")

    # 加载模型（会自动下载到缓存）
    model = SentenceTransformer(model_name)

    # 保存到本地目录
    model.save(str(save_path))

    print(f"模型已保存到：{save_path}")
    return save_path


if __name__ == "__main__":
    # 下载 MiniLM 模型
    download_model("sentence-transformers/all-MiniLM-L6-v2")

    # 如果需要下载其他模型，取消注释：
    # download_model("BAAI/bge-large-zh-v1.5")
