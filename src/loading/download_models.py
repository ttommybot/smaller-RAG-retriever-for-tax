#!/usr/bin/env python
"""下载 HuggingFace 模型到本地目录"""

from huggingface_hub import snapshot_download
import os

# 使用相对路径（相对于项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# 下载 bge-large-zh-v1.5
print("Downloading BAAI/bge-large-zh-v1.5...")
snapshot_download(
    repo_id="BAAI/bge-large-zh-v1.5",
    local_dir=os.path.join(MODELS_DIR, "BAAI--bge-large-zh-v1.5"),
)
print("Done: BAAI/bge-large-zh-v1.5")

# 下载 bge-reranker-v2-gemma
print("Downloading BAAI/bge-reranker-v2-gemma...")
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-gemma",
    local_dir=os.path.join(MODELS_DIR, "BAAI--bge-reranker-v2-gemma"),
)
print("Done: BAAI/bge-reranker-v2-gemma")

# 下载 MiniLM 模型（注释掉，需要时取消注释）
# print("Downloading sentence-transformers/all-MiniLM-L6-v2...")
# snapshot_download(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     local_dir=os.path.join(MODELS_DIR, "sentence-transformers--all-MiniLM-L6-v2"),
# )
# print("Done: sentence-transformers/all-MiniLM-L6-v2")

print("\nAll models downloaded successfully!")
