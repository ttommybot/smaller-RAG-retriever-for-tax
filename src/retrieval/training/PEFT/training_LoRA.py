# -*- coding: utf-8 -*-
import json
import time
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.losses import TripletDistanceMetric
from peft import LoraConfig, get_peft_model

def load_training_data(data_file: str, chunks_file: str, chunk_method: str) -> List[InputExample]:
    """与 FFT 保持一致的数据加载与 ID 映射逻辑"""
    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "data" / "training" / data_file
    chunks_path = project_root / "data" / "processed" / chunks_file
    
    # 加载文本字典
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        mapping = {str(c.get("id", c.get("chunk_id"))): c.get("content", c.get("text", "")) for c in chunks_data}
    
    # 加载训练三元组
    with open(data_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)
        
    examples = []
    for t in triplets:
        q, p_id, n_id = t["query"], str(t["positive_chunk_id"]), str(t["negative_chunk_id"])
        if p_id in mapping and n_id in mapping:
            examples.append(InputExample(texts=[q, mapping[p_id], mapping[n_id]]))
    return examples

def inject_lora(model: SentenceTransformer, r: int = 16, alpha: int = 32):
    """注入 LoRA 适配器"""
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["query", "value"],
        lora_dropout=0.1, bias="none"
    )
    # 针对 SentenceTransformer 结构的注入
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    return model

def monitor_resources():
    """实时监控 CPU/GPU"""
    cpu_usage = psutil.cpu_percent()
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    return cpu_usage, gpu_mem

# 注意：此处应包含你同学写的 find_learning_rate 逻辑的 LoRA 适配版
# 为了篇幅，假设其接口与 FFT 一致