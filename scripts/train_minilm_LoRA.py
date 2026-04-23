# -*- coding: utf-8 -*-
import sys
import time
import torch
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.training.PEFT.training_LoRA import load_training_data, inject_lora, monitor_resources
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.losses import TripletDistanceMetric
from torch.utils.data import DataLoader

def train_minilm_lora(
    chunk_method: str = "semantic",
    margin: float = 0.5,
    r: int = 16,
    alpha: int = 32,
    # ... 其他与 FFT 一致的参数
):
    print("="*60 + "\nMiniLM LoRA 微调训练 (资源监控版)\n" + "="*60)
    
    start_time = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    # 1. 加载数据 & 2. 学习率查找 (参考 FFT 逻辑)
    # 3. 注入 LoRA 并开始早停训练
    model = SentenceTransformer("/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2")
    model = inject_lora(model, r=r, alpha=alpha)
    model[0].auto_model.print_trainable_parameters()

    # 此处运行早停循环 (同 FFT 逻辑) ...
    
    end_time = time.time()
    final_cpu, final_gpu = monitor_resources()
    print(f"\n[统计] 耗时: {end_time-start_time:.2f}s | 峰值显存: {final_gpu:.2f}GB | CPU负载: {final_cpu}%")

if __name__ == "__main__":
    # 此处套用 FFT 的 argparse 逻辑，增加 --r 和 --alpha 参数
    pass