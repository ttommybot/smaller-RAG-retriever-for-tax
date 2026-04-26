# -*- coding: utf-8 -*-
"""
MiniLM LoRA 微调训练脚本（批量训练对齐版）
循环训练 6 种组合：2 种 chunk 方法 × 3 种 margin
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np
import json

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 复用 FFT 写好的底层工具
from retrieval.training.FFT.training_FFT import (
    load_training_data,
    find_learning_rate,
    get_models_dir,
    get_training_data_path,
    get_processed_data_dir,
    load_chunks_map,
)

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.losses import TripletDistanceMetric
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

# ==================== 训练配置 ====================
CHUNK_METHODS = ["semantic", "sliding"]
MARGINS = [0.3, 0.4, 0.5]
BATCH_SIZE = 64
WARMUP_RATIO = 0.1
MAX_EPOCHS = 10
PATIENCE = 3
LORA_R = 16
LORA_ALPHA = 32

def inject_lora(model: SentenceTransformer) -> SentenceTransformer:
    """注入 LoRA 适配器"""
    lora_config = LoraConfig(
        r=LORA_R, 
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=0.1, 
        bias="none"
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    return model

def train_single_lora_config(chunk_method: str, margin: float, manual_lr: float = 2e-4) -> Dict[str, Any]:
    """执行单次 LoRA 微调"""
    result = {
        "chunk_method": chunk_method, "margin": margin,
        "train_time": 0, "memory_mb": 0, "actual_epochs": 0,
        "best_eval_loss": float('inf'), "learning_rate": manual_lr,
        "output_model": "", "train_samples": 0, "eval_samples": 0,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    print(f"\n{'='*60}\nLoRA 训练配置: {chunk_method} + margin={margin}\n{'='*60}")

    # 1. 加载数据
    data_file = f"{chunk_method}_triplets.json" if chunk_method == "semantic" else "sliding_triplets.json"
    chunks_file = f"chunks_{chunk_method}_cleaned.json"
    
    chunks_map = load_chunks_map(get_processed_data_dir() / chunks_file)
    with open(get_training_data_path() / data_file, 'r', encoding='utf-8') as f:
        triplet_list = json.load(f)

    train_examples = []
    for t in triplet_list:
        q, p_id, n_id = t.get('query'), t.get('positive_chunk_id'), t.get('negative_chunk_id')
        if q and chunks_map.get(p_id) and chunks_map.get(n_id):
            train_examples.append(InputExample(texts=[q, chunks_map[p_id], chunks_map[n_id]], label=1.0))

    # 2. 加载基础模型并注入 LoRA
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_path = get_models_dir() / model_name.replace("/", "--")
    model = SentenceTransformer(str(model_path) if model_path.exists() else model_name)
    model = inject_lora(model)
    
    learning_rate = manual_lr 

    # 3. 早停法训练准备
    eval_size = max(1, len(train_examples) // 10)
    eval_examples, train_examples_actual = train_examples[-eval_size:], train_examples[:-eval_size]
    result["train_samples"], result["eval_samples"] = len(train_examples_actual), len(eval_examples)

    train_dataloader = DataLoader(train_examples_actual, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_examples, batch_size=BATCH_SIZE, shuffle=False)

    train_loss = losses.TripletLoss(model=model, triplet_margin=margin, distance_metric=TripletDistanceMetric.COSINE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_eval_loss, best_epoch, patience_counter = float('inf'), 0, 0

    # 4. 训练循环
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []
        for batch in train_dataloader:
            loss = train_loss(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_dataloader:
                eval_losses.append(train_loss(batch).item())

        avg_train_loss, avg_eval_loss = np.mean(train_losses), np.mean(eval_losses)
        print(f"    Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, eval_loss={avg_eval_loss:.4f}")

        if avg_eval_loss < best_eval_loss:
            best_eval_loss, best_epoch, patience_counter = avg_eval_loss, epoch + 1, 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    早停触发：连续 {PATIENCE} 轮未改善")
                break

    result["actual_epochs"] = epoch + 1
    result["best_eval_loss"] = best_eval_loss

    # 5. 保存与统计
    output_model_name = f"sentence-transformers--all-MiniLM-L6-v2-LoRA-{chunk_method}-{margin}"
    output_model_path = get_models_dir() / output_model_name
    output_model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))
    result["output_model"] = str(output_model_path)

    result["train_time"] = time.time() - start_time
    if torch.cuda.is_available():
        result["memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  耗时: {result['train_time']:.2f}s | 显存: {result.get('memory_mb', 0):.2f}MB")
    return result

def save_lora_results(results: List[Dict[str, Any]], output_path: Path):
    """保存 Markdown 对比表格"""
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# MiniLM LoRA 微调训练结果对比\n\n")
        f.write(f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
        f.write("## 训练配置\n\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n- Loss: TripletLoss\n- LoRA Rank: {LORA_R}\n\n")
        f.write("## 结果对比\n\n")
        f.write("| Chunk 方法 | Margin | 训练时间(s) | 显存(MB) | 实际轮数 | 最佳验证损失 | 学习率 | 模型路径 |\n")
        f.write("|------------|--------|-------------|----------|----------|--------------|--------|----------|\n")
        for r in results:
            f.write(f"| {r['chunk_method']} | {r['margin']} | {r['train_time']:.2f} | {r.get('memory_mb',0):.2f} | {r['actual_epochs']} | {r['best_eval_loss']:.4f} | {r['learning_rate']:.2e} | models/\n")
    print(f"Markdown 表格已保存到：{md_path}")

if __name__ == "__main__":
    results = []
    for chunk in CHUNK_METHODS:
        for margin in MARGINS:
            results.append(train_single_lora_config(chunk, margin))
    save_lora_results(results, PROJECT_ROOT / "lora_training_results")