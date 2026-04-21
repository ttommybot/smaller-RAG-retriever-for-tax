#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniLM LoRA 微调脚本
使用 MultipleNegativesRankingLoss 对 sentence-transformers/all-MiniLM-L6-v2 进行微调
适配税务知识 RAG 检索场景
"""

import os
import math
import json
from datetime import datetime
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback, TrainingArguments


# ==================== 配置参数 ====================
# 数据路径
DATA_PATH = "data/processed/triplets_train.jsonl"

# 模型配置
BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models/minilm-lora-finetuned"

# LoRA 配置
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# MiniLM 使用 BERT 架构， attention 层模块名为 query/key/value
TARGET_MODULES = ["query", "value"]

# 训练超参数
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
FP16 = True

# 其他配置
MAX_SEQ_LENGTH = 512
SEED = 42


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    打印模型中可训练参数的数量和比例
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = 100 * trainable_params / all_params if all_params > 0 else 0
    print(f"\n{'='*60}")
    print(f"可训练参数统计:")
    print(f"  - 可训练参数数量: {trainable_params:,}")
    print(f"  - 总参数数量: {all_params:,}")
    print(f"  - 可训练参数占比: {ratio:.4f}%")
    print(f"{'='*60}\n")


def load_triplets_data(file_path: str) -> List[InputExample]:
    """
    加载三元组数据并转换为 InputExample 格式

    数据格式: {query, pos, neg}
    - query: 税务查询
    - pos: 相关文档块
    - neg: 困难负样本
    """
    print(f"正在加载数据: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                query = data.get("query", "").strip()
                pos = data.get("pos", "").strip()
                neg = data.get("neg", "").strip()

                # 跳过空数据
                if not all([query, pos, neg]):
                    print(f"警告: 第 {line_num} 行存在空字段，已跳过")
                    continue

                # InputExample 格式: texts=[query, positive, negative]
                examples.append(InputExample(texts=[query, pos, neg]))

            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue

    print(f"成功加载 {len(examples)} 个训练样本\n")
    return examples


def setup_lora_model(model_name: str) -> SentenceTransformer:
    """
    加载基础模型并注入 LoRA 适配器
    """
    print(f"正在加载基础模型: {model_name}")

    # 加载基础模型
    model = SentenceTransformer(model_name)
    model.max_seq_length = MAX_SEQ_LENGTH

    # 配置 LoRA
    # 注意: SentenceTransformer 底层是 transformers 模型
    # 我们需要对内部的 auto_model 应用 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    # 获取内部的 Transformer 模型并应用 LoRA
    base_transformer = model._first_module().auto_model

    # 注入 LoRA 适配器
    peft_model = get_peft_model(base_transformer, peft_config)

    # 将 PEFT 模型替换回 SentenceTransformer
    model._first_module().auto_model = peft_model

    print("LoRA 适配器注入成功")
    print(f"  - LoRA rank (r): {LORA_R}")
    print(f"  - LoRA alpha: {LORA_ALPHA}")
    print(f"  - 目标模块: {TARGET_MODULES}")

    return model


def main():
    """
    主训练流程
    """
    print(f"\n{'='*60}")
    print(f"MiniLM LoRA 微调启动")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 设置随机种子
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 检查 GPU 可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"使用 GPU: {gpu_name}")
        print(f"显存大小: {gpu_memory:.2f} GB\n")
    else:
        print("警告: 未检测到 GPU，将使用 CPU 训练（速度会很慢）\n")

    # ========== 1. 加载数据 ==========
    train_examples = load_triplets_data(DATA_PATH)

    if len(train_examples) == 0:
        raise ValueError("没有加载到任何训练数据，请检查数据文件")

    # ========== 2. 初始化模型 ==========
    model = setup_lora_model(BASE_MODEL_NAME)

    # 打印可训练参数
    print_trainable_parameters(model._first_module().auto_model)

    # ========== 3. 配置损失函数 ==========
    # MultipleNegativesRankingLoss: 使用 In-batch Negatives 机制
    # 每个 batch 中，其他样本的 positive 会被当作当前样本的 negative
    train_loss = losses.MultipleNegativesRankingLoss(model)
    print(f"损失函数: MultipleNegativesRankingLoss (MNRL)")
    print(f"  - 利用 In-batch Negatives 机制最大化显存收益\n")

    # ========== 4. 配置训练参数 ==========
    # 计算总训练步数
    total_steps = math.ceil(len(train_examples) / BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"训练配置:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 训练轮数: {NUM_EPOCHS}")
    print(f"  - 总训练步数: {total_steps}")
    print(f"  - Warmup 步数: {warmup_steps} ({WARMUP_RATIO*100:.0f}%)")
    print(f"  - 混合精度 (FP16): {FP16}\n")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # SentenceTransformer 训练参数
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        fp16=FP16 and torch.cuda.is_available(),
        save_strategy="epoch",
        logging_steps=max(1, total_steps // 20),  # 每 5% 记录一次
        logging_first_step=True,
        seed=SEED,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        dataloader_pin_memory=torch.cuda.is_available(),
        # LoRA 相关: 不保存完整模型，只保存适配器
        save_safetensors=True,
    )

    # ========== 5. 创建 Trainer ==========
    # 使用 NoDuplicateDataLoader 确保每个 epoch 没有重复样本
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    # 创建数据加载器
    train_dataloader = NoDuplicatesDataLoader(
        train_examples,
        batch_size=BATCH_SIZE
    )

    # 创建 Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        loss=train_loss,
    )

    # ========== 6. 开始训练 ==========
    print(f"{'='*60}")
    print(f"开始训练...")
    print(f"{'='*60}\n")

    trainer.train()

    # ========== 7. 保存模型 ==========
    print(f"\n{'='*60}")
    print(f"训练完成，正在保存模型...")
    print(f"{'='*60}\n")

    # 保存 LoRA 适配器权重
    peft_model = model._first_module().auto_model

    if isinstance(peft_model, PeftModel):
        # 保存适配器权重
        peft_model.save_pretrained(OUTPUT_DIR)
        print(f"LoRA 适配器权重已保存至: {OUTPUT_DIR}")

        # 保存配置文件
        config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
        if os.path.exists(config_path):
            print(f"适配器配置已保存: {config_path}")

        # 保存基础模型名称信息
        info = {
            "base_model": BASE_MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "trained_at": datetime.now().isoformat(),
        }
        info_path = os.path.join(OUTPUT_DIR, "training_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"训练信息已保存: {info_path}")

    # 同时保存为 SentenceTransformer 格式（包含合并的 LoRA 权重）
    merged_output_dir = os.path.join(OUTPUT_DIR, "merged_model")
    os.makedirs(merged_output_dir, exist_ok=True)

    # 保存完整模型（用于推理）
    model.save(merged_output_dir)
    print(f"完整模型已保存至: {merged_output_dir}")

    # 保存 tokenizer
    model.tokenizer.save_pretrained(merged_output_dir)
    print(f"Tokenizer 已保存")

    print(f"\n{'='*60}")
    print(f"✅ 模型保存成功！")
    print(f"{'='*60}")
    print(f"\n输出目录结构:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── adapter_model.safetensors  # LoRA 权重")
    print(f"    ├── adapter_config.json        # LoRA 配置")
    print(f"    ├── training_info.json         # 训练信息")
    print(f"    └── merged_model/              # 完整模型（用于推理）")
    print(f"        ├── pytorch_model.bin")
    print(f"        ├── config.json")
    print(f"        └── tokenizer.json")
    print(f"\n使用方式:")
    print(f"  1. 加载 LoRA 适配器: 使用 PeftModel.from_pretrained()")
    print(f"  2. 直接推理: SentenceTransformer('{merged_output_dir}')")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
