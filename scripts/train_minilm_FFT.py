# -*- coding: utf-8 -*-
"""
MiniLM 全参数微调训练脚本（批量训练版）

循环训练 6 种组合：2 种 chunk 方法 × 3 种 margin
- chunk_method: semantic, sliding
- margin: 0.3, 0.4, 0.5

固定配置：
- Batch size: 64
- Warmup steps: 10%
- Loss function: TripletLoss

自动确定：
- 学习率：通过学习率查找器
- 训练轮数：通过早停法

记录：
- 训练时间
- 显存占用
- 输出对比表格到根目录

使用方法：
    python scripts/train_minilm_FFT.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.training.FFT.training_FFT import (
    load_training_data,
    find_learning_rate,
    get_models_dir,
    get_training_data_path,
    get_processed_data_dir,
    load_chunks_map,
    move_features_to_device,
)

from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
import yaml


# ==================== 加载配置 ====================

def load_config() -> Dict[str, Any]:
    """从 configs/configs.yaml 加载训练配置。"""
    config_path = PROJECT_ROOT / "configs" / "configs.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('fft', {})


# 加载配置
_config = load_config()
CHUNK_METHODS = _config.get('chunk_methods', ["semantic", "sliding"])
MARGINS = _config.get('margins', [0.3, 0.4, 0.5])
BATCH_SIZE = _config.get('batch_size', 64)
WARMUP_RATIO = _config.get('warmup_ratio', 0.1)
MAX_EPOCHS = _config.get('max_epochs', 20)
PATIENCE = _config.get('patience', 3)


# ==================== 单次训练函数 ====================

def train_single_config(
    chunk_method: str,
    margin: float,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_cuda: bool = True,
    skip_lr_finder: bool = False,
    manual_lr: float = 2e-5,
) -> Dict[str, Any]:
    """
    执行单次 MiniLM 微调训练。

    参数
    ----------
    chunk_method : str
        分块方法，'semantic' 或 'sliding'。
    margin : float
        Triplet Loss 边距。
    model_name : str
        基座模型名称。
    use_cuda : bool
        是否使用 GPU。
    skip_lr_finder : bool
        是否跳过学习率查找器。
    manual_lr : float
        手动指定的学习率。

    返回
    -------
    Dict[str, Any]
        训练结果，包含时间、显存、模型路径等。
    """
    # 结果字典
    result = {
        "chunk_method": chunk_method,
        "margin": margin,
        "train_time": 0,
        "memory_mb": 0,
        "actual_epochs": 0,
        "best_eval_loss": float('inf'),
        "learning_rate": manual_lr,
        "output_model": "",
        "train_samples": 0,
        "eval_samples": 0,
    }

    # 重置显存统计
    if use_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 开始计时
    start_time = time.time()

    print("\n" + "=" * 60)
    print(f"训练配置: {chunk_method} + margin={margin}")
    print("=" * 60)

    # ==================== 加载训练数据 ====================
    print("\n步骤 1: 加载训练数据")

    # 文件名推断
    if chunk_method == "semantic":
        data_file = "semantic_triplets.json"
        chunks_file = "chunks_semantic_cleaned.json"
    elif chunk_method == "sliding":
        data_file = "sliding_triplets.json"
        chunks_file = "chunks_sliding_cleaned.json"
    else:
        raise ValueError(f"未知的 chunk_method: {chunk_method}")

    # 加载 chunks 映射
    chunks_path = get_processed_data_dir() / chunks_file
    chunks_map = load_chunks_map(chunks_path)
    print(f"  加载 {len(chunks_map)} 个 chunks")

    # 加载训练数据
    data_path = get_training_data_path() / data_file
    with open(data_path, 'r', encoding='utf-8') as f:
        triplet_list = json.load(f)

    # 构建 InputExample
    train_examples = []
    for triplet in triplet_list:
        query = triplet.get('query', '')
        positive_id = triplet.get('positive_chunk_id', '')
        negative_id = triplet.get('negative_chunk_id', '')

        positive = chunks_map.get(positive_id)
        negative = chunks_map.get(negative_id)

        if query and positive and negative:
            train_examples.append(InputExample(
                texts=[query, positive, negative],
                label=1.0
            ))

    print(f"  加载 {len(train_examples)} 个 triplet 样本")

    if not train_examples:
        print("  错误：未加载到任何训练数据")
        return result

    # ==================== 确定学习率 ====================
    print("\n步骤 2: 确定学习率")

    models_dir = get_models_dir()
    local_model_name = model_name.replace("/", "--")
    model_path = models_dir / local_model_name

    if model_path.exists():
        print(f"  从本地加载模型：{model_path}")
        model = SentenceTransformer(str(model_path))
    else:
        print(f"  本地未找到，从 HuggingFace 下载：{model_name}")
        model = SentenceTransformer(model_name)

    if not skip_lr_finder:
        print("  运行学习率查找器...")
        lrs, losses_list = find_learning_rate(
            model=model,
            train_examples=train_examples,
            batch_size=BATCH_SIZE,
            margin=margin,
            use_cuda=use_cuda
        )

        # 找最佳学习率
        best_idx = 0
        best_slope = float('inf')
        for i in range(5, len(losses_list) - 5):
            slope = (losses_list[i + 5] - losses_list[i - 5]) / (np.log(lrs[i + 5]) - np.log(lrs[i - 5]))
            if slope < best_slope:
                best_slope = slope
                best_idx = i

        learning_rate = lrs[best_idx]
        print(f"  找到最佳学习率：{learning_rate:.2e}")
        result["learning_rate"] = learning_rate

        # 重新加载模型（学习率查找会修改模型）
        if model_path.exists():
            model = SentenceTransformer(str(model_path))
        else:
            model = SentenceTransformer(model_name)
    else:
        learning_rate = manual_lr
        print(f"  使用手动指定的学习率：{learning_rate}")

    # ==================== 早停法训练 ====================
    print("\n步骤 3: 早停法训练")

    # 划分训练集和验证集
    eval_size = max(1, len(train_examples) // 10)
    eval_examples = train_examples[-eval_size:]
    train_examples_actual = train_examples[:-eval_size]

    result["train_samples"] = len(train_examples_actual)
    result["eval_samples"] = len(eval_examples)

    print(f"  训练集：{len(train_examples_actual)} 样本")
    print(f"  验证集：{len(eval_examples)} 样本")

    # 使用自定义 collate_fn 构建 DataLoader（新版 sentence-transformers 5.x）
    def identity_collate(batch):
        return batch

    train_dataloader = DataLoader(train_examples_actual, batch_size=BATCH_SIZE, shuffle=True, collate_fn=identity_collate)
    eval_dataloader = DataLoader(eval_examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=identity_collate)

    # 创建 TripletLoss（使用 PyTorch TripletMarginLoss）
    train_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_eval_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"  开始训练（最多 {MAX_EPOCHS} 轮，早停容忍 {PATIENCE} 轮）")

    for epoch in range(MAX_EPOCHS):
        # 训练一轮
        model.train()
        train_losses = []

        for batch in train_dataloader:
            # 从 InputExample 提取文本
            texts = [example.texts for example in batch]
            anchors = [t[0] for t in texts]
            positives = [t[1] for t in texts]
            negatives = [t[2] for t in texts]

            # tokenize + forward 计算 embeddings（保留梯度）
            device = model.device
            anchor_features = move_features_to_device(model.tokenize(anchors), device)
            positive_features = move_features_to_device(model.tokenize(positives), device)
            negative_features = move_features_to_device(model.tokenize(negatives), device)

            anchor_emb = model(anchor_features)['sentence_embedding']
            positive_emb = model(positive_features)['sentence_embedding']
            negative_emb = model(negative_features)['sentence_embedding']

            # 计算 triplet loss (使用 PyTorch TripletMarginLoss)
            loss = train_loss_fn(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        # 评估一轮
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_dataloader:
                texts = [example.texts for example in batch]
                anchors = [t[0] for t in texts]
                positives = [t[1] for t in texts]
                negatives = [t[2] for t in texts]

                device = model.device
                anchor_features = move_features_to_device(model.tokenize(anchors), device)
                positive_features = move_features_to_device(model.tokenize(positives), device)
                negative_features = move_features_to_device(model.tokenize(negatives), device)

                anchor_emb = model(anchor_features)['sentence_embedding']
                positive_emb = model(positive_features)['sentence_embedding']
                negative_emb = model(negative_features)['sentence_embedding']

                loss = train_loss_fn(anchor_emb, positive_emb, negative_emb)
                eval_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_eval_loss = np.mean(eval_losses)

        print(f"    Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, eval_loss={avg_eval_loss:.4f}")

        # 检查是否改善
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        # 检查早停
        if patience_counter >= PATIENCE:
            print(f"    早停触发：验证损失连续 {PATIENCE} 轮未改善")
            break

    actual_epochs = epoch + 1
    result["actual_epochs"] = actual_epochs
    result["best_eval_loss"] = best_eval_loss

    print(f"  训练完成：{actual_epochs} 轮，最佳验证损失：{best_eval_loss:.4f}")

    # ==================== 保存模型 ====================
    print("\n步骤 4: 保存模型")

    output_model_name = f"sentence-transformers--all-MiniLM-L6-v2-FFT-{chunk_method}-{margin}"
    output_model_path = models_dir / output_model_name
    output_model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))

    result["output_model"] = str(output_model_path)
    print(f"  模型保存到：{output_model_path}")

    # ==================== 记录时间和显存 ====================
    elapsed_time = time.time() - start_time
    result["train_time"] = elapsed_time
    print(f"  训练时间：{elapsed_time:.2f} 秒")

    if use_cuda and torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        result["memory_mb"] = memory_mb
        print(f"  显存占用：{memory_mb:.2f} MB")

    return result


# ==================== 批量训练函数 ====================

def train_all_configs(
    use_cuda: bool = True,
    skip_lr_finder: bool = False,
    manual_lr: float = 2e-5,
) -> List[Dict[str, Any]]:
    """
    执行 6 次训练（2 种 chunk 方法 × 3 种 margin）。

    参数
    ----------
    use_cuda : bool
        是否使用 GPU。
    skip_lr_finder : bool
        是否跳过学习率查找器。
    manual_lr : float
        手动指定的学习率。

    返回
    -------
    List[Dict[str, Any]]
        所有训练结果列表。
    """
    all_results = []

    print("=" * 60)
    print("MiniLM 全参数微调批量训练")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  Chunk 方法：{CHUNK_METHODS}")
    print(f"  Margin：{MARGINS}")
    print(f"  共 6 次训练")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Warmup: {WARMUP_RATIO * 100:.0f}%")
    print(f"  最大轮数：{MAX_EPOCHS}")
    print(f"  早停容忍：{PATIENCE} 轮")
    print(f"  使用 GPU：{use_cuda and torch.cuda.is_available()}")

    # 循环训练
    for chunk_method in CHUNK_METHODS:
        for margin in MARGINS:
            result = train_single_config(
                chunk_method=chunk_method,
                margin=margin,
                use_cuda=use_cuda,
                skip_lr_finder=skip_lr_finder,
                manual_lr=manual_lr,
            )
            all_results.append(result)

    return all_results


# ==================== 保存结果表格 ====================

def save_results_table(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    保存训练结果对比表格。

    参数
    ----------
    results : List[Dict[str, Any]]
        训练结果列表。
    output_path : Path
        输出文件路径。
    """
    # 保存 JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到：{json_path}")

    # 保存 Markdown 表格
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# MiniLM 全参数微调训练结果对比\n\n")
        f.write(f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 训练配置\n\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n")
        f.write(f"- Warmup: {WARMUP_RATIO * 100:.0f}%\n")
        f.write(f"- Loss: TripletLoss\n")
        f.write(f"- 最大轮数：{MAX_EPOCHS}\n")
        f.write(f"- 早停容忍：{PATIENCE} 轮\n\n")

        f.write("## 结果对比\n\n")
        f.write("| Chunk 方法 | Margin | 训练时间(s) | 显存(MB) | 实际轮数 | 最佳验证损失 | 学习率 | 模型路径 |\n")
        f.write("|------------|--------|-------------|----------|----------|--------------|--------|----------|\n")

        for r in results:
            f.write(f"| {r['chunk_method']} | {r['margin']} | {r['train_time']:.2f} | {r['memory_mb']:.2f} | {r['actual_epochs']} | {r['best_eval_loss']:.4f} | {r['learning_rate']:.2e} | models/\n")

        f.write("\n---\n\n")
        f.write("## 详细结果\n\n")

        for r in results:
            f.write(f"### {r['chunk_method']} + margin={r['margin']}\n\n")
            f.write(f"- 训练样本：{r['train_samples']}\n")
            f.write(f"- 验证样本：{r['eval_samples']}\n")
            f.write(f"- 实际训练轮数：{r['actual_epochs']}\n")
            f.write(f"- 最佳验证损失：{r['best_eval_loss']:.4f}\n")
            f.write(f"- 训练时间：{r['train_time']:.2f} 秒\n")
            f.write(f"- 显存占用：{r['memory_mb']:.2f} MB\n")
            f.write(f"- 学习率：{r['learning_rate']:.2e}\n")
            f.write(f"- 模型路径：{r['output_model']}\n\n")

    print(f"Markdown 表格已保存到：{md_path}")


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniLM 全参数微调批量训练脚本")
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="不使用 GPU 训练"
    )
    parser.add_argument(
        "--skip-lr-finder",
        action="store_true",
        help="跳过学习率查找器，使用默认学习率"
    )
    parser.add_argument(
        "--manual-lr",
        type=float,
        default=2e-5,
        help="手动指定的学习率，默认 2e-5"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="training_results_FFT",
        help="结果文件名（保存到项目根目录）"
    )

    args = parser.parse_args()

    # 执行批量训练
    results = train_all_configs(
        use_cuda=not args.no_cuda,
        skip_lr_finder=args.skip_lr_finder,
        manual_lr=args.manual_lr,
    )

    # 保存结果表格
    output_path = PROJECT_ROOT / args.output_name
    save_results_table(results, output_path)

    print("\n" + "=" * 60)
    print("批量训练完成")
    print("=" * 60)
    print(f"共训练 {len(results)} 次")
    print(f"结果保存到：{PROJECT_ROOT}")