# -*- coding: utf-8 -*-
"""
MiniLM 全参数微调训练脚本

使用学习率查找器和早停法自动确定最佳超参数，对 MiniLM 模型进行全参数微调。

固定配置：
- Batch size: 64
- Warmup steps: 10%
- Loss function: TripletLoss
- 学习率：通过学习率查找器自动确定
- 训练轮数：通过早停法自动确定
- Margin：手动指定

使用方法：
    # 使用 semantic 分块数据，margin=0.5
    python scripts/train_minilm_FFT.py --chunk-method semantic --margin 0.5

    # 使用 sliding_window 分块数据，margin=0.3
    python scripts/train_minilm_FFT.py --chunk-method sliding_window --margin 0.3
"""

import sys
from pathlib import Path
from typing import Optional

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.training.FFT.training_FFT import (
    load_training_data,
    find_learning_rate,
    get_models_dir,
)

# 导入 TripletDistanceMetric 用于 TripletLoss
from sentence_transformers.losses import TripletDistanceMetric
from torch.utils.data import DataLoader


def train_minilm_fft(
    chunk_method: str = "semantic",
    margin: float = 0.5,
    data_file: Optional[str] = None,
    chunks_file: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_name: str = "sentence-transformers/all-MiniLM-L6-v2-FFT",
    max_epochs: int = 10,
    patience: int = 3,
    use_cuda: bool = True,
    skip_lr_finder: bool = False,
    manual_lr: Optional[float] = None,
) -> None:
    """
    执行 MiniLM 全参数微调训练。

    固定配置：
    - Batch size: 64
    - Warmup: 10%
    - Loss: TripletLoss

    自动确定：
    - 学习率：通过学习率查找器
    - 训练轮数：通过早停法

    手动指定：
    - Margin：通过参数

    参数
    ----------
    chunk_method : str
        分块方法，'semantic' 或 'sliding_window'。
    margin : float
        Triplet Loss 边距，推荐 0.3 ~ 0.7。
    data_file : str, optional
        训练数据文件名。默认根据 chunk_method 自动选择：
        - semantic → semantic_trained.json
        - sliding_window → sliding_trained.json
    chunks_file : str, optional
        Chunks 文件名。默认根据 chunk_method 自动选择：
        - semantic → chunks_semantic_cleaned.json
        - sliding_window → chunks_sliding_cleaned.json
    model_name : str
        基座模型名称。
    output_name : str
        输出模型名称。
    max_epochs : int
        最大训练轮数（早停法上限）。
    patience : int
        早停容忍轮数。
    use_cuda : bool
        是否使用 GPU。
    skip_lr_finder : bool
        是否跳过学习率查找。
    manual_lr : float, optional
        手动指定学习率（跳过学习率查找器）。
    """
    import torch
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("MiniLM 全参数微调训练")
    print("=" * 60)

    # ==================== 自动推断文件名 ====================
    if data_file is None:
        if chunk_method == "semantic":
            data_file = "semantic_trained.json"
        elif chunk_method == "sliding_window":
            data_file = "sliding_trained.json"
        else:
            raise ValueError(f"未知的 chunk_method: {chunk_method}")

    print(f"\n配置:")
    print(f"  分块方法：{chunk_method}")
    print(f"  训练数据：{data_file}")
    print(f"  Margin: {margin}")
    print(f"  Batch size: 64 (固定)")
    print(f"  Warmup: 10% (固定)")
    print(f"  Loss: TripletLoss (固定)")
    print(f"  最大轮数：{max_epochs}")
    print(f"  早停容忍：{patience} 轮")

    # ==================== 步骤 1: 加载训练数据 ====================
    print("\n" + "=" * 60)
    print("步骤 1: 加载训练数据")
    print("=" * 60)

    train_examples = load_training_data(
        data_file=data_file,
        chunks_file=chunks_file,
        chunk_method=chunk_method
    )

    if not train_examples:
        print("错误：未加载到任何训练数据")
        return

    print(f"共 {len(train_examples)} 个 triplet 样本")

    # ==================== 步骤 2: 确定学习率 ====================
    print("\n" + "=" * 60)
    print("步骤 2: 确定学习率")
    print("=" * 60)

    if skip_lr_finder or manual_lr is not None:
        if manual_lr is not None:
            learning_rate = manual_lr
            print(f"使用手动指定的学习率：{learning_rate}")
        else:
            learning_rate = 2e-5
            print(f"使用默认学习率：{learning_rate}")
    else:
        print("运行学习率查找器...")
        print("这将训练约 100 个 batch，耗时 1-3 分钟")

        # 加载模型进行学习率查找
        models_dir = get_models_dir()
        local_model_name = model_name.replace("/", "--")
        model_path = models_dir / local_model_name

        if model_path.exists():
            print(f"从本地加载模型：{model_path}")
            model = SentenceTransformer(str(model_path))
        else:
            print(f"本地未找到模型，从 HuggingFace 下载：{model_name}")
            model = SentenceTransformer(model_name)

        # 运行学习率查找器
        lrs, losses_list = find_learning_rate(
            model=model,
            train_examples=train_examples,
            batch_size=64,
            margin=margin,
            use_cuda=use_cuda
        )

        # 从曲线中找到最佳学习率
        best_idx = 0
        best_slope = float('inf')

        for i in range(5, len(losses_list) - 5):
            slope = (losses_list[i + 5] - losses_list[i - 5]) / (np.log(lrs[i + 5]) - np.log(lrs[i - 5]))
            if slope < best_slope:
                best_slope = slope
                best_idx = i

        learning_rate = lrs[best_idx]
        print(f"\n找到最佳学习率：{learning_rate:.2e}")

        # 保存学习率查找结果用于绘图
        lr_finder_output = PROJECT_ROOT / "data" / "training" / "lr_finder_results.json"
        lr_finder_output.parent.mkdir(parents=True, exist_ok=True)
        with open(lr_finder_output, 'w', encoding='utf-8') as f:
            json.dump({
                'learning_rates': lrs,
                'losses': losses_list,
                'best_lr': learning_rate,
                'best_idx': best_idx
            }, f, ensure_ascii=False, indent=2)
        print(f"学习率查找结果已保存到：{lr_finder_output}")

    # ==================== 步骤 3: 训练模型（早停法） ====================
    print("\n" + "=" * 60)
    print("步骤 3: 训练模型（早停法）")
    print("=" * 60)

    # 加载模型
    models_dir = get_models_dir()
    local_model_name = model_name.replace("/", "--")
    model_path = models_dir / local_model_name

    if model_path.exists():
        print(f"从本地加载：{model_path}")
        model = SentenceTransformer(str(model_path))
    else:
        print(f"本地未找到，从 HuggingFace 下载：{model_name}")
        model = SentenceTransformer(model_name)

    # 早停法训练
    batch_size = 64
    warmup_ratio = 0.1

    # 划分训练集和验证集
    eval_size = max(1, len(train_examples) // 10)
    eval_examples = train_examples[-eval_size:]
    train_examples_actual = train_examples[:-eval_size]

    print(f"\n训练集：{len(train_examples_actual)} 样本")
    print(f"验证集：{len(eval_examples)} 样本")

    train_dataloader = DataLoader(train_examples_actual, batch_size=batch_size, shuffle=True)  # type: ignore
    eval_dataloader = DataLoader(eval_examples, batch_size=batch_size, shuffle=False)  # type: ignore

    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=margin,
        distance_metric=TripletDistanceMetric.COSINE
    )

    # 配置优化器 - 直接存储在 model 属性中
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_eval_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"\n开始训练（最多 {max_epochs} 轮，早停容忍 {patience} 轮）")
    print(f"学习率：{learning_rate:.2e}, Margin: {margin}")

    for epoch in range(max_epochs):
        # 训练一轮
        model.train()
        train_losses = []

        for batch in train_dataloader:
            loss = train_loss(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        # 评估一轮
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_dataloader:
                loss = train_loss(batch)
                eval_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_eval_loss = np.mean(eval_losses)

        print(f"\nEpoch {epoch + 1}/{max_epochs}:")
        print(f"  训练损失：{avg_train_loss:.4f}")
        print(f"  验证损失：{avg_eval_loss:.4f}")

        # 检查是否改善
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  ✓ 验证损失改善（最佳：{best_eval_loss:.4f}）")
        else:
            patience_counter += 1
            print(f"  - 未改善（容忍：{patience_counter}/{patience}）")

        # 检查早停
        if patience_counter >= patience:
            print(f"\n早停触发：验证损失连续 {patience} 轮未改善")
            break

    actual_epochs = epoch + 1

    print(f"\n训练完成:")
    print(f"  实际训练轮数：{actual_epochs}")
    print(f"  最佳轮数：{best_epoch} (验证损失：{best_eval_loss:.4f})")

    # 保存模型
    output_model_path = models_dir / output_name.replace("/", "--")
    output_model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))

    print(f"\n微调后的模型已保存到：{output_model_path}")
    print(f"模型名称：{output_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniLM 全参数微调训练脚本")
    parser.add_argument(
        "--chunk-method",
        type=str,
        choices=["semantic", "sliding_window"],
        default="semantic",
        help="分块方法，默认 semantic"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Triplet Loss 边距，默认 0.5"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="训练数据文件名（默认根据 chunk_method 自动选择）"
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=None,
        help="Chunks 文件名（默认根据 chunk_method 自动选择）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="基座模型名称"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2-FFT",
        help="输出模型名称"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="最大训练轮数"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="早停容忍轮数"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="不使用 GPU 训练"
    )
    parser.add_argument(
        "--skip-lr-finder",
        action="store_true",
        help="跳过学习率查找器"
    )
    parser.add_argument(
        "--manual-lr",
        type=float,
        default=None,
        help="手动指定学习率（跳过学习率查找器）"
    )

    args = parser.parse_args()

    train_minilm_fft(
        chunk_method=args.chunk_method,
        margin=args.margin,
        data_file=args.data_file,
        chunks_file=args.chunks_file,
        model_name=args.model_name,
        output_name=args.output_name,
        max_epochs=args.max_epochs,
        patience=args.patience,
        use_cuda=not args.no_cuda,
        skip_lr_finder=args.skip_lr_finder,
        manual_lr=args.manual_lr,
    )
