# -*- coding: utf-8 -*-
"""
全参数微调训练核心模块（Full Parameter Fine-Tuning）

使用 triplet 损失对 sentence-transformers 模型进行全参数微调。
支持从 JSON 文件加载 {query, positive_chunk_id, negative_chunk_id} 三元组数据，
自动从 chunks 文件中查找对应 chunk 内容。

模型配置：
- 基座模型：sentence-transformers/all-MiniLM-L6-v2（从 /models 加载）
- 输出模型：sentence-transformers/all-MiniLM-L6-v2-FFT（保存到 /models）

使用方法：
    from retrieval.training.FFT.training_FFT import train_fft, load_training_data

    # 加载训练数据
    train_data = load_training_data(
        data_file="semantic_trained.json",
        chunks_file="chunks_semantic_cleaned.json"
    )

    # 训练
    train_fft(
        train_examples=train_data,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        output_name="sentence-transformers/all-MiniLM-L6-v2-FFT",
        batch_size=64,
        learning_rate=2e-5,
        num_epochs=5,
        margin=0.5
    )
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.losses import TripletDistanceMetric
from torch.utils.data import DataLoader, Dataset
import numpy as np


# 类型别名，用于忽略类型检查
from typing import cast


# ==================== 路径配置 ====================

def get_models_dir() -> Path:
    """获取 models 目录路径。"""
    return PROJECT_ROOT / "models"


def get_training_data_path() -> Path:
    """获取训练数据路径。"""
    return PROJECT_ROOT / "data" / "training"


def get_processed_data_dir() -> Path:
    """获取 processed 数据目录路径。"""
    return PROJECT_ROOT / "data" / "processed"


# ==================== 数据加载 ====================

def load_chunks_map(chunks_file: Path) -> Dict[str, str]:
    """
    从 chunks JSON 文件加载 chunk_id 到 content 的映射。

    参数
    ----------
    chunks_file : Path
        chunks JSON 文件路径。

    返回
    -------
    Dict[str, str]
        {chunk_id: chunk_content} 映射字典。
    """
    import json

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    return {chunk['id']: chunk['content'] for chunk in chunks}


def load_training_data(
    data_file: str,
    chunks_file: Optional[str] = None,
    chunk_method: str = "semantic"
) -> List[InputExample]:
    """
    从训练数据文件和 chunks 文件加载 triplet 训练数据。

    训练数据格式：
    {"query": "问题文本", "positive_chunk_id": "chunk_xxx", "negative_chunk_id": "chunk_yyy"}

    参数
    ----------
    data_file : str
        训练数据文件名（相对于 data/training 目录）。
    chunks_file : str
        chunks 文件名（相对于 data/processed 目录）。
    chunk_method : str
        分块方法，用于自动推断 chunks 文件名。
        'semantic' → chunks_semantic_cleaned.json
        'sliding_window' → chunks_sliding_cleaned.json

    返回
    -------
    List[InputExample]
        SentenceTransformers 格式的 InputExample 列表。
    """
    import json

    training_data_path = get_training_data_path()
    processed_data_dir = get_processed_data_dir()

    # 加载 chunks 映射
    if chunks_file is None:
        # 根据 chunk_method 自动推断
        if chunk_method == "semantic":
            chunks_file = "chunks_semantic_cleaned.json"
        elif chunk_method == "sliding_window":
            chunks_file = "chunks_sliding_cleaned.json"
        else:
            raise ValueError(f"未知的 chunk_method: {chunk_method}")

    chunks_path = processed_data_dir / chunks_file
    print(f"加载 chunks 映射：{chunks_path}")
    chunks_map = load_chunks_map(chunks_path)
    print(f"加载了 {len(chunks_map)} 个 chunks")

    # 加载训练数据
    data_path = training_data_path / data_file
    print(f"加载训练数据：{data_path}")

    examples = []
    missing_positive = 0
    missing_negative = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析失败：{e}")
                continue

            query = data.get('query', '')
            positive_id = data.get('positive_chunk_id', '')
            negative_id = data.get('negative_chunk_id', '')

            if not query:
                print(f"第 {line_num} 行：缺少 query 字段")
                continue

            # 查找 chunk 内容
            positive = chunks_map.get(positive_id)
            negative = chunks_map.get(negative_id)

            if positive is None:
                missing_positive += 1
                continue
            if negative is None:
                missing_negative += 1
                continue

            examples.append(InputExample(
                texts=[query, positive, negative],
                label=1.0
            ))

    print(f"加载完成：{len(examples)} 个有效 triplet")
    if missing_positive > 0:
        print(f"  警告：{missing_positive} 个样本的 positive_chunk_id 未找到")
    if missing_negative > 0:
        print(f"  警告：{missing_negative} 个样本的 negative_chunk_id 未找到")

    return examples


# ==================== 学习率查找器 ====================

def find_learning_rate(
    model: SentenceTransformer,
    train_examples: List[InputExample],
    batch_size: int = 64,
    min_lr: float = 1e-6,
    max_lr: float = 1e-3,
    num_steps: int = 100,
    margin: float = 0.5,
    use_cuda: bool = True
) -> Tuple[List[float], List[float]]:
    """
    使用学习率查找器找到最佳学习率。

    原理：从最小学习率开始，每个 batch 后指数增长学习率，
    记录学习率 - 损失曲线，选择损失下降最快的点。

    参数
    ----------
    model : SentenceTransformer
        要训练的模型。
    train_examples : List[InputExample]
        训练数据。
    batch_size : int
        批次大小。
    min_lr : float
        最小学习率。
    max_lr : float
        最大学习率。
    num_steps : int
        查找步数。
    margin : float
        Triplet Loss 边距。
    use_cuda : bool
        是否使用 GPU。

    返回
    -------
    Tuple[List[float], List[float]]
        (学习率列表，损失列表)，可用于绘图分析。
    """
    import torch
    from torch.optim import AdamW

    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)  # type: ignore

    # 创建 Triplet Loss - 使用 triplet_margin 参数（新版 sentence-transformers）
    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=margin,
        distance_metric=TripletDistanceMetric.COSINE
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=min_lr)

    # 学习率调度器（指数增长）
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(max_lr / min_lr) ** (1 / num_steps)
    )

    lrs = []
    losses_list = []

    model.train()
    step = 0

    print("=" * 60)
    print("学习率查找")
    print("=" * 60)

    for epoch in range(num_steps // len(train_dataloader) + 1):
        for batch in train_dataloader:
            if step >= num_steps:
                break

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)

            # 训练一步
            loss = train_loss(batch)
            losses_list.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 更新学习率
            lr_scheduler.step()

            step += 1
            if step % 10 == 0:
                print(f"  Step {step}/{num_steps}: lr={current_lr:.2e}, loss={loss.item():.4f}")

    # 找到最佳学习率（损失下降最快的点）
    best_lr = _find_best_lr_from_curve(lrs, losses_list)
    print(f"\n推荐学习率：{best_lr:.2e}")

    return lrs, losses_list


def _find_best_lr_from_curve(lrs: List[float], losses: List[float]) -> float:
    """
    从学习率 - 损失曲线中找到最佳学习率。

    方法：计算每个点的斜率，选择斜率最陡（负值最大）的点。
    """
    if len(losses) < 5:
        return lrs[len(lrs) // 2]

    # 计算滑动平均斜率
    window = 5
    best_slope = float('inf')
    best_lr = lrs[0]

    for i in range(window, len(losses) - window):
        # 计算前向斜率
        slope = (losses[i + window] - losses[i - window]) / (np.log(lrs[i + window]) - np.log(lrs[i - window]))
        if slope < best_slope:
            best_slope = slope
            best_lr = lrs[i]

    return best_lr


# ==================== 早停训练器 ====================

class EarlyStopping:
    """早停法回调。"""

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        参数
        ----------
        patience : int
            容忍多少轮不改善。
        min_delta : float
            最小改善阈值。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: Any) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def train_with_early_stopping(
    model: SentenceTransformer,
    train_examples: List[InputExample],
    eval_examples: Optional[List[InputExample]] = None,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_epochs: int = 10,
    warmup_ratio: float = 0.1,
    margin: float = 0.5,
    patience: int = 3,
    min_delta: float = 0.001,
    use_cuda: bool = True
) -> Tuple[int, float]:
    """
    使用早停法训练模型。

    参数
    ----------
    model : SentenceTransformer
        要训练的模型。
    train_examples : List[InputExample]
        训练数据。
    eval_examples : Optional[List[InputExample]]
        验证数据（用于早停判断）。如果没有，使用训练数据的一部分。
    batch_size : int
        批次大小。
    learning_rate : float
        学习率。
    max_epochs : int
        最大训练轮数。
    warmup_ratio : float
        预热步数比例（0.1 = 10%）。
    margin : float
        Triplet Loss 边距。
    patience : int
        早停容忍轮数。
    min_delta : float
        最小改善阈值。
    use_cuda : bool
        是否使用 GPU。

    返回
    -------
    Tuple[int, float]
        (实际训练轮数，最佳验证损失)。
    """
    import torch

    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)  # type: ignore

    # 验证集
    if eval_examples is None:
        # 从训练集划分 10% 作为验证集
        eval_size = max(1, len(train_examples) // 10)
        eval_examples = train_examples[-eval_size:]
        train_examples = train_examples[:-eval_size]
        train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)  # type: ignore

    eval_dataloader = DataLoader(eval_examples, batch_size=batch_size, shuffle=False)  # type: ignore

    # 创建 Triplet Loss - 使用 triplet_margin 参数（新版 sentence-transformers）
    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=margin,
        distance_metric=TripletDistanceMetric.COSINE
    )

    # 自动计算 warmup_steps
    total_steps = len(train_dataloader) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"\n训练配置:")
    print(f"  训练样本：{len(train_examples)}")
    print(f"  验证样本：{len(eval_examples)}")
    print(f"  最大轮数：{max_epochs}")
    print(f"  早停容忍：{patience} 轮")
    print(f"  Warmup: {warmup_steps} 步 ({warmup_ratio * 100:.0f}%)")

    # 早停回调
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # 训练循环
    actual_epochs = 0
    best_eval_loss = float('inf')

    for epoch in range(max_epochs):
        # 训练一轮
        model.train()
        train_losses = []

        for batch in train_dataloader:
            loss = train_loss(batch)
            loss.backward()
            torch.optim.AdamW(model.parameters(), lr=learning_rate).step()
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

        actual_epochs = epoch + 1

        # 检查早停
        if early_stopping(avg_eval_loss):
            print(f"\n早停触发：验证损失连续 {patience} 轮未改善")
            break

    return actual_epochs, best_eval_loss


# ==================== 训练主函数 ====================

def train_fft(
    train_examples: List[InputExample],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_name: str = "sentence-transformers/all-MiniLM-L6-v2-FFT",
    num_epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    margin: float = 0.5,
    use_cuda: bool = True
) -> SentenceTransformer:
    """
    执行全参数微调训练。

    参数
    ----------
    train_examples : List[InputExample]
        训练数据。
    model_name : str
        基座模型名称（用于从 models 目录加载）。
    output_name : str
        输出模型名称（保存到 models 目录）。
    num_epochs : int
        训练轮数。
    batch_size : int
        批次大小。
    learning_rate : float
        学习率。
    warmup_ratio : float
        预热步数比例（0.1 = 10%）。
    margin : float
        Triplet Loss 的边距。
    use_cuda : bool
        是否使用 GPU 训练。

    返回
    -------
    SentenceTransformer
        微调后的模型。
    """
    import torch

    models_dir = get_models_dir()

    # 加载模型
    local_model_name = model_name.replace("/", "--")
    model_path = models_dir / local_model_name

    print("=" * 60)
    print("加载模型")
    print("=" * 60)

    if model_path.exists():
        print(f"从本地加载：{model_path}")
        model = SentenceTransformer(str(model_path))
    else:
        print(f"本地未找到，从 HuggingFace 下载：{model_name}")
        model = SentenceTransformer(model_name)

    print(f"模型设备：{model.device}")

    # 自动计算 warmup_steps
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)  # type: ignore
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"\n训练配置:")
    print(f"  样本数：{len(train_examples)}")
    print(f"  Batch size: {batch_size}")
    print(f"  轮数：{num_epochs}")
    print(f"  学习率：{learning_rate}")
    print(f"  Warmup: {warmup_steps} 步 ({warmup_ratio * 100:.0f}%)")
    print(f"  Margin: {margin}")

    # 创建 Triplet Loss - 使用 triplet_margin 参数（新版 sentence-transformers）
    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=margin,
        distance_metric=TripletDistanceMetric.COSINE
    )

    # 训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        use_amp=torch.cuda.is_available() and use_cuda,
        show_progress_bar=True,
    )

    # 保存模型
    output_model_path = models_dir / output_name.replace("/", "--")
    output_model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))

    print(f"\n训练完成！")
    print(f"模型已保存到：{output_model_path}")

    return model
