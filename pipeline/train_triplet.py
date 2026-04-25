# -*- coding: utf-8 -*-
"""
基于 triplet 训练 embedding 模型。

输入：triplet jsonl 文件，格式必须包含 query / positive / negative / positive_score / negative_score / score_margin。
输出：训练后的 SentenceTransformer 模型文件夹。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from sentence_transformers import InputExample, SentenceTransformer, losses
except ImportError as exc:
    raise ImportError(
        "请先安装 sentence-transformers：pip install sentence-transformers"
    ) from exc

from torch.utils.data import DataLoader


def load_triplet_jsonl(file_path: str) -> List[Dict[str, object]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"triplet 文件不存在: {file_path}")

    triplets: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not all(k in data for k in ("query", "positive", "negative")):
                continue
            triplets.append(data)
    return triplets


def filter_triplets(
    triplets: List[Dict[str, object]],
    min_positive_score: float,
    max_negative_score: float,
    min_score_margin: float,
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for triplet in triplets:
        pos_score = float(triplet.get("positive_score", 0.0))
        neg_score = float(triplet.get("negative_score", 0.0))
        margin = float(triplet.get("score_margin", pos_score - neg_score))

        if pos_score < min_positive_score:
            continue
        if neg_score > max_negative_score:
            continue
        if margin < min_score_margin:
            continue
        filtered.append(triplet)
    return filtered


def build_examples(triplets: List[Dict[str, object]], max_samples: Optional[int] = None):
    examples: List[InputExample] = []
    for triplet in triplets[:max_samples] if max_samples else triplets:
        query = str(triplet["query"]).strip()
        positive = str(triplet["positive"]).strip()
        negative = str(triplet["negative"]).strip()
        if not query or not positive or not negative:
            continue
        examples.append(InputExample(texts=[query, positive, negative]))
    return examples


def train_triplet_model(
    triplet_path: str,
    output_path: str,
    base_model: str,
    batch_size: int,
    epochs: int,
    triplet_margin: float,
    warmup_steps: int,
    min_positive_score: float,
    max_negative_score: float,
    min_score_margin: float,
    max_samples: Optional[int],
    device: Optional[str] = None,
) -> None:
    print("[1/5] 读取 triplet 数据...")
    triplets = load_triplet_jsonl(triplet_path)
    print(f"  - 原始 triplet 数量: {len(triplets)}")

    triplets = filter_triplets(
        triplets,
        min_positive_score=min_positive_score,
        max_negative_score=max_negative_score,
        min_score_margin=min_score_margin,
    )
    print(f"  - 过滤后 triplet 数量: {len(triplets)}")
    if not triplets:
        raise ValueError("没有符合条件的 triplet。请放宽过滤条件或检查输入数据。")

    examples = build_examples(triplets, max_samples=max_samples)
    print(f"  - 训练样本数量: {len(examples)}")
    if not examples:
        raise ValueError("没有有效的训练样本。请检查 triplet 数据格式。")

    print(f"[2/5] 加载基础模型：{base_model}")
    model = SentenceTransformer(base_model, device=device)

    print("[3/5] 构建 DataLoader...")
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    print(f"[4/5] 创建 TripletLoss (margin={triplet_margin})...")
    train_loss = losses.TripletLoss(model=model, triplet_margin=triplet_margin)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[5/5] 开始训练...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    print(f"训练完成，模型已保存到: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 triplet 训练检索向量模型")
    parser.add_argument("--triplet-path", required=True, help="triplet jsonl 文件路径")
    parser.add_argument("--output-path", default="models/triplet-embedding", help="训练后模型保存目录")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2", help="基础 embedding 模型")
    parser.add_argument("--batch-size", type=int, default=16, help="训练 batch size")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--triplet-margin", type=float, default=1.0, help="Triplet loss margin")
    parser.add_argument("--warmup-steps", type=int, default=100, help="warmup steps")
    parser.add_argument("--min-positive-score", type=float, default=0.5, help="保留正样本最低分数")
    parser.add_argument("--max-negative-score", type=float, default=0.0, help="保留负样本最高分数")
    parser.add_argument("--min-score-margin", type=float, default=0.2, help="正负样本分差最低阈值")
    parser.add_argument("--max-samples", type=int, default=0, help="最多训练样本数，0 表示全部")
    parser.add_argument("--device", default=None, help="训练设备，例如 cuda 或 cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_triplet_model(
        triplet_path=args.triplet_path,
        output_path=args.output_path,
        base_model=args.base_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        triplet_margin=args.triplet_margin,
        warmup_steps=args.warmup_steps,
        min_positive_score=args.min_positive_score,
        max_negative_score=args.max_negative_score,
        min_score_margin=args.min_score_margin,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        device=args.device,
    )
