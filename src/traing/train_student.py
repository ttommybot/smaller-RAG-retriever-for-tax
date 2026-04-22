# -*- coding: utf-8 -*-
"""
Fine-tune MiniLM student model using LoRA with triplet training data.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config


class TripletDataset(Dataset):
    """Dataset for triplet training."""

    def __init__(self, triplets: List[Dict[str, Any]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'query': triplet['query'],
            'positive': triplet['positive'],
            'negative': triplet['negative']
        }


def load_triplets(triplet_file: str) -> List[Dict[str, Any]]:
    """Load triplet data from file."""
    with open(triplet_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def train_student_model(
    triplet_file: str,
    output_dir: str,
    base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    max_seq_length: int = 512,
    evaluation_steps: int = 500,
    save_steps: int = 500,
    logging_steps: int = 100
) -> Dict[str, Any]:
    """Fine-tune student model with LoRA."""
    print("=" * 60)
    print("Training Student Model")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return {}

    print(f"\nStep 1: Loading triplet data...")
    triplets = load_triplets(triplet_file)
    print(f"Loaded {len(triplets)} triplets")

    print(f"\nStep 2: Loading base model...")
    print(f"Model: {base_model}")
    model = SentenceTransformer(base_model)

    if use_lora:
        print(f"\nStep 3: Applying LoRA...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value"],
                bias="none"
            )

            model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
            model[0].auto_model.print_trainable_parameters()

        except ImportError:
            print("Warning: peft not installed, training without LoRA")
            print("Install with: pip install peft")
            use_lora = False

    print(f"\nStep 4: Preparing training data...")
    train_examples = []
    for triplet in triplets:
        train_examples.append(InputExample(
            texts=[triplet['query'], triplet['positive'], triplet['negative']]
        ))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    print(f"\nStep 5: Setting up training...")
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Use LoRA: {use_lora}")
    if use_lora:
        print(f"  LoRA r: {lora_r}")
        print(f"  LoRA alpha: {lora_alpha}")
        print(f"  LoRA dropout: {lora_dropout}")

    print(f"\nStep 6: Training model...")
    print("-" * 60)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=True,
        save_best_model=True
    )

    print(f"\nStep 7: Saving model...")
    model.save(str(output_path))
    print(f"Model saved: {output_path}")

    training_info = {
        'base_model': base_model,
        'num_triplets': len(triplets),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'use_lora': use_lora,
        'lora_config': {
            'r': lora_r,
            'alpha': lora_alpha,
            'dropout': lora_dropout
        } if use_lora else None,
        'output_dir': str(output_path)
    }

    info_file = output_path / "training_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    print(f"Training info saved: {info_file}")

    print(f"\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    return training_info
