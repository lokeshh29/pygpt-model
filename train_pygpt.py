#!/usr/bin/env python3
"""
CLI entry script to pretrain PyGPT model using Next-Token Prediction & Validation Loss Monitoring.

Usage:
    python3 train_pygpt.py [--epochs 3] [--batch_size 4] [--seq_len 512] [--lr 1e-4] [--device cuda/cpu]
"""

import argparse
import sys
from pathlib import Path

from app.schemas.model import PyGPTModelConfig, ModelSize
from app.training.trainer import PyGPTTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain PyGPT Transformer using Next-Token Prediction on dataset shards."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train_tokens.json",
        help="Path to tokenized training dataset shard (default: data/processed/train_tokens.json)",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/processed/val_tokens.json",
        help="Path to tokenized validation dataset shard (default: data/processed/val_tokens.json)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of pretraining epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per step (default: 4)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence context length per sample (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=50,
        help="Number of steps between validation loss evaluations (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device override ('cuda', 'mps', 'cpu')",
    )

    args = parser.parse_args()

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    if not train_path.exists():
        print(f"❌ Error: Training token file '{args.train_file}' not found.")
        print("💡 Hint: Run `python3 prepare_dataset.py <your_dataset.zip>` first to generate token shards.")
        sys.exit(1)

    print("🧠 PyGPT Pre-training Initialization")
    print(f"   Train File: {train_path.resolve()}")
    print(f"   Val File:   {val_path.resolve()}")
    print(f"   Epochs: {args.epochs} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len} | LR: {args.lr}")

    # Build PyGPT Configuration for Pretraining
    config = PyGPTModelConfig(
        name="PyGPT-1.3B-Pretrain",
        size=ModelSize.BASE,
    )
    # Configure model sequence length
    config.context.native_context_length = args.seq_len

    trainer = PyGPTTrainer(
        config=config,
        checkpoint_dir="checkpoints",
        learning_rate=args.lr,
        device=args.device,
    )

    results = trainer.train(
        train_file=str(train_path),
        val_file=str(val_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_interval=args.eval_interval,
    )

    print("\n✅ Training Complete!")
    print(f"   Best Validation Loss: {results['best_val_loss']}")
    print(f"   Checkpoint Location:  {results['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__":
    main()
