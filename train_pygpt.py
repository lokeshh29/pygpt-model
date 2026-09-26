#!/usr/bin/env python3
"""
CLI entry script to pretrain PyGPT model using Next-Token Prediction & Validation Loss Monitoring.

Usage:
    python3 train_pygpt.py [--full_dataset] [--model_size nano] [--epochs 3] [--batch_size 2] [--seq_len 256] [--lr 1e-4]
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
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "base", "pro"],
        help="PyGPT model variant ('nano'=350M for local CPU/RAM, 'base'=1.3B, 'pro'=7B)",
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
        default=2,
        help="Batch size per step (default: 2)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Sequence context length per sample (default: 256)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit max training sequence samples (default: None for 100%% of all tokens)",
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
        "--resume",
        action="store_true",
        help="Resume pre-training from latest checkpoint (checkpoints/latest_model.pt)",
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
    print(f"   Model Variant: {args.model_size.upper()}")
    print(f"   Dataset Usage: {'FULL (100% of all tokens)' if args.max_samples is None else f'Capped at {args.max_samples:,} samples'}")
    print(f"   Resume Mode:   {'Enabled' if args.resume else 'Disabled'}")
    print(f"   Train File:    {train_path.resolve()}")
    print(f"   Val File:      {val_path.resolve()}")
    print(f"   Epochs: {args.epochs} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len} | LR: {args.lr}")

    # Build PyGPT Configuration for Pretraining
    if args.model_size == "nano":
        config = PyGPTModelConfig(
            name="PyGPT-350M-Pretrain",
            size=ModelSize.NANO,
            total_parameters="350 Million",
        )
        config.architecture.hidden_size = 512
        config.architecture.num_hidden_layers = 6
        config.architecture.num_attention_heads = 8
        config.architecture.num_key_value_heads = 2
        config.architecture.intermediate_size = 2048
    else:
        config = PyGPTModelConfig(
            name="PyGPT-1.3B-Pretrain",
            size=ModelSize.BASE,
        )

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
        resume=args.resume,
    )

    print("\n✅ Pre-training Complete!")
    print(f"   Best Validation Loss: {results['best_val_loss']}")
    print(f"   Checkpoint Location:  {results['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__":
    main()
