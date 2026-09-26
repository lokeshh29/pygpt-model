#!/usr/bin/env python3
"""
CLI script to instruction-tune PyGPT model on task instructions, program explanations, and bug fixing.

Usage:
    python3 finetune_pygpt.py [--epochs 3] [--lr 2e-5] [--model_size nano]
"""

import argparse
import sys
from pathlib import Path

from app.dataset.instruction_pipeline import InstructionDatasetPipeline
from app.schemas.model import PyGPTModelConfig, ModelSize
from app.training.trainer import PyGPTTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Instruction-tune PyGPT model for code generation, explanation, and bug fixing."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "base"],
        help="PyGPT model variant ('nano'=350M, 'base'=1.3B)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of instruction fine-tuning epochs (default: 3)",
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
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for SFT instruction tuning (default: 2e-5)",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to pre-trained base model checkpoint",
    )

    args = parser.parse_args()

    print("🎯 PyGPT Instruction-Tuning (SFT) Pipeline")
    print(f"   Task Scope: Code Generation, Program Explanation, Bug Fixing")
    print(f"   Base Checkpoint: {args.base_checkpoint}")
    print(f"   Epochs: {args.epochs} | Learning Rate: {args.lr}")

    # 1. Build Instruction Dataset Shard
    pipeline = InstructionDatasetPipeline()
    inst_train_file = pipeline.build_and_save_dataset()

    # 2. Build PyGPT Configuration
    if args.model_size == "nano":
        config = PyGPTModelConfig(
            name="PyGPT-350M-Instruct",
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
            name="PyGPT-1.3B-Instruct",
            size=ModelSize.BASE,
        )

    config.context.native_context_length = args.seq_len

    # 3. Initialize Trainer and Load Base Checkpoint
    trainer = PyGPTTrainer(
        config=config,
        checkpoint_dir="checkpoints",
        learning_rate=args.lr,
    )

    base_ckpt = Path(args.base_checkpoint)
    if base_ckpt.exists():
        trainer.load_checkpoint(str(base_ckpt))

    # 4. Execute Fine-Tuning Loop
    results = trainer.train(
        train_file=inst_train_file,
        val_file=inst_train_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_interval=10,
    )

    # 5. Save as dedicated instruction model checkpoint
    inst_ckpt = Path("checkpoints/instruction_model.pt")
    trainer.save_checkpoint(args.epochs, 100, float(results["best_val_loss"]), is_best=True)
    print(f"\n🎉 Instruction Fine-Tuning Complete!")
    print(f"   Instruction Model Checkpoint: {inst_ckpt.resolve()}")


if __name__ == "__main__":
    main()
