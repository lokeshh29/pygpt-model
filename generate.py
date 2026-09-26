#!/usr/bin/env python3
"""
CLI script to prompt the trained PyGPT model for code generation, program explanation, and bug fixing.

Usage:
    python3 generate.py --prompt "def fibonacci(n: int) -> int:"
    python3 generate.py --instruction "Explain this function" --code "def add(a, b): return a + b" --task explain
    python3 generate.py --interactive
"""

import argparse
import sys
from pathlib import Path

from app.inference.generator import PyGPTGenerator
from app.schemas.model import PyGPTModelConfig, ModelSize


def interactive_mode(generator: PyGPTGenerator):
    print("\n🤖 PyGPT Interactive Code Assistant (Type 'exit' or 'quit' to stop)\n" + "=" * 60)
    while True:
        try:
            print("\nSelect Task Type:")
            print("  1. Code Generation / Completion")
            print("  2. Explain Program")
            print("  3. Fix Bug in Code")
            choice = input("Enter choice (1-3, default 1): ").strip()

            if choice in ["quit", "exit"]:
                break

            task_type = "generate"
            instruction = ""
            input_code = ""

            if choice == "2":
                task_type = "explain"
                instruction = "Explain what this Python code snippet does."
                print("\n💻 Enter Python Code to Explain (type/paste code, press Enter):")
                input_code = input().strip()
            elif choice == "3":
                task_type = "fix_bug"
                instruction = "Identify and fix the bug in the following Python code snippet."
                print("\n💻 Enter Python Code to Fix (type/paste code, press Enter):")
                input_code = input().strip()
            else:
                task_type = "generate"
                instruction = input("\n📝 Enter Code Instruction / Prompt (e.g. 'Write a factorial function'): ").strip()

            if not instruction and not input_code:
                continue

            print("\n⚡ PyGPT Generating Response...\n" + "-" * 50)
            res = generator.generate_instruction(
                instruction=instruction,
                input_code=input_code,
                task_type=task_type,
                max_new_tokens=150,
                temperature=0.5,
            )
            print(res["generated_text"])
            print("-" * 50)
            print(f"⏱️ Generated {res['tokens_generated']} tokens in {res['generation_time_seconds']}s")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Prompt trained PyGPT model for code generation, explanation, and bug fixing."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Raw prompt string or Python code prefix to complete",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Instruction prompt (e.g. 'Write a function to...')",
    )
    parser.add_argument(
        "--code",
        type=str,
        default="",
        help="Context input code snippet for explanation or bug fixing",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="generate",
        choices=["generate", "explain", "fix_bug"],
        help="Task preset ('generate', 'explain', 'fix_bug')",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model weights checkpoint (.pt)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal chat session",
    )

    args = parser.parse_args()

    # Build PyGPT Nano Config matching pretraining & fine-tuning
    config = PyGPTModelConfig(
        name="PyGPT-350M-Inference",
        size=ModelSize.NANO,
        total_parameters="350 Million",
    )
    config.architecture.hidden_size = 512
    config.architecture.num_hidden_layers = 6
    config.architecture.num_attention_heads = 8
    config.architecture.num_key_value_heads = 2
    config.architecture.intermediate_size = 2048

    generator = PyGPTGenerator(checkpoint_path=args.checkpoint, config=config)

    if args.interactive:
        interactive_mode(generator)
        return

    if args.instruction:
        res = generator.generate_instruction(
            instruction=args.instruction,
            input_code=args.code,
            task_type=args.task,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n🤖 PyGPT Response:\n" + "=" * 50)
        print(res["generated_text"])
        print("=" * 50)
        print(f"⚡ {res['tokens_generated']} tokens generated in {res['generation_time_seconds']}s")
    elif args.prompt:
        res = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n🤖 PyGPT Completion:\n" + "=" * 50)
        print(res["generated_text"])
        print("=" * 50)
        print(f"⚡ {res['tokens_generated']} tokens generated in {res['generation_time_seconds']}s")
    else:
        print("💡 Usage: Provide --prompt 'def foo():' or --instruction 'Write...' or run with --interactive")


if __name__ == "__main__":
    main()
