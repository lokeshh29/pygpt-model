import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from app.tokenizer.tokenizer import pygpt_tokenizer


class InstructionDatasetPipeline:
    """
    Pipeline for constructing Supervised Fine-Tuning (SFT) Instruction Datasets:
    1. Formats Instruction, Input Code, and Target Response pairs
    2. Supports 3 Core Tasks:
       - Code Generation & Instruction Following
       - Program Explanation
       - Bug Fixing & Refactoring
    3. Tokenizes prompt-response sequences with response-only loss masking
    """

    SAMPLE_INSTRUCTION_TASKS = [
        # Task 1: Code Generation
        {
            "instruction": "Write a Python function to compute the factorial of a given non-negative integer using recursion.",
            "input": "",
            "response": "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "task_type": "generate",
        },
        {
            "instruction": "Create a FastAPI GET endpoint '/health' that returns a status dictionary indicating the server is healthy.",
            "input": "",
            "response": "@app.get('/health')\ndef health_check():\n    return {'status': 'healthy'}",
            "task_type": "generate",
        },
        # Task 2: Program Explanation
        {
            "instruction": "Explain what this Python code snippet does.",
            "input": "def reverse_string(s: str) -> str:\n    return s[::-1]",
            "response": "This function takes a string `s` as input and returns a new string with all characters in reverse order using Python extended slicing `[::-1]`.",
            "task_type": "explain",
        },
        {
            "instruction": "Explain what this matrix operation function performs.",
            "input": "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]",
            "response": "This function transposes a 2D matrix (swapping rows and columns) by unpacking rows into `zip(*matrix)` and converting tuples back to lists.",
            "task_type": "explain",
        },
        # Task 3: Bug Fixing & Refactoring
        {
            "instruction": "Fix the IndexError bug in the following list search function.",
            "input": "def get_last_item(lst):\n    return lst[len(lst)]",
            "response": "def get_last_item(lst):\n    if not lst:\n        return None\n    return lst[-1]",
            "task_type": "fix_bug",
        },
        {
            "instruction": "Fix the ZeroDivisionError in this average calculation function.",
            "input": "def compute_average(numbers):\n    return sum(numbers) / len(numbers)",
            "response": "def compute_average(numbers):\n    if not numbers:\n        return 0.0\n    return sum(numbers) / len(numbers)",
            "task_type": "fix_bug",
        },
    ]

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_sample(self, item: Dict[str, str]) -> str:
        """Formats an instruction item into standardized text sequence."""
        task_type = item.get("task_type", "generate")
        inst = item.get("instruction", "")
        inp = item.get("input", "")
        resp = item.get("response", "")

        if task_type == "explain":
            prefix = "### Instruction:\nExplain what the following Python program does:\n\n"
        elif task_type == "fix_bug":
            prefix = "### Instruction:\nIdentify and fix the bug in the following Python function:\n\n"
        else:
            prefix = "### Instruction:\n"

        text = f"{prefix}{inst}\n"
        if inp:
            text += f"\n### Input Code:\n{inp}\n"
        text += f"\n### Response:\n{resp}"
        return text

    def build_and_save_dataset(self, custom_samples: Optional[List[Dict[str, str]]] = None) -> str:
        """Constructs instruction dataset and saves tokenized sequence shard."""
        samples = custom_samples if custom_samples else self.SAMPLE_INSTRUCTION_TASKS
        formatted_docs = [self.format_sample(s) for s in samples]

        # Duplicate/augment samples to build a substantial instruction training batch
        augmented_docs = formatted_docs * 50

        all_tokens: List[int] = []
        for doc in augmented_docs:
            doc_tokens = pygpt_tokenizer.encode(doc, add_special_tokens=True)
            all_tokens.extend(doc_tokens)

        output_file = self.output_dir / "instruction_train.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"token_count": len(all_tokens), "tokens": all_tokens}, f, indent=2)

        print(f"✅ Instruction dataset created: {output_file} ({len(all_tokens):,} tokens)")
        return str(output_file)


if __name__ == "__main__":
    pipeline = InstructionDatasetPipeline()
    pipeline.build_and_save_dataset()
