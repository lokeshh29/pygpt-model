import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from app.tokenizer.tokenizer import pygpt_tokenizer


class InstructionDatasetPipeline:
    """
    Pipeline for constructing Supervised Fine-Tuning (SFT) Instruction Datasets.
    Expanded with diverse Python programming tasks:
    - Loops, conditionals, odd/even calculations, math operations
    - Data structures (lists, dicts, sets, tuples, strings)
    - Functions, recursion, object-oriented programming (OOP)
    - FastAPI endpoints, error handling, and bug fixes
    """

    DIVERSE_INSTRUCTION_TASKS = [
        # Task Type: Generate (Code Generation)
        {
            "instruction": "Write a Python program to print all odd numbers from 1 to 10.",
            "input": "",
            "response": "for num in range(1, 11):\n    if num % 2 != 0:\n        print(num)",
            "task_type": "generate",
        },
        {
            "instruction": "Write a Python function to print even numbers up to n.",
            "input": "",
            "response": "def print_evens(n: int):\n    for i in range(2, n + 1, 2):\n        print(i)",
            "task_type": "generate",
        },
        {
            "instruction": "Write a Python function to check if a given number is prime.",
            "input": "",
            "response": "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            "task_type": "generate",
        },
        {
            "instruction": "Write a Python function to compute the factorial of a given non-negative integer using recursion.",
            "input": "",
            "response": "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "task_type": "generate",
        },
        {
            "instruction": "Write a Python script to reverse a list in-place.",
            "input": "",
            "response": "def reverse_list(lst: list) -> list:\n    lst.reverse()\n    return lst",
            "task_type": "generate",
        },
        {
            "instruction": "Create a FastAPI GET endpoint '/health' that returns a status dictionary indicating the server is healthy.",
            "input": "",
            "response": "@app.get('/health')\ndef health_check():\n    return {'status': 'healthy'}",
            "task_type": "generate",
        },
        {
            "instruction": "Write a Python function that counts the frequency of each word in a string.",
            "input": "",
            "response": "def word_count(text: str) -> dict:\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
            "task_type": "generate",
        },
        {
            "instruction": "Write a Python function to calculate the Fibonacci sequence up to n terms.",
            "input": "",
            "response": "def fibonacci(n: int) -> list:\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]",
            "task_type": "generate",
        },

        # Task Type: Explain (Program Explanation)
        {
            "instruction": "Explain what this Python code snippet does.",
            "input": "def reverse_string(s: str) -> str:\n    return s[::-1]",
            "response": "This function takes a string `s` as input and returns a new string with all characters in reverse order using Python extended slicing `[::-1]`.",
            "task_type": "explain",
        },
        {
            "instruction": "Explain what this matrix transpose operation performs.",
            "input": "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]",
            "response": "This function transposes a 2D matrix (swapping rows and columns) by unpacking rows into `zip(*matrix)` and converting tuples back to lists.",
            "task_type": "explain",
        },
        {
            "instruction": "Explain what this list comprehension does.",
            "input": "evens = [x for x in range(20) if x % 2 == 0]",
            "response": "This list comprehension creates a list of all even numbers from 0 to 19 by filtering `range(20)` with `x % 2 == 0`.",
            "task_type": "explain",
        },

        # Task Type: Fix Bug (Bug Fixing & Refactoring)
        {
            "instruction": "Fix the IndexError bug in the following list search function.",
            "input": "def get_last_item(lst):\n    return lst[len(lst)]",
            "response": "def get_last_item(lst):\n    if not lst:\n        return None\n    return lst[-1]",
            "task_type": "fix_bug",
        },
        {
            "instruction": "Fix the SyntaxError and ZeroDivisionError in this average calculation function.",
            "input": "def compute_average(numbers):\n    return sum(numbers) / len(numbers)",
            "response": "def compute_average(numbers):\n    if not numbers:\n        return 0.0\n    return sum(numbers) / len(numbers)",
            "task_type": "fix_bug",
        },
        {
            "instruction": "Fix the syntax error in this input print statement.",
            "input": "val = int(\"Enter number\"",
            "response": "val = int(input(\"Enter number\"))",
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

        text = f"### Instruction:\n{inst}\n"
        if inp:
            text += f"\n### Input Code:\n{inp}\n"
        text += f"\n### Response:\n{resp}\n"
        return text

    def build_and_save_dataset(self, custom_samples: Optional[List[Dict[str, str]]] = None) -> str:
        """Constructs instruction dataset and saves tokenized sequence shard."""
        samples = custom_samples if custom_samples else self.DIVERSE_INSTRUCTION_TASKS
        formatted_docs = [self.format_sample(s) for s in samples]

        # Augment diverse samples
        augmented_docs = formatted_docs * 30

        all_tokens: List[int] = []
        for doc in augmented_docs:
            doc_tokens = pygpt_tokenizer.encode(doc, add_special_tokens=True)
            all_tokens.extend(doc_tokens)

        output_file = self.output_dir / "instruction_train.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"token_count": len(all_tokens), "tokens": all_tokens}, f, indent=2)

        print(f"✅ Diverse instruction dataset created: {output_file} ({len(all_tokens):,} tokens)")
        return str(output_file)


if __name__ == "__main__":
    pipeline = InstructionDatasetPipeline()
    pipeline.build_and_save_dataset()
