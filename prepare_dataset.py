#!/usr/bin/env python3
"""
CLI script to prepare, clean, deduplicate, and tokenize dataset zip files for PyGPT pre-training.

Usage:
    python3 prepare_dataset.py "/mnt/c/Users/LOKESH KUMAR R/Downloads/python_first_ds.zip"
"""

import argparse
import os
import sys
from pathlib import Path

from app.dataset.pipeline import DatasetPipeline


def convert_path(path_str: str) -> Path:
    """Converts Windows style paths (e.g., C:\\Users\\...) to WSL paths (/mnt/c/Users/...) if running on Posix/WSL."""
    clean_str = path_str.strip('"\'')
    if os.name == "posix" and len(clean_str) >= 2 and clean_str[1] == ":":
        drive = clean_str[0].lower()
        rest = clean_str[2:].replace("\\", "/")
        clean_str = f"/mnt/{drive}{rest}"
    return Path(clean_str)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare, clean, deduplicate, and tokenize dataset zip files for PyGPT."
    )
    parser.add_argument(
        "input_source",
        type=str,
        help="Path to dataset .zip file or dataset directory (wrap in quotes if path has spaces)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory to save tokenized dataset shards (default: data/processed)",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=8192,
        help="Target context length sequence packing window (default: 8192)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation dataset split ratio (default: 0.05)",
    )

    args = parser.parse_args()

    input_path = convert_path(args.input_source)
    if not input_path.exists():
        print(f"❌ Error: Input path '{input_path}' does not exist.")
        print("💡 Hint: Make sure to wrap paths containing spaces in double quotes, e.g.:")
        print('   python3 prepare_dataset.py "/mnt/c/Users/LOKESH KUMAR R/Downloads/python_first_ds.zip"')
        sys.exit(1)

    print(f"🚀 Starting PyGPT Dataset Preparation Pipeline...")
    print(f"Source: {input_path.resolve()}")
    print(f"Output Directory: {args.output_dir}")

    pipeline = DatasetPipeline(
        output_dir=args.output_dir,
        context_length=args.context_length,
        val_split_ratio=args.val_ratio,
    )

    metadata = pipeline.run_pipeline(str(input_path))

    print("\n🎉 Dataset Preparation Complete!")
    print(f"Train Tokens: {metadata['train_token_count']:,}")
    print(f"Validation Tokens: {metadata['val_token_count']:,}")
    print(f"Metadata File: {metadata['output_dir']}/dataset_metadata.json")


if __name__ == "__main__":
    main()
