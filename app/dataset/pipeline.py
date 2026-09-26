import ast
import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.tokenizer.tokenizer import pygpt_tokenizer


class DatasetPipeline:
    """
    Complete dataset preparation pipeline for PyGPT pre-training data:
    1. Extract zip files of Python & text datasets
    2. Support .py, .csv, .jsonl, .txt, and .md dataset formats
    3. Clean text, filter out non-printable chars & auto-generated code
    4. Validate Python syntax using AST parsing where applicable
    5. Deduplicate using content SHA-256 hashing
    6. Tokenize using PyGPTTokenizer and pack tokens into dataset shards
    """

    CODE_COLUMN_NAMES = [
        "code", "python_code", "solution", "script", "python",
        "output", "code_snippet", "canonical_solution", "target"
    ]
    PROMPT_COLUMN_NAMES = [
        "question", "prompt", "instruction", "input", "description",
        "problem", "task", "title"
    ]

    def __init__(
        self,
        output_dir: str = "data/processed",
        context_length: int = 8192,
        val_split_ratio: float = 0.05,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context_length = context_length
        self.val_split_ratio = val_split_ratio
        self.seen_hashes: Set[str] = set()

    def extract_zip(self, zip_path: str, extract_to: Optional[str] = None) -> Path:
        """Extracts a zip dataset archive to target directory."""
        zip_file = Path(zip_path)
        if not zip_file.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        target_dir = Path(extract_to) if extract_to else self.output_dir / "raw_extracted"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"📦 Extracting dataset archive: {zip_file.name} -> {target_dir}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(target_dir)

        return target_dir

    def is_valid_python(self, code: str) -> bool:
        """Checks whether code string is syntactically valid Python code."""
        if not code or len(code.strip()) < 5:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            # If code is a single expression or function snippet, still accept if non-empty
            return len(code.strip()) > 10

    def is_duplicate(self, text: str) -> bool:
        """Deduplicates content using SHA-256 hash matching."""
        normalized = " ".join(text.split())
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False

    def clean_text(self, text: str) -> str:
        """Removes non-printable control characters while preserving indentations and line breaks."""
        return "".join(c for c in text if c.isprintable() or c in "\n\t\r")

    def process_csv_file(self, file_path: Path) -> List[str]:
        """Parses CSV files containing Python questions and code snippets."""
        extracted_docs: List[str] = []
        # Increase field size limit for large CSV entries
        csv.field_size_limit(sys.maxsize)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []

            # Map header column names (case-insensitive)
            header_lower = [h.strip().lower() for h in header]
            code_col_idx = next(
                (i for i, h in enumerate(header_lower) if h in self.CODE_COLUMN_NAMES),
                None,
            )
            prompt_col_idx = next(
                (i for i, h in enumerate(header_lower) if h in self.PROMPT_COLUMN_NAMES),
                None,
            )

            # Fallback if column names differ: if 2 columns, assume col 0 = prompt, col 1 = code
            if code_col_idx is None and len(header) >= 2:
                code_col_idx = 1
                prompt_col_idx = 0
            elif code_col_idx is None and len(header) == 1:
                code_col_idx = 0

            for row in reader:
                if not row or code_col_idx >= len(row):
                    continue

                code_content = row[code_col_idx].strip()
                prompt_content = (
                    row[prompt_col_idx].strip()
                    if prompt_col_idx is not None and prompt_col_idx < len(row)
                    else ""
                )

                if not code_content:
                    continue

                # Format prompt and code together
                if prompt_content:
                    doc = f'"""\n{prompt_content}\n"""\n{code_content}'
                else:
                    doc = code_content

                cleaned_doc = self.clean_text(doc)
                if not self.is_duplicate(cleaned_doc):
                    extracted_docs.append(cleaned_doc)

        return extracted_docs

    def process_directory(self, input_dir: str) -> List[str]:
        """
        Recursively scans input directory for Python (.py), CSV (.csv), JSONL (.jsonl),
        and Markdown/Text (.md, .txt) files, cleans, syntax checks, and deduplicates them.
        """
        input_path = Path(input_dir)
        clean_documents: List[str] = []
        scanned_files = 0

        print(f"🔍 Scanning & extracting documents in: {input_path}")

        for file_path in input_path.rglob("*"):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()

            # Handle CSV datasets
            if suffix == ".csv":
                scanned_files += 1
                try:
                    docs = self.process_csv_file(file_path)
                    clean_documents.extend(docs)
                    print(f"  📄 Processed CSV: {file_path.name} -> Extracted {len(docs):,} code snippets")
                except Exception as e:
                    print(f"  ⚠️ Warning processing CSV {file_path.name}: {e}")

            # Handle standard Python (.py) files
            elif suffix == ".py":
                scanned_files += 1
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = self.clean_text(f.read())

                    if self.is_valid_python(content) and not self.is_duplicate(content):
                        clean_documents.append(content)
                except Exception:
                    continue

            # Handle Text / Markdown / JSONL files
            elif suffix in [".txt", ".md", ".jsonl"]:
                scanned_files += 1
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = self.clean_text(f.read())

                    if content and not self.is_duplicate(content):
                        clean_documents.append(content)
                except Exception:
                    continue

        print(
            f"✅ Scanned {scanned_files} dataset file(s) | "
            f"Extracted & Deduplicated Documents: {len(clean_documents):,}"
        )
        return clean_documents

    def tokenize_and_pack(self, documents: List[str]) -> Tuple[List[int], List[int]]:
        """
        Tokenizes clean documents using PyGPTTokenizer and splits into train & validation token sequences.
        """
        all_tokens: List[int] = []

        print(f"🔤 Tokenizing {len(documents):,} extracted documents using PyGPTTokenizer...")
        for idx, doc in enumerate(documents):
            doc_tokens = pygpt_tokenizer.encode(doc, add_special_tokens=True)
            all_tokens.extend(doc_tokens)
            if (idx + 1) % 10000 == 0 or (idx + 1) == len(documents):
                print(f"  ⚡ Progress: {idx + 1:,} / {len(documents):,} documents tokenized...")

        total_tokens = len(all_tokens)
        val_size = int(total_tokens * self.val_split_ratio)
        train_tokens = all_tokens[:-val_size] if val_size > 0 else all_tokens
        val_tokens = all_tokens[-val_size:] if val_size > 0 else []

        print(f"📊 Total Tokens: {total_tokens:,} | Train Tokens: {len(train_tokens):,} | Val Tokens: {len(val_tokens):,}")
        return train_tokens, val_tokens

    def save_dataset_shards(self, train_tokens: List[int], val_tokens: List[int]) -> Dict[str, str]:
        """Saves tokenized dataset shards as binary JSON files."""
        train_file = self.output_dir / "train_tokens.json"
        val_file = self.output_dir / "val_tokens.json"

        with open(train_file, "w", encoding="utf-8") as f:
            json.dump({"token_count": len(train_tokens), "tokens": train_tokens}, f)

        with open(val_file, "w", encoding="utf-8") as f:
            json.dump({"token_count": len(val_tokens), "tokens": val_tokens}, f)

        metadata = {
            "output_dir": str(self.output_dir),
            "train_tokens_file": str(train_file),
            "val_tokens_file": str(val_file),
            "train_token_count": len(train_tokens),
            "val_token_count": len(val_tokens),
            "unique_documents_seen": len(self.seen_hashes),
        }

        meta_file = self.output_dir / "dataset_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Tokenized dataset shards saved to: {self.output_dir}")
        return metadata

    def run_pipeline(self, input_source: str) -> Dict[str, str]:
        """Runs full end-to-end dataset preparation pipeline on zip file or directory."""
        input_path = Path(input_source)

        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            extracted_dir = self.extract_zip(str(input_path))
            docs = self.process_directory(str(extracted_dir))
        elif input_path.is_dir():
            docs = self.process_directory(str(input_path))
        else:
            raise ValueError(f"Invalid input source path: {input_source}")

        train_tokens, val_tokens = self.tokenize_and_pack(docs)
        return self.save_dataset_shards(train_tokens, val_tokens)


if __name__ == "__main__":
    pipeline = DatasetPipeline()
    print("DatasetPipeline initialized and ready for training data processing.")
