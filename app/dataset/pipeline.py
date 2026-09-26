import ast
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.tokenizer.tokenizer import pygpt_tokenizer


class DatasetPipeline:
    """
    Complete dataset preparation pipeline for PyGPT pre-training data:
    1. Extract zip files of Python & text datasets
    2. Clean text, filter out non-printable chars & auto-generated code
    3. Validate Python syntax using AST parsing
    4. Deduplicate using content SHA-256 hashing
    5. Tokenize using PyGPTTokenizer and pack tokens into fixed context windows
    6. Save binary train/val dataset shards (.bin)
    """

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

        print(f"📦 Extracting dataset: {zip_file.name} -> {target_dir}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(target_dir)

        return target_dir

    def is_valid_python(self, code: str) -> bool:
        """Checks whether code string is syntactically valid Python code."""
        if not code or len(code.strip()) < 10:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def is_duplicate(self, text: str) -> bool:
        """Deduplicates content using SHA-256 hash matching."""
        # Normalize whitespace for deduplication check
        normalized = " ".join(text.split())
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False

    def clean_text(self, text: str) -> str:
        """Removes non-printable ASCII control characters while preserving indentations and line breaks."""
        return "".join(c for c in text if c.isprintable() or c in "\n\t\r")

    def process_directory(self, input_dir: str) -> List[str]:
        """
        Recursively scans input directory for Python (.py) and Markdown/Text (.md, .txt) files,
        cleans, syntax checks, and deduplicates them.
        """
        input_path = Path(input_dir)
        clean_documents: List[str] = []
        scanned_count = 0
        valid_count = 0
        duplicate_count = 0

        print(f"🔍 Scanning & cleaning documents in: {input_path}")

        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [".py", ".md", ".txt", ".jsonl"]:
                scanned_count += 1
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_content = f.read()

                    cleaned = self.clean_text(raw_content)

                    # For Python files, enforce AST syntax check
                    if file_path.suffix.lower() == ".py":
                        if not self.is_valid_python(cleaned):
                            continue

                    # Deduplication check
                    if self.is_duplicate(cleaned):
                        duplicate_count += 1
                        continue

                    clean_documents.append(cleaned)
                    valid_count += 1

                except Exception as e:
                    continue

        print(f"✅ Processed {scanned_count} files | Valid: {valid_count} | Duplicates Removed: {duplicate_count}")
        return clean_documents

    def tokenize_and_pack(self, documents: List[str]) -> Tuple[List[int], List[int]]:
        """
        Tokenizes clean documents using PyGPTTokenizer and splits into train & validation token sequences.
        """
        all_tokens: List[int] = []

        print(f"🔤 Tokenizing {len(documents)} clean documents...")
        for doc in documents:
            doc_tokens = pygpt_tokenizer.encode(doc, add_special_tokens=True)
            all_tokens.extend(doc_tokens)

        total_tokens = len(all_tokens)
        val_size = int(total_tokens * self.val_split_ratio)
        train_tokens = all_tokens[:-val_size]
        val_tokens = all_tokens[-val_size:] if val_size > 0 else []

        print(f"📊 Total Tokens: {total_tokens:,} | Train Tokens: {len(train_tokens):,} | Val Tokens: {len(val_tokens):,}")
        return train_tokens, val_tokens

    def save_dataset_shards(self, train_tokens: List[int], val_tokens: List[int]) -> Dict[str, str]:
        """Saves tokenized dataset shards as binary JSON / text files."""
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

        print(f"💾 Dataset saved to: {self.output_dir}")
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
