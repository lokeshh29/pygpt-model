# 📊 Step 03: PyGPT Dataset Preparation Pipeline

This guide documents the end-to-end dataset collection, cleaning, deduplication, and tokenization pipeline for preparing training datasets from local `.zip` dataset archives.

---

## 🔄 Dataset Preparation Workflow

```mermaid
flowchart TD
    A[📦 Raw Dataset .zip Archive] --> B[🔓 Unzip & Extract Files]
    B --> C[🧹 Clean Control Characters]
    C --> D[⚙️ AST Syntax Filter (.py files)]
    D --> E[🔑 SHA-256 Deduplication]
    E --> F[🔤 PyGPTTokenizer Encoding]
    F --> G[📦 Token Sequence Packing]
    G --> H[💾 Save Train & Validation Shards]
```

---

## 🛠️ Pipeline Features

1. **Extraction & Scanning**:
   - Automatically unpacks dataset `.zip` archives containing `.py`, `.md`, `.txt`, and `.jsonl` files.
2. **AST Syntax Cleaning**:
   - Parses code using Python's `ast.parse()` module.
   - Rejects syntactically broken or truncated Python files to prevent training on garbage input.
3. **Exact Content Deduplication**:
   - Computes SHA-256 content hashes after whitespace normalization.
   - Eliminates duplicate code repositories and boilerplate files across dataset shards.
4. **Tokenization & Context Packing**:
   - Encodes cleaned documents into token IDs using `PyGPTTokenizer`.
   - Appends `<bos>` and `<eos>` special control tokens.
   - Splits data into `train_tokens.json` (95%) and `val_tokens.json` (5%).

---

## 💻 How to Process Your Local `.zip` Dataset

Run the dataset processing CLI script on your local zip file:

```bash
python prepare_dataset.py path/to/your_dataset.zip
```

### Options & Flags:
```bash
python prepare_dataset.py path/to/your_dataset.zip \
    --output_dir data/processed \
    --context_length 8192 \
    --val_ratio 0.05
```

### Output Files Produced:
- `data/processed/train_tokens.json`: Primary tokenized training dataset shard.
- `data/processed/val_tokens.json`: Tokenized validation dataset shard.
- `data/processed/dataset_metadata.json`: Dataset metadata summary (token counts, unique files processed, paths).

---

## 📋 Example Dataset Processing Log

```text
🚀 Starting PyGPT Dataset Preparation Pipeline...
Source: /path/to/python_dataset.zip
Output Directory: data/processed
📦 Extracting dataset: python_dataset.zip -> data/processed/raw_extracted
🔍 Scanning & cleaning documents...
✅ Processed 1,420 files | Valid: 1,385 | Duplicates Removed: 35
🔤 Tokenizing 1,385 clean documents...
📊 Total Tokens: 4,850,210 | Train Tokens: 4,607,699 | Val Tokens: 242,511
💾 Dataset saved to: data/processed

🎉 Dataset Preparation Complete!
```
