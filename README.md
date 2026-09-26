# PyGPT Model

Custom open-weights decoder-only transformer model tailored specifically for Python code completion, automated debugging, type annotation, refactoring, and stack trace analysis.

---

## 📁 Repository Structure

```
pygpt-model/
├── app/
│   ├── dataset/               # 📊 Dataset preparation & cleaning pipeline
│   │   └── pipeline.py        # Zip extraction, AST syntax filter, SHA-256 deduplication
│   ├── schemas/               # 📋 Pydantic request/response schemas
│   │   ├── model.py           # PyGPT model config & architecture schemas
│   │   └── tokenizer.py       # Tokenizer request & response schemas
│   ├── model/                 # 🧠 Model specifications & configuration logic
│   │   └── config.py          # PyGPT model configuration loader
│   ├── tokenizer/             # 🔤 Tokenizer implementation & vocabulary
│   │   ├── tokenizer.py       # PyGPTTokenizer BPE & Python syntax lexer
│   │   └── vocab.json         # Serialized vocabulary (32,000 tokens)
│   └── routers/               # 🚦 FastAPI route handlers
│       ├── model.py           # /model/* endpoints
│       └── tokenizer.py       # /tokenizer/* endpoints
├── main.py                    # 🚀 FastAPI server entry point & router registration
├── prepare_dataset.py         # 📊 CLI script to prepare & tokenize local dataset .zip files
├── MODEL_SPEC.md              # 📖 Detailed architecture & context specification
├── TOKENIZER_SPEC.md          # 📖 Tokenizer vocabulary & features specification
├── DATASET_PREPARATION.md     # 📖 Dataset preparation & cleaning pipeline guide
├── curl_commands.md           # 🛠️ cURL request guide for all endpoints
└── pyproject.toml             # 📦 Project dependencies & configuration
```

---

## 📊 Dataset Preparation Pipeline

To prepare training data from a local `.zip` file of Python code datasets:

```bash
python prepare_dataset.py path/to/dataset.zip
```
For detailed dataset cleaning and deduplication specifications, refer to [`DATASET_PREPARATION.md`](DATASET_PREPARATION.md).

---

## 🚀 Running the Server

### Using `uv`

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Run the Development Server**:
   ```bash
   uv run uvicorn main:app --reload
   ```

---

## 📌 Main Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check endpoint
- `GET /model/info`: Returns model parameters, transformer architecture, context length & capabilities
- `POST /tokenizer/encode`: Encodes Python code text into token IDs
- `POST /tokenizer/decode`: Decodes token IDs back into source code text
- `GET /docs`: Interactive Swagger API documentation

For comprehensive cURL commands, refer to [`curl_commands.md`](curl_commands.md).
