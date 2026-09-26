# PyGPT Model

Custom open-weights decoder-only transformer model tailored specifically for Python code completion, automated debugging, type annotation, refactoring, and stack trace analysis.

---

## 📁 Repository Structure

```
pygpt-model/
├── app/
│   ├── dataset/               # 📊 Dataset preparation & DataLoader modules
│   │   ├── pipeline.py        # Zip extraction, AST syntax filter, SHA-256 deduplication
│   │   └── loader.py          # PyTorch Token Dataset & DataLoaders for next-token prediction
│   ├── training/              # 🏋️ Trainer engine & loss monitoring
│   │   └── trainer.py         # Autoregressive Cross-Entropy, AdamW, Cosine LR & val loss monitoring
│   ├── schemas/               # 📋 Pydantic request/response schemas
│   │   ├── model.py           # PyGPT model config & architecture schemas
│   │   └── tokenizer.py       # Tokenizer request & response schemas
│   ├── model/                 # 🧠 PyTorch Transformer Architecture
│   │   ├── transformer.py     # Embeddings, RoPE, GQA Attention, SwiGLU FFN, RMSNorm, LM Head
│   │   └── config.py          # PyGPT model configuration loader
│   ├── tokenizer/             # 🔤 Tokenizer implementation & vocabulary
│   │   ├── tokenizer.py       # PyGPTTokenizer BPE & Python syntax lexer
│   │   └── vocab.json         # Serialized vocabulary (32,000 tokens)
│   └── routers/               # 🚦 FastAPI route handlers
│       ├── model.py           # /model/* endpoints
│       └── tokenizer.py       # /tokenizer/* endpoints
├── main.py                    # 🚀 FastAPI server entry point & router registration
├── prepare_dataset.py         # 📊 CLI script to prepare & tokenize local dataset .zip files
├── train_pygpt.py             # 🏋️ CLI script to pretrain PyGPT model with val loss monitoring
├── test_transformer.py        # 🧪 Verification test suite for PyTorch Transformer
├── MODEL_SPEC.md              # 📖 Detailed architecture & context specification
├── TOKENIZER_SPEC.md          # 📖 Tokenizer vocabulary & features specification
├── DATASET_PREPARATION.md     # 📖 Dataset preparation & cleaning pipeline guide
├── PRETRAINING_GUIDE.md        # 📖 Model pretraining & validation loss guide
├── curl_commands.md           # 🛠️ cURL request guide for all endpoints
└── pyproject.toml             # 📦 Project dependencies & configuration
```

---

## 🏋️ Model Pre-training

To pretrain the PyGPT model using Next-Token Prediction on your prepared dataset shards:

```bash
python3 train_pygpt.py
```

For detailed pre-training equations and monitoring guidelines, refer to [`PRETRAINING_GUIDE.md`](PRETRAINING_GUIDE.md).

---

## 🚀 Running the API Server

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
