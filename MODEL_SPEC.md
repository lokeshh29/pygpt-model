# 🧠 PyGPT Model Specification

## 01. PyGPT Model Definition

The **PyGPT Model** is a specialized, open-weights decoder-only transformer model optimized specifically for Python code generation, automated debugging, type annotation, refactoring, and stack trace analysis.

---

## 📊 1. Model Size Variants

PyGPT is defined with three model parameter tiers tailored for different execution targets (edge/local development vs. server GPU deployment):

| Variant | Total Parameters | Active Parameters | Hidden Dim ($d_{\text{model}}$) | Layers ($n_{\text{layer}}$) | Attention Heads ($n_{\text{head}}$) | KV Heads ($n_{\text{kv}}$) | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyGPT-Nano** | **350M** | 350M | 1,024 | 16 | 8 | 2 | CPU / Local Dev / Mobile |
| **PyGPT-Base** *(Default)* | **1.3B** | 1.3B | 2,048 | 24 | 16 | 4 | Single GPU / Dev Server |
| **PyGPT-Pro** | **7B** | 7B | 4,096 | 32 | 32 | 8 | Enterprise GPU Cluster |

### Default Model (PyGPT-Base 1.3B):
- **Total Parameter Count**: ~1.3 Billion parameters
- **Precision Support**: FP16, BF16, INT8, and INT4 (via AWQ / GPTQ quantization)

---

## 🏗️ 2. Transformer Architecture

PyGPT incorporates state-of-the-art modern decoder-only transformer design choices to optimize token throughput and memory efficiency during Python code inference:

### Architecture Stack Highlights:
1. **Decoder-Only Transformer Structure**:
   - Autoregressive causal language modeling.
   - Pre-Layer Normalization placement for training stability.

2. **Attention Mechanism — Grouped-Query Attention (GQA)**:
   - **16 Query Heads** and **4 Key/Value Heads** (4:1 GQA ratio).
   - Drastically reduces Key-Value (KV) cache memory footprint during long context generation, enabling higher batch sizes and faster inference.
   - Fully compatible with **FlashAttention-2** and **SDPA (Scaled Dot-Product Attention)**.

3. **Positional Encoding — Rotary Position Embedding (RoPE)**:
   - Eliminates absolute positional embeddings in favor of relative rotational encodings.
   - Frequency scaling factor ($\theta = 100,000$) configured for context extension without degradation.

4. **Feed-Forward Network (FFN) — SwiGLU Activation**:
   - Replaces standard GELU/ReLU with **SwiGLU (Swish Gated Linear Unit)**.
   - Intermediate dimension size ($d_{\text{ff}}$): $8,192$ (4x hidden size multiplier adjusted for SwiGLU gating).

5. **Normalization — RMSNorm**:
   - Uses **Root Mean Square Normalization (RMSNorm)** over standard LayerNorm for $\sim 10-50\%$ faster normalization kernel execution.
   - $\epsilon = 1\times 10^{-6}$.

6. **Bias-Free Layers**:
   - All linear projection layers (Attention Query/Key/Value/Output and FFN Gate/Up/Down projections) omit additive bias parameters to improve numerical stability and compute efficiency.

7. **Tokenizer**:
   - Custom **32,000 token vocabulary** Byte-Pair Encoding (BPE) tokenizer trained on Python repositories (preserving whitespace indentations, Python keywords, AST nodes, and common variable patterns).

---

## 📏 3. Context Length

PyGPT is engineered to process large Python codebases, multi-file dependencies, and long execution stack traces:

- **Native Context Window**: **8,192 tokens** (8K context).
- **Extended Context Window**: Up to **32,768 tokens** (32K context) via **YaRN (Yet Another RoPE Extension)** and RoPE base frequency scaling ($\theta = 100,000$).

### Context Allocation Breakdown:
- **System Prompt & Rules**: Up to 1,024 tokens.
- **Repository Context & Imports**: Up to 16,384 tokens.
- **Active File / Query Input**: Up to 8,192 tokens.
- **Generated Code Response**: Up to 7,168 tokens.

---

## 🎯 4. Target Capabilities

PyGPT is specialized for Python-centric development workflows:

1. **Python Code Generation & Completion**:
   - Zero-shot and few-shot generation of functions, classes, modules, and boilerplate.
   - Real-time line and block auto-completion.

2. **Automated Debugging & Stack Trace Diagnostics**:
   - Parsing Python exceptions (`SyntaxError`, `TypeError`, `KeyError`, `AttributeError`, `RecursionError`).
   - Root-cause analysis from raw execution logs and traceback output.

3. **Strict Type Annotations & Static Analysis**:
   - Adding precise `typing` annotations (`TypeVar`, `Union`, `Optional`, `Callable`, `ParamSpec`) matching `mypy` and `pyright` strict compliance.

4. **Docstring & Specification Authoring**:
   - Generating standard Google, NumPy, or Sphinx formatted docstrings.

5. **Unit Test Generation**:
   - Generating comprehensive `pytest` test suites, fixtures, parametrizations, and mock objects.

6. **Deep Ecosystem Mastery**:
   - Built-in proficiency with top Python packages and frameworks:
     - **Web**: FastAPI, Uvicorn, Django, Flask, Pydantic.
     - **AI & Math**: PyTorch, NumPy, Pandas, SciPy, Scikit-learn.
     - **Tooling & Package Managers**: `uv`, `poetry`, `pipenv`, `ruff`, `mypy`.
     - **Database & Async**: `asyncio`, `SQLAlchemy`, `Tortoise-ORM`, `aiohttp`.

7. **Refactoring & Code Optimization**:
   - PEP 8 formatting alignment.
   - Vectorization of loops using NumPy/Pandas.
   - Converting sync blocking code into async `asyncio` routines.
