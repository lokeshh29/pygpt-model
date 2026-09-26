# 🧠 PyGPT Model & Transformer Specification

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

---

## 🏗️ 2. PyTorch Transformer Module Implementation

The complete architecture is implemented in PyTorch under [`app/model/transformer.py`](app/model/transformer.py):

### 1. Token Embeddings & Output Projection
- **Embedding Layer** (`nn.Embedding`): Maps input token IDs $\in [0, V-1]$ into $d_{\text{model}}$-dimensional vectors.
- **Language Model Head** (`nn.Linear`): Unbiased linear projection layer mapping output states from $d_{\text{model}}$ back to vocabulary logits $V$.

### 2. Positional Encoding — Rotary Position Embedding (RoPE)
- **Module**: `RotaryEmbedding`
- Computes relative positional frequency matrices $\theta_i = \text{rope\_theta}^{-2i/d}$.
- Applies complex rotation $R_{\Theta, m}^d$ to Query ($Q$) and Key ($K$) tensors per head:
  $$\text{RoPE}(q, k, \text{cos}, \text{sin}) = (q \odot \text{cos}) + (\text{rotate\_half}(q) \odot \text{sin})$$

### 3. Attention — Grouped-Query Attention (GQA)
- **Module**: `GroupedQueryAttention`
- **Ratio**: 16 Query heads to 4 Key/Value heads ($4:1$ ratio).
- **KV Replication**: `repeat_kv` duplicates KV heads across query groups.
- **FlashAttention-2**: Automatically dispatches to `torch.nn.functional.scaled_dot_product_attention` (SDPA) for hardware-accelerated memory bandwidth utilization.

### 4. Feed-Forward Network — SwiGLU FFN
- **Module**: `SwiGLUFeedForward`
- **Equation**:
  $$\text{SwiGLU}(x) = \Big(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\Big) W_{\text{down}}$$
- **Dimensions**: $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

### 5. Normalization — RMSNorm
- **Module**: `RMSNorm`
- **Equation**:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
- Applied pre-normalization before Attention and FFN blocks.

---

## 📏 3. Context Length

- **Native Context Window**: **8,192 tokens** (8K context).
- **Extended Context Window**: Up to **32,768 tokens** (32K context) via **YaRN (Yet Another RoPE Extension)** and RoPE base frequency scaling ($\theta = 100,000$).

---

## 💻 PyTorch Usage Example

```python
import torch
from app.model.transformer import build_pygpt_model
from app.schemas.model import default_model_config

# Build PyGPT PyTorch Transformer Model
model = build_pygpt_model(default_model_config)

# Input token IDs (batch_size=1, seq_len=32)
input_ids = torch.randint(0, 32000, (1, 32))

# Forward pass -> Logits shape (1, 32, 32000)
logits = model(input_ids)
print("Logits shape:", logits.shape)
```
