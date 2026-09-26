import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.schemas.model import PyGPTModelConfig, default_model_config


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Pre-norm placement for improved numerical stability & training throughput.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for relative position encodings.
    Frequency theta base set to 100,000 for context window scaling up to 32k.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 100000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions for RoPE vector multiplication."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to Query and Key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats Key/Value heads to match Query head count in Grouped-Query Attention (GQA)."""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) Module.
    16 Query heads, 4 Key/Value heads (4:1 GQA ratio) for reduced KV cache memory footprint.
    """

    def __init__(self, config: PyGPTModelConfig):
        super().__init__()
        self.hidden_size = config.architecture.hidden_size
        self.num_heads = config.architecture.num_attention_heads
        self.num_kv_heads = config.architecture.num_key_value_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Linear projections without bias for improved stability
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Linear projections & reshape to (batch, heads, seq_len, head_dim)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE positional encodings to Query & Key
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache for fast inference generation
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        # Repeat KV heads for GQA
        k = repeat_kv(k, self.num_queries_per_kv)
        v = repeat_kv(v, self.num_queries_per_kv)

        # Compute Scaled Dot-Product Attention (FlashAttention-2 / SDPA)
        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=(mask is None and seq_len > 1)
            )
        else:
            # Manual fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)

        # Transpose back and project output
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(output)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.
    Formula: SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class PyGPTDecoderBlock(nn.Module):
    """
    Decoder Transformer Block for PyGPT.
    Structure: Pre-RMSNorm Attention -> Residual Add -> Pre-RMSNorm SwiGLU FFN -> Residual Add
    """

    def __init__(self, config: PyGPTModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.architecture.hidden_size)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.architecture.hidden_size)
        self.ffn = SwiGLUFeedForward(
            config.architecture.hidden_size,
            config.architecture.intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Pre-norm Self Attention + Residual connection
        h = x + self.attn(self.attn_norm(x), cos, sin, mask=mask, kv_cache=kv_cache)
        # Pre-norm SwiGLU FFN + Residual connection
        out = h + self.ffn(self.ffn_norm(h))
        return out


class PyGPTTransformer(nn.Module):
    """
    Full PyGPT Decoder-Only Transformer Architecture.
    Includes Embeddings, RoPE, Grouped-Query Attention, SwiGLU FFN, RMSNorm, and Output Projection Head.
    """

    def __init__(self, config: PyGPTModelConfig = default_model_config):
        super().__init__()
        self.config = config
        self.vocab_size = config.architecture.vocab_size
        self.hidden_size = config.architecture.hidden_size

        # 1. Token Embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)

        # 2. Rotary Positional Embeddings
        self.rope = RotaryEmbedding(
            dim=self.hidden_size // config.architecture.num_attention_heads,
            max_position_embeddings=config.context.extended_context_length,
            base=config.context.rope_theta,
        )

        # 3. Stack of Decoder Blocks
        self.layers = nn.ModuleList(
            [PyGPTDecoderBlock(config) for _ in range(config.architecture.num_hidden_layers)]
        )

        # 4. Final Normalization
        self.norm = RMSNorm(self.hidden_size)

        # 5. Output LM Head Projection
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for PyGPT Transformer model.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            mask: Optional attention mask tensor
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        bsz, seq_len = input_ids.shape
        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope(x, seq_len)

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Calculates trainable and total parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


# Global helper instantiation
def build_pygpt_model(config: PyGPTModelConfig = default_model_config) -> PyGPTTransformer:
    """Factory function to build PyGPTTransformer instance."""
    return PyGPTTransformer(config)
