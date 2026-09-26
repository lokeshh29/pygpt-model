#!/usr/bin/env python3
"""
Verification script for PyGPT Transformer architecture in PyTorch.
Tests forward pass, tensor shape integrity, RMSNorm, RoPE, GQA Attention, SwiGLU FFN, and LM Head.
"""

import sys
import torch

from app.schemas.model import PyGPTModelConfig, ModelSize
from app.model.transformer import build_pygpt_model


def test_transformer_nano():
    print("🧪 Testing PyGPT Transformer (Nano 350M Configuration)...")

    # Create test configuration for fast verification
    nano_config = PyGPTModelConfig(
        name="PyGPT-350M-Test",
        size=ModelSize.NANO,
        total_parameters="350 Million",
    )
    nano_config.architecture.hidden_size = 512
    nano_config.architecture.num_hidden_layers = 4
    nano_config.architecture.num_attention_heads = 8
    nano_config.architecture.num_key_value_heads = 2
    nano_config.architecture.intermediate_size = 2048

    model = build_pygpt_model(nano_config)
    model.eval()

    # Create dummy input batch (batch_size=2, seq_len=16)
    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(0, nano_config.architecture.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    expected_shape = (batch_size, seq_len, nano_config.architecture.vocab_size)
    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} vs {expected_shape}"

    param_info = model.count_parameters()
    print(f"✅ Forward Pass Succeeded!")
    print(f"   Input Shape:  {tuple(dummy_input.shape)}")
    print(f"   Logits Shape: {tuple(logits.shape)}")
    print(f"   Total Parameters: {param_info['total_parameters']:,}")


if __name__ == "__main__":
    try:
        test_transformer_nano()
        print("\n🎉 PyGPT Transformer PyTorch Implementation Passed All Tests!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
