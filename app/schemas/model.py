from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class ModelSize(str, Enum):
    NANO = "350M"
    BASE = "1.3B"
    PRO = "7B"


class TransformerArchitectureConfig(BaseModel):
    architecture_type: str = Field(
        "Decoder-Only Transformer (Modern Stack)",
        description="Transformer architecture paradigm",
    )
    vocab_size: int = Field(
        32000,
        description="Python-optimized Byte-Pair Encoding (BPE) vocabulary size",
    )
    hidden_size: int = Field(
        2048,
        description="Hidden dimension size (d_model)",
    )
    num_hidden_layers: int = Field(
        24,
        description="Number of decoder transformer layers",
    )
    num_attention_heads: int = Field(
        16,
        description="Number of query attention heads",
    )
    num_key_value_heads: int = Field(
        4,
        description="Number of key/value heads for Grouped-Query Attention (GQA)",
    )
    intermediate_size: int = Field(
        8192,
        description="SwiGLU feed-forward network intermediate dimension (d_ff)",
    )
    activation_function: str = Field(
        "SwiGLU",
        description="Activation function type",
    )
    norm_type: str = Field(
        "RMSNorm",
        description="Root Mean Square normalization with pre-norm placement",
    )
    positional_embedding: str = Field(
        "RoPE (Rotary Position Embedding)",
        description="Positional encoding strategy",
    )
    use_bias: bool = Field(
        False,
        description="Bias-free linear layers for training stability & throughput",
    )


class ContextConfig(BaseModel):
    native_context_length: int = Field(
        8192,
        description="Native context window size in tokens (8K)",
    )
    extended_context_length: int = Field(
        32768,
        description="Extended context window size via RoPE frequency scaling / YaRN (32K)",
    )
    rope_theta: float = Field(
        100000.0,
        description="RoPE base frequency theta parameter",
    )


class TargetCapabilitiesConfig(BaseModel):
    primary_language: str = "Python 3.10+"
    capabilities: List[str] = Field(
        default_factory=lambda: [
            "Zero-shot and few-shot Python code completion & snippet generation",
            "Automated syntax & logic debugging with stack trace diagnostics",
            "Docstring generation and strict typing annotations (mypy-compliant)",
            "Automated pytest unit test case generation",
            "Deep comprehension of Python ecosystem (FastAPI, PyTorch, Pandas, NumPy, Pydantic, uv)",
            "PEP 8 refactoring, AST inspection, and runtime performance optimization",
        ]
    )


class PyGPTModelConfig(BaseModel):
    name: str = "PyGPT-1.3B-Code"
    version: str = "1.0.0"
    size: ModelSize = ModelSize.BASE
    total_parameters: str = "1.3 Billion"
    architecture: TransformerArchitectureConfig = TransformerArchitectureConfig()
    context: ContextConfig = ContextConfig()
    target_capabilities: TargetCapabilitiesConfig = TargetCapabilitiesConfig()
