"""
Legacy import compatibility wrapper.
Refactored implementation is located in app/schemas/model.py and app/model/config.py.
"""

from app.model.config import default_model_config, get_model_config
from app.schemas.model import (
    ContextConfig,
    ModelSize,
    PyGPTModelConfig,
    TargetCapabilitiesConfig,
    TransformerArchitectureConfig,
)

__all__ = [
    "ModelSize",
    "TransformerArchitectureConfig",
    "ContextConfig",
    "TargetCapabilitiesConfig",
    "PyGPTModelConfig",
    "default_model_config",
    "get_model_config",
]
