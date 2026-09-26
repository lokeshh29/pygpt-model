from app.schemas.model import PyGPTModelConfig

# Default global instance of the PyGPT model configuration
default_model_config = PyGPTModelConfig()


def get_model_config() -> PyGPTModelConfig:
    """Returns the current PyGPT model configuration."""
    return default_model_config
