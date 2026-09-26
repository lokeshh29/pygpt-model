from fastapi import APIRouter

from app.model.config import default_model_config
from app.schemas.model import PyGPTModelConfig

router = APIRouter(prefix="/model", tags=["Model Specifications"])


@router.get("/info", response_model=PyGPTModelConfig)
def get_model_info():
    """Returns the PyGPT model architecture, size, context length, and target capabilities."""
    return default_model_config
