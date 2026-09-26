from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.inference.generator import PyGPTGenerator
from app.schemas.model import PyGPTModelConfig, ModelSize

router = APIRouter(prefix="/model", tags=["Code Generation & Inference"])

# Lazy-loaded global generator instance
_generator: Optional[PyGPTGenerator] = None


def get_generator() -> PyGPTGenerator:
    global _generator
    if _generator is None:
        config = PyGPTModelConfig(
            name="PyGPT-350M-Inference",
            size=ModelSize.NANO,
            total_parameters="350 Million",
        )
        config.architecture.hidden_size = 512
        config.architecture.num_hidden_layers = 6
        config.architecture.num_attention_heads = 8
        config.architecture.num_key_value_heads = 2
        config.architecture.intermediate_size = 2048
        _generator = PyGPTGenerator(config=config)
    return _generator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt or Python code prefix")
    max_new_tokens: int = Field(128, description="Maximum new tokens to generate")
    temperature: float = Field(0.5, description="Sampling temperature (0.0=deterministic, 1.0=creative)")
    top_k: int = Field(50, description="Top-K sampling cutoff")
    top_p: float = Field(0.9, description="Top-P nucleus sampling probability")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int
    generation_time_seconds: float


class InstructionRequest(BaseModel):
    instruction: str = Field(..., description="Task instruction (e.g. 'Write a function to...')")
    input_code: str = Field("", description="Optional input code snippet for explanation or bug fixing")
    task_type: str = Field("generate", description="Task type: 'generate', 'explain', or 'fix_bug'")
    max_new_tokens: int = Field(150, description="Maximum new tokens to generate")
    temperature: float = Field(0.5, description="Sampling temperature")


class InstructionResponse(BaseModel):
    instruction: str
    task_type: str
    generated_text: str
    tokens_generated: int
    generation_time_seconds: float


@router.post("/generate", response_model=GenerateResponse)
def generate_code(request: GenerateRequest):
    """Generates Python code completion from input prompt using PyGPT model."""
    generator = get_generator()
    res = generator.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
    return GenerateResponse(**res)


@router.post("/instruction", response_model=InstructionResponse)
def execute_instruction(request: InstructionRequest):
    """Executes coding instruction, program explanation, or bug fixing using PyGPT model."""
    generator = get_generator()
    res = generator.generate_instruction(
        instruction=request.instruction,
        input_code=request.input_code,
        task_type=request.task_type,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    return InstructionResponse(
        instruction=res["instruction"],
        task_type=res["task_type"],
        generated_text=str(res["generated_text"]),
        tokens_generated=int(res["tokens_generated"]),
        generation_time_seconds=float(res["generation_time_seconds"]),
    )
