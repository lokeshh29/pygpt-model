import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from app.model.config import default_model_config
from app.model.transformer import PyGPTTransformer, build_pygpt_model
from app.schemas.model import PyGPTModelConfig, ModelSize
from app.tokenizer.tokenizer import pygpt_tokenizer


class PyGPTGenerator:
    """
    Inference & Text Generation Engine for PyGPT:
    - Autoregressive next-token sampling with Temperature, Top-K, and Top-P (Nucleus) sampling
    - Formats instruction prompts for code generation, program explanation, and bug fixing
    - Manages model checkpoint loading from PyTorch weights (.pt)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: PyGPTModelConfig = default_model_config,
        device: Optional[str] = None,
    ):
        self.config = config

        # Auto-detect device
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        # Build PyGPT Transformer Model
        self.model = build_pygpt_model(self.config).to(self.device)
        self.model.eval()

        # Load checkpoint weights if specified or available
        if checkpoint_path is None:
            best_ckpt = Path("checkpoints/instruction_model.pt")
            if not best_ckpt.exists():
                best_ckpt = Path("checkpoints/best_model.pt")
            if best_ckpt.exists():
                checkpoint_path = str(best_ckpt)

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_weights(checkpoint_path)

    def load_weights(self, checkpoint_path: str) -> None:
        """Loads PyTorch model weights from checkpoint file."""
        print(f"📦 Loading weights from checkpoint: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded successfully!")

    def sample_top_k_top_p(
        self, logits: torch.Tensor, temperature: float = 0.5, top_k: int = 50, top_p: float = 0.9
    ) -> torch.Tensor:
        """Applies Temperature scaling, Top-K filtering, and Top-P (Nucleus) sampling."""
        if temperature > 0:
            logits = logits / temperature
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        # Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.5,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Dict[str, Union[str, int, float]]:
        """Generates Python code completion from input prompt."""
        start_time = time.time()
        input_ids = pygpt_tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = list(input_ids)
        eos_id = pygpt_tokenizer.token_to_id.get(pygpt_tokenizer.EOS_TOKEN, 3)

        for _ in range(max_new_tokens):
            current_input = input_tensor[:, -self.config.context.native_context_length :]
            logits = self.model(current_input)
            next_token_logits = logits[:, -1, :]

            next_token = self.sample_top_k_top_p(
                next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            token_id = next_token.item()

            if token_id == eos_id:
                break

            generated_ids.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

        generation_time = time.time() - start_time
        full_text = pygpt_tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract only response portion if instruction markers exist
        if "### Response:\n" in full_text:
            response_text = full_text.split("### Response:\n")[-1].strip()
        else:
            response_text = full_text

        return {
            "prompt": prompt,
            "generated_text": response_text,
            "full_output": full_text,
            "tokens_generated": len(generated_ids) - len(input_ids),
            "generation_time_seconds": round(generation_time, 3),
        }

    def format_instruction_prompt(
        self, instruction: str, input_code: str = "", task_type: str = "generate"
    ) -> str:
        """Formats prompt into standard instruction-following schema."""
        prompt = f"### Instruction:\n{instruction}\n"
        if input_code:
            prompt += f"\n### Input Code:\n{input_code}\n"
        prompt += "\n### Response:\n"
        return prompt

    def generate_instruction(
        self,
        instruction: str,
        input_code: str = "",
        task_type: str = "generate",
        max_new_tokens: int = 150,
        temperature: float = 0.5,
    ) -> Dict[str, Union[str, int, float]]:
        """Executes instruction-following code generation, explanation, or bug fixing."""
        formatted_prompt = self.format_instruction_prompt(instruction, input_code, task_type)
        res = self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        res["instruction"] = instruction
        res["task_type"] = task_type
        return res
