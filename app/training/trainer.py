import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app.dataset.loader import create_dataloaders
from app.model.config import default_model_config
from app.model.transformer import PyGPTTransformer, build_pygpt_model
from app.schemas.model import PyGPTModelConfig


class PyGPTTrainer:
    """
    Pre-training engine for PyGPT Model using Next-Token Prediction:
    - Autoregressive Cross-Entropy Loss computation
    - AdamW optimizer with weight decay and Cosine Annealing scheduler
    - Validation loss monitoring & perplexity (PPL) calculation
    - Automatic Checkpoint Saving & Resume Capabilities
    """

    def __init__(
        self,
        config: PyGPTModelConfig = default_model_config,
        checkpoint_dir: str = "checkpoints",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Auto-detect computation device
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        print(f"🖥️ Initializing PyGPT Trainer on Device: {self.device}")

        # Build PyGPT Transformer Model
        self.model = build_pygpt_model(self.config).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        self.start_epoch = 1
        self.start_step = 0
        self.best_val_loss = float("inf")
        self.history: List[Dict[str, float]] = []

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Next-Token Prediction Cross-Entropy Loss."""
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        return F.cross_entropy(logits_flat, targets_flat)

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Runs validation evaluation loop and returns (val_loss, val_perplexity)."""
        self.model.eval()
        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.compute_loss(logits, y)
                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        perplexity = math.exp(min(avg_val_loss, 20))
        return avg_val_loss, perplexity

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        val_loss: float,
        is_best: bool = False,
    ) -> str:
        """Saves model state_dict, optimizer state, and progress metrics."""
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.model_dump(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }

        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint_data, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            print(f"🌟 Best model checkpoint saved: {best_path.name} (Val Loss: {val_loss:.4f})")
            return str(best_path)

        return str(latest_path)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Loads model weights, optimizer state, and step history to resume training."""
        if checkpoint_path is None:
            checkpoint_path = str(self.checkpoint_dir / "latest_model.pt")

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            print(f"⚠️ No checkpoint found at {checkpoint_path}. Starting fresh training.")
            return False

        print(f"🔄 Resuming training from checkpoint: {ckpt_file.name}...")
        checkpoint_data = torch.load(ckpt_file, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.start_epoch = checkpoint_data.get("epoch", 1)
        self.start_step = checkpoint_data.get("step", 0)
        self.best_val_loss = checkpoint_data.get("best_val_loss", checkpoint_data.get("val_loss", float("inf")))

        print(f"✅ Checkpoint Loaded! Resuming at Epoch {self.start_epoch}, Step {self.start_step} (Best Val Loss: {self.best_val_loss:.4f})")
        return True

    def train(
        self,
        train_file: str = "data/processed/train_tokens.json",
        val_file: str = "data/processed/val_tokens.json",
        epochs: int = 3,
        batch_size: int = 4,
        seq_len: int = 512,
        eval_interval: int = 20,
        resume: bool = False,
    ) -> Dict[str, str]:
        """Runs the main pre-training loop over epochs with optional resume support."""
        if resume:
            self.load_checkpoint()

        print(f"🚀 PyGPT Pre-training Running...")
        print(f"   Epochs: {epochs} | Batch Size: {batch_size} | Context Seq Len: {seq_len}")

        train_loader, val_loader = create_dataloaders(
            train_file=train_file,
            val_file=val_file,
            seq_len=seq_len,
            batch_size=batch_size,
        )

        total_steps = epochs * len(train_loader)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps), eta_min=1e-5)
        global_step = self.start_step
        start_time = time.time()

        for epoch in range(self.start_epoch, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            step_in_epoch = 0

            for x, y in train_loader:
                global_step += 1
                step_in_epoch += 1

                # Skip steps if resuming mid-epoch
                if global_step <= self.start_step:
                    continue

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.compute_loss(logits, y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

                # Real-time step progress logging
                if step_in_epoch % 10 == 0 or step_in_epoch == 1:
                    print(
                        f"  ⚡ Step [{step_in_epoch}/{len(train_loader)}] (Epoch {epoch}/{epochs}) | "
                        f"Current Step Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
                    )

                # Periodic evaluation & logging
                if global_step % eval_interval == 0 or step_in_epoch == len(train_loader):
                    val_loss, val_ppl = self.evaluate(val_loader)
                    self.model.train()

                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss

                    self.save_checkpoint(epoch, global_step, val_loss, is_best=is_best)

                    current_lr = scheduler.get_last_lr()[0]
                    avg_train_loss = epoch_loss / max(1, step_in_epoch)
                    print(
                        f"📊 Epoch [{epoch}/{epochs}] | Step [{step_in_epoch}/{len(train_loader)}] | "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Val PPL: {val_ppl:.2f} | LR: {current_lr:.6f}"
                    )

                    self.history.append(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                            "val_ppl": val_ppl,
                            "lr": current_lr,
                        }
                    )

        total_time = time.time() - start_time
        metrics_file = self.checkpoint_dir / "training_history.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_training_time_seconds": total_time,
                    "final_best_val_loss": self.best_val_loss,
                    "history": self.history,
                },
                f,
                indent=2,
            )

        print(f"\n🎉 PyGPT Pre-training Finished in {total_time / 60:.2f} minutes!")
        print(f"   Best Validation Loss: {self.best_val_loss:.4f}")
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_val_loss": str(self.best_val_loss),
            "metrics_file": str(metrics_file),
        }
