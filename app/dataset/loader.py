import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class PyGPTTokenDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction with memory-efficient token slicing.
    Given a sequence of token IDs, generates inputs x = tokens[i : i + seq_len]
    and targets y = tokens[i + 1 : i + seq_len + 1].
    """

    def __init__(self, token_file: str, seq_len: int = 512, max_samples: Optional[int] = None):
        self.token_file = Path(token_file)
        self.seq_len = seq_len

        if not self.token_file.exists():
            raise FileNotFoundError(f"Token file not found: {token_file}")

        print(f"📖 Loading token dataset from: {self.token_file.name}...")
        with open(self.token_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_tokens = data["tokens"]
        self.total_tokens = len(raw_tokens)
        
        # If max_samples specified, cap for memory safety; otherwise use 100% of tokens
        if max_samples is not None:
            max_tokens_to_keep = (max_samples * self.seq_len) + 1
            if self.total_tokens > max_tokens_to_keep:
                raw_tokens = raw_tokens[:max_tokens_to_keep]

        self.tokens = torch.tensor(raw_tokens, dtype=torch.long)
        self.num_samples = max(0, (len(self.tokens) - 1) // self.seq_len)
        print(f"✅ Loaded {len(self.tokens):,} tokens -> {self.num_samples:,} samples (seq_len={self.seq_len})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1 : end_idx + 1]
        return x, y


def create_dataloaders(
    train_file: str = "data/processed/train_tokens.json",
    val_file: str = "data/processed/val_tokens.json",
    seq_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for training and validation datasets."""
    train_dataset = PyGPTTokenDataset(train_file, seq_len=seq_len, max_samples=max_samples)
    val_dataset = PyGPTTokenDataset(val_file, seq_len=seq_len, max_samples=max_samples // 10 if max_samples else None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader
