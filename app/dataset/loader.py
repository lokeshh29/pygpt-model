import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class PyGPTTokenDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction.
    Given a sequence of token IDs, generates inputs x = tokens[i : i + seq_len]
    and targets y = tokens[i + 1 : i + seq_len + 1].
    """

    def __init__(self, token_file: str, seq_len: int = 512):
        self.token_file = Path(token_file)
        self.seq_len = seq_len

        if not self.token_file.exists():
            raise FileNotFoundError(f"Token file not found: {token_file}")

        print(f"📖 Loading token dataset from: {self.token_file.name}...")
        with open(self.token_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.tokens = torch.tensor(data["tokens"], dtype=torch.long)
        self.total_tokens = len(self.tokens)

        # Calculate number of valid full context sequences
        self.num_samples = max(0, (self.total_tokens - 1) // self.seq_len)
        print(f"✅ Loaded {self.total_tokens:,} tokens -> {self.num_samples:,} samples (seq_len={self.seq_len})")

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
) -> Tuple[DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for training and validation datasets."""
    train_dataset = PyGPTTokenDataset(train_file, seq_len=seq_len)
    val_dataset = PyGPTTokenDataset(val_file, seq_len=seq_len)

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
