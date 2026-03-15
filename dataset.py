"""Dataset helpers for next-token prediction."""

import torch
from torch.utils.data import Dataset


class CharLanguageModelingDataset(Dataset):
    """Create (input, target) pairs for autoregressive training.

    Input sequence x and target sequence y have same length `block_size`.
    y is x shifted by one token to the left.

    Shapes:
        x: [block_size]
        y: [block_size]
    """

    def __init__(self, token_ids: list[int], block_size: int):
        """Initialize from encoded text and context length."""
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        """Number of trainable windows."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one autoregressive training pair."""
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
