"""Dataset helpers for next-token prediction."""

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """Build fixed-length next-token prediction samples from full token sequence.

    For each i:
    x = tokens[i : i + seq_len]
    y = tokens[i + 1 : i + seq_len + 1]

    x.shape == [seq_len]
    y.shape == [seq_len]
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        self.tokens = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.tokens[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y
