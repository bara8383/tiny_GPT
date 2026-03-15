"""Transformer block components for tiny GPT."""

import torch
import torch.nn as nn

from attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    [B, T, d_model] -> [B, T, d_model]
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block.

    Pre-LN means we apply LayerNorm before each sub-layer.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # Attention sub-layer (with residual):
        # x <- x + MHA(LN(x))
        attn_out, weights = self.attn(self.ln1(x), return_weights=return_weights)
        x = x + attn_out

        # FFN sub-layer (with residual):
        # x <- x + FFN(LN(x))
        x = x + self.ffn(self.ln2(x))

        return x, weights
