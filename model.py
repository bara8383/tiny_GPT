"""GPT-like language model for character-level next-token prediction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from block import TransformerBlock


class GPTLanguageModel(nn.Module):
    """A tiny GPT-style decoder-only Transformer."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        # input_ids: [batch, seq_len]
        bsz, t = input_ids.shape
        if t > self.seq_len:
            raise ValueError(f"input length {t} exceeds model seq_len={self.seq_len}")

        # token_emb: [batch, seq_len, d_model]
        token_emb = self.token_embedding(input_ids)

        # pos_emb: [seq_len, d_model] -> broadcast to [batch, seq_len, d_model]
        pos_ids = torch.arange(t, device=input_ids.device)
        pos_emb = self.position_embedding(pos_ids)

        # x after embedding sum: [batch, seq_len, d_model]
        x = token_emb + pos_emb

        all_weights = []
        for block in self.blocks:
            x, weights = block(x, return_weights=return_attention)
            if return_attention and weights is not None:
                all_weights.append(weights)

        x = self.final_ln(x)

        # logits: [batch, seq_len, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # reshape so cross entropy sees each time-step as one classification sample
            # logits: [B, T, V] -> [B*T, V], targets: [B, T] -> [B*T]
            loss = F.cross_entropy(logits.reshape(bsz * t, -1), targets.reshape(bsz * t))

        return logits, loss, all_weights
