"""Minimal GPT-style language model from basic PyTorch modules."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    """Single attention head with causal mask.

    Input/Output shape:
        x: [B, T, C]
        out: [B, T, head_dim]

    The causal mask allows attention only to current and past tokens.
    Mask meaning:
        mask[i, j] == 1 if token i can attend to token j (j <= i)
    """

    def __init__(self, d_model: int, head_dim: int, dropout: float):
        """Create projection layers for one attention head."""
        super().__init__()
        self.query = nn.Linear(d_model, head_dim)
        self.key = nn.Linear(d_model, head_dim)
        self.value = nn.Linear(d_model, head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute causal self-attention for one head."""
        # x: [B, T, C]
        bsz, seq_len, _ = x.shape

        # q, k, v: [B, T, head_dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # attention scores: [B, T, T]
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))

        # Lower triangular causal mask. True means "keep".
        # mask: [T, T], where mask[i, j] = (j <= i)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        # Fill future positions with -inf so softmax gives them zero probability.
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # attn_weights: [B, T, T]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # out: [B, T, head_dim]
        out = attn_weights @ v
        return out, attn_weights if return_attention else None


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Shapes:
        x: [B, T, C]
        out: [B, T, C]
        attn_weights (optional): [B, n_heads, T, T]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """Build multiple causal attention heads and output projection."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        head_dim = d_model // n_heads
        self.heads = nn.ModuleList(
            [CausalSelfAttentionHead(d_model, head_dim, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply all heads, concatenate outputs, and project back to d_model."""
        head_outs = []
        head_weights = []

        for head in self.heads:
            out, attn = head(x, return_attention=return_attention)
            head_outs.append(out)
            if return_attention and attn is not None:
                head_weights.append(attn)

        # concat_out: [B, T, d_model]
        concat_out = torch.cat(head_outs, dim=-1)
        out = self.proj(concat_out)
        out = self.dropout(out)

        weights = None
        if return_attention:
            # [B, n_heads, T, T]
            weights = torch.stack(head_weights, dim=1)

        return out, weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network used in Transformer blocks."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """Create two-layer MLP with GELU activation."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transform.

        Shapes:
            x: [B, T, C]
            out: [B, T, C]
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        """Initialize attention, MLP, and layer norms."""
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run attention + MLP with residual paths."""
        # x: [B, T, C]
        attn_out, attn_weights = self.attn(self.ln1(x), return_attention=return_attention)
        x = x + attn_out  # residual connection
        x = x + self.ff(self.ln2(x))  # residual connection
        return x, attn_weights


class GPTLanguageModel(nn.Module):
    """Minimal GPT-style autoregressive language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ):
        """Build token/position embeddings, transformer stack, and LM head."""
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor]]:
        """Forward pass.

        Args:
            idx: Token IDs of shape [B, T].
            targets: Optional target IDs [B, T] for CE loss.
            return_attention: If True, return attention weights per block.

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar tensor or None
            all_attn_weights: list of [B, n_heads, T, T]
        """
        bsz, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size={self.block_size}")

        # token_emb: [B, T, C]
        token_emb = self.token_embedding(idx)

        # pos indices: [T]
        positions = torch.arange(seq_len, device=idx.device)
        # pos_emb: [T, C] -> broadcast to [B, T, C]
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        all_attn_weights = []
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention and attn is not None:
                all_attn_weights.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten for cross entropy:
            # logits: [B, T, V] -> [B*T, V]
            # targets: [B, T] -> [B*T]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, all_attn_weights

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate tokens.

        Args:
            idx: Prompt tokens [B, T_start].
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature (>0). Lower is greedier.

        Returns:
            Tensor of shape [B, T_start + max_new_tokens].
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
