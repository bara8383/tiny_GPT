"""Self-attention modules (single-head and multi-head) for tiny GPT."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadCausalSelfAttention(nn.Module):
    """Single-head causal self-attention.

    Input: x [batch, seq_len, d_model]
    Output: out [batch, seq_len, head_dim]
    """

    def __init__(self, d_model: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim)
        self.k_proj = nn.Linear(d_model, head_dim)
        self.v_proj = nn.Linear(d_model, head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # x: [B, T, C]
        _, seq_len, _ = x.shape

        # Q, K, V: [B, T, head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # score: [B, T, T] = Q @ K^T / sqrt(head_dim)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask shape: [T, T]
        # Keep lower triangle (j <= i), mask upper-right triangle (future j > i).
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        score = score.masked_fill(~causal_mask, float("-inf"))

        # softmax over last dim (keys dimension)
        attn_weights = F.softmax(score, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # out: [B, T, head_dim]
        out = attn_weights @ v

        if return_weights:
            return out, attn_weights
        return out, None


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (batch_first)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model % n_heads must be 0")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # x: [B, T, C]
        bsz, seq_len, _ = x.shape

        # q, k, v projected in d_model: [B, T, C]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split C into [n_heads, head_dim], then transpose for attention math.
        # [B, T, C] -> [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # scores per head: [B, n_heads, T, T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # mask: [T, T], upper-right (future) is masked
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # softmax on last axis (keys axis)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # head outputs: [B, n_heads, T, head_dim]
        head_out = attn_weights @ v

        # Merge heads back to d_model.
        # [B, n_heads, T, head_dim] -> [B, T, n_heads, head_dim] -> [B, T, C]
        merged = head_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        # final output projection: [B, T, C]
        out = self.out_proj(merged)

        if return_weights:
            return out, attn_weights
        return out, None
