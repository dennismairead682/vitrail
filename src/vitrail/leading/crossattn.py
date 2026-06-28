from __future__ import annotations

import math

import torch
from torch import nn


class CameFusion(nn.Module):
    def __init__(self, dense_dim: int, token_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dense_dim % heads != 0:
            raise ValueError("dense_dim must divide the head count")
        self.heads = heads
        self.head_dim = dense_dim // heads
        self.q = nn.Conv3d(dense_dim, dense_dim, 1)
        self.k = nn.Linear(token_dim, dense_dim)
        self.v = nn.Linear(token_dim, dense_dim)
        self.proj = nn.Conv3d(dense_dim, dense_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        dense: torch.Tensor,
        tokens: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, d, h, w = dense.shape
        n = d * h * w
        q = self.q(dense).reshape(b, self.heads, self.head_dim, n).transpose(2, 3)
        k = self.k(tokens).reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(tokens).reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if present is not None:
            bias = (1.0 - present).view(b, 1, 1, -1) * -1e4
            scores = scores + bias
        weights = self.drop(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(weights, v)
        ctx = ctx.transpose(2, 3).reshape(b, c, d, h, w)
        residual = self.gamma * self.proj(ctx)
        if present is not None:
            gate = (present.sum(dim=1) > 0).view(b, 1, 1, 1, 1).to(residual.dtype)
            residual = residual * gate
        return dense + residual
