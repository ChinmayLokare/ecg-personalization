# src/models/adapter.py
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """
    Small residual MLP adapter: y = x + f(x)
    Used for "personalized layer" on top of frozen foundation representations.
    """

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)