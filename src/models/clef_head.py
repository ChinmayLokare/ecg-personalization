# src/models/clef_head.py
from __future__ import annotations

import torch
import torch.nn as nn


class CLEFClassifierHead(nn.Module):
    """
    Simple classification head for CLEF embeddings.
    CLEF sanity output: (B, 256)
    """

    def __init__(self, in_dim: int = 256, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)