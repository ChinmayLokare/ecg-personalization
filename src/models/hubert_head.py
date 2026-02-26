# src/models/hubert_head.py
from __future__ import annotations

import torch
import torch.nn as nn


class HuBERTClassifierHead(nn.Module):
    """
    Head for HuBERT-ECG outputs.

    HuBERT-ECG (HF) returns last_hidden_state: (B, L, 512) for hubert-ecg-small.
    We pool over L (mean) -> (B, 512) then classify -> (B, 4).
    """

    def __init__(self, in_dim: int = 512, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)