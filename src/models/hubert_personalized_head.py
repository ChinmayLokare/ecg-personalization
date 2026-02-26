# src/models/hubert_personalized_head.py
from __future__ import annotations

import torch
import torch.nn as nn

from models.adapter import ResidualAdapter


class HuBERTPersonalizedHead(nn.Module):
    """
    Pooled HuBERT-ECG embedding (B, 512) -> adapter -> classifier logits (B, 4)
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 4,
        adapter_bottleneck: int = 128,
        adapter_dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.adapter = ResidualAdapter(dim=in_dim, bottleneck=adapter_bottleneck, dropout=adapter_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.classifier(x)