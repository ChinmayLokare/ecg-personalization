# src/models/baseline_cnn.py
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineECGCNN(nn.Module):
    """
    Simple 1D CNN baseline for 10s single-lead ECG windows.
    Input:  x  -> (B, 1, 3000)
    Output: logits -> (B, num_classes)
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()

        def conv_block(cin: int, cout: int, k: int, p: int):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=p, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            )

        # Keep it small but strong enough
        self.features = nn.Sequential(
            conv_block(in_channels, 32, k=7, p=3),
            conv_block(32, 64, k=5, p=2),
            conv_block(64, 128, k=5, p=2),
            conv_block(128, 256, k=3, p=1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x