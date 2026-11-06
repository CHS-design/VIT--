"""models/patch_embed.py — 把图像切成 patch 并投影成 token"""

import torch
from torch import nn

from configs import config as cfg


class PatchEmbedding(nn.Module):
    """(B, 3, 32, 32) → (B, 64, 256)"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.D_MODEL,
            kernel_size=cfg.PATCH_SIZE,
            stride=cfg.PATCH_SIZE,
        )

    def forward(self, x):
        x = self.proj(x)
        x = torch.flatten(x, start_dim=2)
        x = x.transpose(1, 2)
        return x
