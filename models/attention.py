"""models/attention.py — 多头自注意力（无 mask）"""

import math

import torch.nn.functional as F
from torch import nn

from configs import config as cfg


class MultiHeadSelfAttention(nn.Module):
    """(B, N, 256) → (B, N, 256)；N 由调用方决定，本模块不硬编码"""

    def __init__(self):
        super().__init__()
        self.num_heads = cfg.NUM_HEADS
        self.d_k = cfg.D_MODEL // cfg.NUM_HEADS  # 32
        # TODO 1: 四个线性层
        self.q_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL, bias=cfg.QKV_BIAS)
        self.k_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL, bias=cfg.QKV_BIAS)
        self.v_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL, bias=cfg.QKV_BIAS)
        self.out_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)

    def forward(self, x):
        B, N, _ = x.shape
        # TODO 2: 投影 + 拆头 → q/k/v 各 (B, 8, N, 32)
        q = self.q_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        # TODO 3: 打分、缩放、softmax → (B, 8, N, N)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        # TODO 4: 加权和、拼头、输出投影 → (B, N, 256)
        out = (attn @ v).transpose(1, 2).reshape(B, N, cfg.D_MODEL)
        out = self.out_proj(out)
        return out
