"""models/block.py — ViT Encoder Block（Pre-LN）"""

from torch import nn

from configs import config as cfg
from models.attention import MultiHeadSelfAttention
from models.mlp import MLP


class EncoderBlock(nn.Module):
    """(B, N, 256) → (B, N, 256)"""

    def __init__(self):
        super().__init__()
        # 空1：两个独立的 LayerNorm + 两个子层，全部挂 self
        self.ln1 = nn.LayerNorm(cfg.D_MODEL)
        self.ln2 = nn.LayerNorm(cfg.D_MODEL)
        self.attn = MultiHeadSelfAttention()
        self.mlp = MLP()

    def forward(self, x):
        # 空2：x = x + attn(ln1(x))
        x = x + self.attn(self.ln1(x))
        # 空3：x = x + mlp(ln2(x))
        x = x + self.mlp(self.ln2(x))
        return x
