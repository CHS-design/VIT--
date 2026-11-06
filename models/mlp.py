"""models/mlp.py — 逐位置前馈网络"""

from torch import nn

from configs import config as cfg


class MLP(nn.Module):
    """(B, N, 256) → (B, N, 256)"""

    def __init__(self):
        super().__init__()
        # 空1：256 → 1024 的线性层
        self.linear1 = nn.Linear(cfg.D_MODEL, cfg.MLP_HIDDEN)
        # 空2：激活函数
        self.GELU = nn.GELU()
        # 空3：1024 → 256 的线性层
        self.linear2 = nn.Linear(cfg.MLP_HIDDEN, cfg.D_MODEL)

    def forward(self, x):
        # 空4：三步串起来
        x = self.linear1(x)
        x = self.GELU(x)
        x = self.linear2(x)
        return x
