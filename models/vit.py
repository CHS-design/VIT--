"""models/vit.py — ViT 总装"""
import torch
from torch import nn
from configs import config as cfg
from models.patch_embed import PatchEmbedding
from models.block import EncoderBlock


class ViT(nn.Module):
    """(B, 3, 32, 32) → (B, 10)"""

    def __init__(self):
        super().__init__()
        # 空1：self.patch_embed
        self.patch_embed = PatchEmbedding()
        # 空2：self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.D_MODEL))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.D_MODEL))
        # 空3：self.pos_embed = nn.Parameter(torch.zeros(1, cfg.SEQ_LEN, cfg.D_MODEL))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.SEQ_LEN, cfg.D_MODEL))
        # 空4：self.blocks = nn.ModuleList([...])
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(cfg.NUM_LAYERS)])
        # 空5：self.ln、self.head
        self.ln = nn.LayerNorm(cfg.D_MODEL)
        self.head = nn.Linear(cfg.D_MODEL, cfg.NUM_CLASSES)

    def forward(self, x):
        B = x.shape[0]
        # 空6：x = self.patch_embed(x)
        x = self.patch_embed(x)
        # 空7：拼 CLS（torch.cat，dim=1）
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 空8：x = x + self.pos_embed
        x = x + self.pos_embed
        # 空9：for blk in self.blocks: x = blk(x)
        for blk in self.blocks:
            x = blk(x)
        # 空10：x = self.ln(x) → 取 x[:, 0] → self.head
        x = self.ln(x)
        x = x[:, 0]
        x = self.head(x)
        return x
