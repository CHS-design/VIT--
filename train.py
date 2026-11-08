"""train.py — 训练入口（项目根目录脚本）

职责（待实现，阶段 7）：
    1. 读 configs/config.py
    2. 构建 DataLoader             ← utils/data.py
    3. 构建模型并 .to(cfg.DEVICE)  ← models/vit.py
    4. 标准五步循环：zero_grad → forward → loss → backward → step
    5. 每 epoch 验证：model.eval() + torch.no_grad()
    6. best_loss 选优存 best.pt；patience 早停
    7. loss / acc 历史落盘 loss_history.json

状态：空壳，等你写。
"""

import json
import math
import os
import random

import numpy as np
import torch
from torch import nn

from configs import config as cfg
from models.vit import ViT
from utils.data import build_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name in ("cls_token", "pos_embed"):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        params=[
            {"params": decay, "weight_decay": cfg.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.LR,
    )
    return optimizer


def build_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(1, cfg.EPOCHS - cfg.WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  # ①
        logits = model(x)  # ②
        loss = criterion(logits, y)  # ③
        loss.backward()  # ④
        optimizer.step()  # ⑤
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def main(max_train_batches=None, max_val_batches=None):
    set_seed(cfg.SEED)
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    tr, va, te = build_dataloaders()
    model = ViT().to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    best_loss, patience, history = float("inf"), 0, []

    for epoch in range(1, cfg.EPOCHS + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, tr, criterion, optimizer, cfg.DEVICE, max_train_batches
        )
        val_loss, val_acc = evaluate(model, va, criterion, cfg.DEVICE, max_val_batches)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr_now,
            }
        )
        print(
            "epoch %3d | train %.4f acc %.3f | val %.4f acc %.3f | lr %.2e"
            % (epoch, train_loss, train_acc, val_loss, val_acc, lr_now)
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.CKPT_DIR, "best.pt"))
            patience = 0
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print("early stop at epoch %d" % epoch)
                break

    with open(
        os.path.join(cfg.CKPT_DIR, "loss_history.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
