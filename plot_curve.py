"""plot_curve.py — 由 loss_history.json 独立画图（训练与可视化分离）"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # 必须在 import pyplot 之前：服务器无图形界面也能存图
import matplotlib.pyplot as plt

from configs import config as cfg


def main():
    hist_path = os.path.join(cfg.CKPT_DIR, "loss_history.json")
    out_path = os.path.join(cfg.CKPT_DIR, "loss_curve.png")
    if not os.path.exists(hist_path):
        raise FileNotFoundError("找不到训练历史: %s" % hist_path)

    with open(hist_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, lrs, color="tab:green")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].set_title("LR schedule")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
