"""evaluate.py — 测试集评估入口（项目根目录脚本）

职责（待实现，阶段 8）：
    1. 读 configs/config.py，定位 checkpoints/{RUN_NAME}/best.pt
    2. 构建模型 + load_state_dict + model.eval()
    3. 在 test 集（10000 张，全程不参与训练与选优）上统计 top-1 准确率
    4. 汇报：总准确率 + 分类别准确率

注意：test 集只在最后用一次。训练期用 val 集选优。

状态：空壳，等你写。
"""

"""evaluate.py — 测试集评估入口（项目根目录脚本）"""
import os

import torch

from configs import config as cfg
from models.vit import ViT
from utils.data import build_dataloaders

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_model(ckpt_path, device):
    model = ViT().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_test(model, loader, device, num_classes, max_batches=None):
    per_correct = [0] * num_classes
    per_total = [0] * num_classes
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct = pred == y
            for c in range(num_classes):
                mask = y == c
                per_total[c] += mask.sum().item()
                per_correct[c] += (correct & mask).sum().item()
    return sum(per_correct) / sum(per_total), per_correct, per_total


def main(max_batches=None):
    ckpt = os.path.join(cfg.CKPT_DIR, "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("找不到权重: %s" % ckpt)
    _, _, test_loader = build_dataloaders()
    model = load_model(ckpt, cfg.DEVICE)
    acc, per_correct, per_total = evaluate_test(
        model, test_loader, cfg.DEVICE, cfg.NUM_CLASSES, max_batches
    )

    print(
        "top-1 accuracy: %.2f%%  (%d/%d)"
        % (acc * 100, sum(per_correct), sum(per_total))
    )
    print("分类别准确率:")
    for c, name in enumerate(CLASS_NAMES):
        print(
            "  %-12s %6.2f%%  (%d/%d)"
            % (
                name,
                100.0 * per_correct[c] / per_total[c],
                per_correct[c],
                per_total[c],
            )
        )


if __name__ == "__main__":
    main()
