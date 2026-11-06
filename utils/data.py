"""utils/data.py — CIFAR-10 数据管线（只被 import，不直接运行）"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from configs import config as cfg


def build_transform():
    """返回确定性 transform：ToTensor + Normalize。"""
    return transforms.Compose(
        transforms=[
            # TODO: 第一件（ToTensor）
            transforms.ToTensor(),
            # TODO: 第二件（Normalize，两个参数都从 cfg 取）
            transforms.Normalize(mean=cfg.CIFAR_MEAN, std=cfg.CIFAR_STD),
        ]
    )


def build_datasets():
    """返回 (train_set, val_set, test_set)，均已套好 transform。"""
    tf_train = build_transform()
    tf_eval = build_transform()

    train_dataset = datasets.CIFAR10(
        root=cfg.DATA_ROOT, train=True, download=True, transform=tf_train
    )
    val_dataset = datasets.CIFAR10(
        root=cfg.DATA_ROOT, train=True, download=True, transform=tf_eval
    )
    test_dataset = datasets.CIFAR10(
        root=cfg.DATA_ROOT, train=False, download=True, transform=tf_eval
    )

    # 从 train 中切出 val 的索引（不直接切，避免 val 被视为 test  contaminate 下游评估）
    g = torch.Generator().manual_seed(cfg.SEED)
    idx = torch.randperm(len(train_dataset), generator=g)
    train_idx, val_idx = idx[cfg.VAL_SIZE :].tolist(), idx[: cfg.VAL_SIZE].tolist()

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)

    return train_set, val_set, test_dataset


def build_dataloaders():
    """返回 (train_loader, val_loader, test_loader)。"""
    train_set, val_set, test_set = build_datasets()

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_set, val_set, test_set = build_datasets()
    tr, va, te = build_dataloaders()

    x, y = next(iter(tr))
    print("x:", tuple(x.shape), x.dtype, "| y:", tuple(y.shape), y.dtype)
    print(
        "min/max/mean/std: %.3f %.3f %.4f %.4f" % (x.min(), x.max(), x.mean(), x.std())
    )
    print("len  :", len(train_set), len(val_set), len(test_set))
    print("iters:", len(tr), len(va), len(te))
    print("overlap:", len(set(train_set.indices) & set(val_set.indices)))
