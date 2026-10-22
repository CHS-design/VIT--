# data/ — 数据集落盘目录（不入 git）

## 作用

存放 torchvision 自动下载的 CIFAR-10。首次运行 `datasets.CIFAR10(root="./data", download=True)`
会自动创建子目录，不需要手写任何代码。

## 将要有的内容

```
data/
└── cifar-10-batches-py/
    ├── data_batch_1 ~ data_batch_5   # 训练集 5 批，共 50000 张
    ├── test_batch                    # 测试集 10000 张
    ├── batches.meta                  # 类别名（airplane ~ truck）
    └── readme.html
```

## 约定

- 整个目录 git 忽略，仅保留本 README。
- 数据文件不要手改。需要重下就删掉 `cifar-10-batches-py/` 再跑一次。
- 归一化用的均值/方差写在 `configs/config.py`（`CIFAR_MEAN` / `CIFAR_STD`），不在这里。
- 本目录**只放数据**，不放代码、不放模型权重。
