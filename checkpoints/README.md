# checkpoints/ — 训练产物（不入 git）

## 作用

按 `RUN_NAME` 分实验子目录保存结果。同名重跑 = 覆盖旧档。

## 将要有的结构

```
checkpoints/
└── vit_L8_D256_P4/            # = config.RUN_NAME
    ├── best.pt                # 验证集最优权重（state_dict）
    ├── loss_history.json      # 每 epoch 的 train/val loss 与 acc
    └── loss_curve.png         # 由 json 独立画图，可随时重跑
```

## 规矩

- 目录名**等于** `config.RUN_NAME`，固定名，**不用时间戳**——训练与推理是两个独立进程，
  时间戳目录会导致推理时找不到权重。
- 换配置（层数/维度/batch）就改 `RUN_NAME`，别覆盖有意义的旧结果。
- `*.pt` / `*.json` / `*.png` 全部 git 忽略，仅保留本 README。
- 权重只存 `state_dict`，不存整个模型对象（跨环境加载更稳）。
