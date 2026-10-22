# configs/ — 超参唯一真源

## 作用

全项目**唯一**的配置来源。模型结构、数据管线、训练策略、路径，所有可调数字在这里定义一次；
其它文件只 `import configs.config as cfg` 然后读 `cfg.D_MODEL`，禁止就地硬编码数字。

## 将要有的文件

| 文件 | 职责 |
|---|---|
| `config.py` | 四分区常量（模型 / 数据 / 训练 / 路径）+ 派生量 + 硬约束断言 |

（不需要 `__init__.py`，Python 3 的命名空间包能直接 `from configs import config`。）

## `config.py` 的五条规矩

1. **四分区**用注释分隔，变量名全大写。
2. **每个常量一句"为什么是这个值"**——写不出理由的值不许出现。
3. **派生量由原始常量算**，不手写数字：`NUM_PATCHES` / `SEQ_LEN` / `MLP_HIDDEN`。
4. **末尾断言硬约束**：`D_MODEL % NUM_HEADS == 0`。让它在 import 时就崩，而不是训练到第 3 层 shape 才炸。
5. 不写 `from configs.config import *`，用模块别名访问。

## 验收

```bash
"C:\Users\29142\miniconda3\envs\pytorch\python.exe" -c "from configs import config as c; print(c.NUM_PATCHES, c.SEQ_LEN, c.MLP_HIDDEN, c.RUN_NAME)"
```

期望输出：`64 65 1024 vit_L8_D256_P4`
