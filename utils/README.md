# utils/ — 工具层（不含模型结构）

## 作用

只放"数据的搬运与加工"，不放任何 `nn.Module`（模型结构归 `models/`）。

## 将要有的文件

| 文件 | 职责 |
|---|---|
| `data.py` | CIFAR-10 的 Dataset 包装、train/val 切分（45000 / 5000）、transforms（ToTensor + Normalize，基础版不加增强）、DataLoader 构建 |

后续可能新增（用到再写，不预建空文件）：

| 文件 | 职责 |
|---|---|
| `metrics.py` | top-1 准确率统计（也可先写在 train.py 里） |
| `seed.py` | 随机种子固定（`SEED` 已在 config 中预留） |

## 规矩

- 本目录文件**只被 import，不直接运行**。入口脚本一律放项目根目录（Python 模块搜索起点规则）。
- 归一化的均值/方差从 `configs/config.py` 读，不在这里写死。
- `NUM_WORKERS` 在 Windows 本机设 0（避免多进程 spawn 重复下载/卡死），服务器 Linux 再改 4。
