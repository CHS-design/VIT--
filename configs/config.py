# configs/config.py —— 全项目超参唯一真源
# 其它文件一律 `import configs.config as cfg` 后读 cfg.XXX，禁止就地硬编码数字

import torch  # 仅用于分区三的 DEVICE 自动判断

# ============================================================
# 一、模型结构（决定参数量与张量形状）
# ============================================================
IMG_SIZE    = 32    # CIFAR-10 原生 32×32，不 resize：省一次插值，也避免上采样失真
PATCH_SIZE  = 4     # 4×4 → (32/4)² = 64 个 patch；换 16×16 只剩 4 个 token，粒度太粗
IN_CHANNELS = 3     # RGB 三通道
NUM_CLASSES = 10    # CIFAR-10 共 10 类

D_MODEL     = 256   # 4GB 显存 + 6.35M 参数量级的折中；翻到 512 参数约 ×3.5，CIFAR 吃不下
NUM_HEADS   = 8     # d_k = 256/8 = 32，与上个项目同量级；头太少表达弱，太多注意力碎片化
NUM_LAYERS  = 8     # 先跑通链路；终版升 12 只改这一行（6.35M → 9.51M）
MLP_RATIO   = 4     # 论文原值：FFN 隐藏维 = 4 × D_MODEL = 1024

DROPOUT     = 0.0   # 基础版先关掉，排除干扰变量；等 train/val 曲线分叉了再开 0.1
QKV_BIAS    = True  # 论文 ViT-B 设定：Q/K/V 投影层带 bias

# ============================================================
# 二、数据
# ============================================================
DATA_ROOT   = "./data"  # 相对项目根；对应 .gitignore 的 `data/*`，首次运行自动下载
BATCH_SIZE  = 32        # 本机 4GB 只做冒烟；服务器改 128（5080 上 256 也放得下）
NUM_WORKERS = 0         # Windows 多进程 spawn 需 __main__ 保护，先 0；服务器 Linux 改 4
VAL_SIZE    = 5000      # 从 train 的 50000 里切；test 10000 由 torchvision 自带，全程不动
CIFAR_MEAN  = (0.4914, 0.4822, 0.4465)  # 按通道均值（3 元组）：Normalize 逐通道对齐，单值会报错
CIFAR_STD   = (0.2470, 0.2435, 0.2616)  # 按通道标准差；归一化后像素分布 ≈ N(0,1)

# ============================================================
# 三、训练
# ============================================================
EPOCHS        = 100     # 方案 A 的量级；5080 上可拉到 200，配合早停不会白跑
LR            = 3e-4    # ViT 从零训练用 AdamW 的常用起点，比 CNN 的 1e-3 低一档
WEIGHT_DECAY  = 0.05    # 论文值；只作用于权重矩阵，bias 与 LayerNorm 参数不加（train.py 里分组）
WARMUP_EPOCHS = 5       # 前 5 轮 lr 线性升到 LR：无 warmup 时 ViT 常在头两轮直接崩到随机水平
PATIENCE      = 10      # 连续 10 轮 val loss 不创新低就早停，避免过拟合后空耗
SEED          = 42      # torch / random / numpy 三处都要设，切分与初始化才可复现
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"  # 存字符串不存 torch.device，config 保持纯数据可打印

# ============================================================
# 四、路径
# ============================================================
RUN_NAME = "vit_L8_D256_P4"              # 实验名编码关键配置；固定名不用时间戳——训练与推理是两个进程，时间戳目录会找不到权重
CKPT_DIR = f"./checkpoints/{RUN_NAME}"   # 从 RUN_NAME 派生，禁止另写一遍路径

# ============================================================
# 派生量（由上面常量算出，勿手改）
# ============================================================
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 8 × 8 = 64：每边切 8 块，共 64 个图像 token
SEQ_LEN     = NUM_PATCHES + 1                 # 64 + 1(CLS) = 65：送进 Transformer 的序列长度
MLP_HIDDEN  = D_MODEL * MLP_RATIO             # 256 × 4 = 1024：FFN 中间层宽度

# ============================================================
# 硬约束（import 时就炸，别等训练到一半才发现）
# ============================================================
assert D_MODEL % NUM_HEADS == 0, f"D_MODEL({D_MODEL}) 不能被 NUM_HEADS({NUM_HEADS}) 整除，拆头时 reshape 会失败"
assert IMG_SIZE % PATCH_SIZE == 0, f"IMG_SIZE({IMG_SIZE}) 不能被 PATCH_SIZE({PATCH_SIZE}) 整除，边缘像素会被静默丢弃"
assert len(CIFAR_MEAN) == IN_CHANNELS, f"CIFAR_MEAN 长度 {len(CIFAR_MEAN)} ≠ 通道数 {IN_CHANNELS}，Normalize 会形状不匹配"
