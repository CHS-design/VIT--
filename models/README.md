# models/ — 模型零件（一零件一文件）

## 作用

ViT 的全部结构。每个零件独立成文件，可单独实例化、单独喂玩具张量验证。
零件全部通过三关（形状 / dtype / 数值）后才允许组装。

## 将要有的文件

| 文件 | 零件 | 输入 → 输出 | 要点 |
|---|---|---|---|
| `patch_embed.py` | PatchEmbedding | `B×3×32×32` → `B×64×256` | Conv2d(3,256,k=4,s=4) 与 reshape+Linear 数学等价，两种写法都要能对上 |
| `attention.py` | MultiHeadSelfAttention | `B×65×256` → `B×65×256` | 复用上个项目，只改一处：**去掉 mask** |
| `mlp.py` | MLP | `B×65×256` → `B×65×256` | 复用 PositionwiseFeedForward，隐藏维 1024，逐位置独立 |
| `block.py` | EncoderBlock | `B×65×256` → `B×65×256` | **Pre-LN**：`x + MHSA(LN(x))`，`x + MLP(LN(x))`（与上项目 Post-LN 相反） |
| `vit.py` | ViT 总装 | `B×3×32×32` → `B×10` | CLS token + 可学习位置嵌入 + 8×Block + LayerNorm + 分类头 |

## 规矩

- **禁止**：`timm`、`torchvision.models`、`nn.MultiheadAttention`、`nn.TransformerEncoderLayer`、`F.scaled_dot_product_attention`。
  这些是"论文里的结构或数学"，必须自己写。
- **允许**：`nn.Linear`、`nn.Conv2d`、`nn.LayerNorm`、`nn.Parameter`、`F.softmax` 等基础算子。
- 一个文件一个"论文级零件"，工具函数下沉到 `utils/`。
- 每个文件的 `__main__` 里可放本零件的自检代码（喂玩具张量打印形状）。
