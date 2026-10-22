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
