"""evaluate.py — 测试集评估入口（项目根目录脚本）

职责（待实现，阶段 8）：
    1. 读 configs/config.py，定位 checkpoints/{RUN_NAME}/best.pt
    2. 构建模型 + load_state_dict + model.eval()
    3. 在 test 集（10000 张，全程不参与训练与选优）上统计 top-1 准确率
    4. 汇报：总准确率 + 分类别准确率

注意：test 集只在最后用一次。训练期用 val 集选优。

状态：空壳，等你写。
"""
