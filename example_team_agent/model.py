import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):  # 继承 PyTorch 的 nn.Module，定义 DQN 的 Q 网络
    def __init__(self, state_size, action_size, seed=0):
        """
        A fully connected neural network with two hidden layers.

        Parameters
        ----------
        state_size (int): Observation dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        super(QNetwork, self).__init__()  # 调用父类初始化，注册所有子模块
        self.seed = torch.manual_seed(seed)  # 设置随机种子，保证结果可复现
        self.fc1 = nn.Linear(state_size, 32)   # 第一层：输入层 → 32 个神经元
        self.fc2 = nn.Linear(32, 64)            # 第二层：32 → 64 个神经元
        self.fc3 = nn.Linear(64, action_size)   # 第三层：64 → 输出每个动作的 Q 值

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))  # 第一层线性变换 + ReLU 激活（引入非线性）
        x = F.relu(self.fc2(x))  # 第二层线性变换 + ReLU 激活
        x = self.fc3(x)          # 第三层线性变换，输出各动作的 Q 值（不加激活）

        return x  # 返回形状为 [batch_size, action_size] 的 Q 值张量
