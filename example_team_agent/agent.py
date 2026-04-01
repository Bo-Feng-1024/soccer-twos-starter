import os

from gym_unity.envs import ActionFlattener  # 把 MultiDiscrete 动作空间压平成单一 Discrete 空间
import numpy as np
import torch
from soccer_twos import AgentInterface  # 所有自定义 agent 必须实现的接口基类

from .model import QNetwork  # 导入本地定义的 Q 网络模型


class TeamAgent(AgentInterface):
    """
    An agent definition for policies trained with DQN on `team_vs_policy` variation with `single_player=True`.
    """

    def __init__(self, env):
        # 用 ActionFlattener 把 MultiDiscrete 动作空间（如 [3,3,3]）压平成一个 Discrete(27) 空间
        # env.action_space.nvec 是每个子动作维度的大小数组
        self.flattener = ActionFlattener(env.action_space.nvec)

        # 初始化 Q 网络：
        #   输入维度 = 观测空间的向量长度（每个球员的观测）
        #   输出维度 = 压平后的动作总数（所有组合）
        self.model = QNetwork(
            env.observation_space.shape[0],  # 观测向量的维度，例如 336
            self.flattener.action_space.n,   # 压平后的动作数量，例如 27
            seed=0,
        )

        # 拼接出 checkpoint.pth 的绝对路径（与 agent.py 同目录）
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth"
        )

        if os.path.isfile(weights_path):
            self.model.load_state_dict(torch.load(weights_path))  # 加载已训练好的权重
        else:
            print("Checkpoint not found.")  # 找不到权重文件时给出提示，模型保持随机初始化

        self.model.eval()  # 切换到推理模式：关闭 Dropout / BatchNorm 的训练行为

    def act(self, observation):
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        # 遍历队伍中每个球员（通常是 2 名球员，player_id = 0 或 1）
        for player_id in observation:
            # 把 numpy 观测转换为 PyTorch 张量，并增加 batch 维度 → shape: [1, state_size]
            state = torch.from_numpy(observation[player_id]).float().unsqueeze(0)

            # 前向传播，得到每个动作的 Q 值，shape: [1, action_size]
            action_values = self.model(state)

            # 取 Q 值最大的动作索引（贪婪策略）
            action = np.argmax(action_values.data.numpy())

            # 把 Discrete 动作索引还原回 MultiDiscrete 格式（如 27 → [1, 0, 2]）
            actions[player_id] = self.flattener.lookup_action(action)

        return actions  # 返回字典：{player_id: MultiDiscrete 动作数组}
