import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from agent_vs.ppo_agent import PPOAgent as _PPOAgent

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../ray_results/PPO_reward_shaped"
    "/PPO_Soccer_62a5a_00000_0_2026-03-30_21-31-50"
    "/checkpoint_000890/checkpoint-890",
)


class Agent(_PPOAgent):
    name = "RewardShaped"

    def __init__(self, env):
        super().__init__(env, CHECKPOINT)
