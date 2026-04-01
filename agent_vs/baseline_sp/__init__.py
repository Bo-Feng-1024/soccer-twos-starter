import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from agent_vs.ppo_agent import PPOAgent as _PPOAgent

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../ray_results/PPO_SP"
    "/PPO_Soccer_58633_00000_0_2026-03-31_15-10-56"
    "/checkpoint_000900/checkpoint-900",
)


class Agent(_PPOAgent):
    name = "BaselineSP"

    def __init__(self, env):
        super().__init__(env, CHECKPOINT)
