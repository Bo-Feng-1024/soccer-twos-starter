"""
Generic PPO agent loader for watch comparisons.
Uses the same approach as watch_trained_agent.py:
  - builds PPOTrainer with a real Soccer env (worker_id offset to avoid port conflict)
  - loads only model weights from the checkpoint pickle (skips optimizer state)
"""
import pickle
import os
import sys

import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from gym_unity.envs import ActionFlattener
from soccer_twos import AgentInterface, EnvType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import create_rllib_env

# Each agent needs a different base_port so Unity envs don't collide.
# watch.py's env uses mlagents default (5004); PPOTrainer envs start at 50100.
_next_base_port = [50100]


def _load_weights(trainer, checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    worker_state = pickle.loads(checkpoint_data["worker"])
    weights = worker_state["state"]["default_policy"]["weights"]
    trainer.get_policy("default_policy").set_weights(weights)


class PPOAgent(AgentInterface):
    def __init__(self, env, checkpoint_path):
        super().__init__()

        ray.init(ignore_reinit_error=True)
        ray.tune.registry.register_env("Soccer", create_rllib_env)

        base_port = _next_base_port[0]
        _next_base_port[0] += 1

        trainer = PPOTrainer(
            config={
                "num_workers": 0,
                "framework": "torch",
                "env": "Soccer",
                "env_config": {
                    "variation": EnvType.team_vs_policy,
                    "multiagent": False,
                    "single_player": True,
                    "flatten_branched": True,
                    "opponent_policy": lambda *_: 0,
                    "base_port": base_port,
                },
                "model": {
                    "vf_share_layers": True,
                    "fcnet_hiddens": [512],
                },
            }
        )
        _load_weights(trainer, checkpoint_path)
        self.policy = trainer.get_policy("default_policy")
        self.flattener = ActionFlattener(env.action_space.nvec)

    def act(self, observation):
        actions = {}
        for player_id, obs in observation.items():
            flat_action = self.policy.compute_single_action(obs)[0]
            actions[player_id] = self.flattener.lookup_action(flat_action)
        return actions
