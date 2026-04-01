"""
Visualize a trained PPO agent playing SoccerTwos.

Usage:
    # Blue team (trained) vs still opponent
    python watch_trained_agent.py <checkpoint_path>

    # Blue team (trained) vs random opponent
    python watch_trained_agent.py <checkpoint_path> --random

Example:
    python watch_trained_agent.py ray_results/PPO_reward_shaped/PPO_Soccer_xxx/checkpoint_000890/checkpoint-890 --random
"""

import sys
import pickle
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from gym_unity.envs import ActionFlattener
import soccer_twos
from soccer_twos import EnvType

from utils import create_rllib_env


def load_weights(trainer, checkpoint_path):
    # trainer.restore() crashes on Ray 1.4.0 due to numpy.object_ in optimizer state.
    # Workaround: load the checkpoint pickle manually and apply only model weights.
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    worker_state = pickle.loads(checkpoint_data["worker"])
    weights = worker_state["state"]["default_policy"]["weights"]
    trainer.get_policy("default_policy").set_weights(weights)


def main(checkpoint_path, random_opponent):
    ray.init(ignore_reinit_error=True)
    ray.tune.registry.register_env("Soccer", create_rllib_env)

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
                "reward_shaping": True,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
        }
    )
    load_weights(trainer, checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")

    env = soccer_twos.make(render=True, base_port=50100)
    obs = env.reset()

    # Trainer outputs a flat Discrete int; rendered env needs MultiDiscrete [3,3,3].
    flattener = ActionFlattener(env.action_space.nvec)

    opponent_label = "random" if random_opponent else "still"
    print(f"Blue = trained agent  |  Orange = {opponent_label} opponent")
    print("Close the Unity window to stop.\n")

    team0_reward = 0
    team1_reward = 0
    episode = 1
    wins = losses = draws = 0

    while True:
        flat0 = trainer.compute_single_action(obs[0], policy_id="default_policy")
        flat1 = trainer.compute_single_action(obs[1], policy_id="default_policy")
        action0 = flattener.lookup_action(flat0)
        action1 = flattener.lookup_action(flat1)

        if random_opponent:
            action2 = env.action_space.sample()
            action3 = env.action_space.sample()
        else:
            action2 = np.array([0, 0, 0], dtype=int)
            action3 = np.array([0, 0, 0], dtype=int)

        obs, reward, done, info = env.step({
            0: action0, 1: action1,
            2: action2, 3: action3,
        })

        team0_reward += reward[0] + reward[1]
        team1_reward += reward[2] + reward[3]

        if max(done.values()):
            if team0_reward > team1_reward:
                wins += 1
                result = "WIN"
            elif team0_reward < team1_reward:
                losses += 1
                result = "LOSS"
            else:
                draws += 1
                result = "DRAW"

            total = wins + losses + draws
            print(f"Ep {episode:4d} [{result}]  Blue: {team0_reward:+.2f}  Orange: {team1_reward:+.2f}"
                  f"  |  W/L/D: {wins}/{losses}/{draws}  win%: {wins/total*100:.0f}%")
            team0_reward = 0
            team1_reward = 0
            episode += 1
            obs = env.reset()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    random_opponent = "--random" in sys.argv
    main(sys.argv[1], random_opponent)
