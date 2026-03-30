"""
Visualize a trained PPO agent playing SoccerTwos.

Usage:
    python watch_trained_agent.py <checkpoint_path>

Example:
    python watch_trained_agent.py ray_results/PPO_reward_shaped/PPO_Soccer_xxx/checkpoint_000010/checkpoint-10

The checkpoint_path is the file named "checkpoint-N" (no extension),
inside the checkpoint_XXXXXX folder.
"""

import sys
import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from utils import create_rllib_env


def main(checkpoint_path):
    ray.init(ignore_reinit_error=True)
    tune_registry = ray.tune.registry
    tune_registry.register_env("Soccer", create_rllib_env)

    # Restore the trainer with the same config used during training.
    # num_workers=0 → no parallel workers, runs inference in the main process.
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
    # trainer.restore() crashes on Ray 1.4.0 due to numpy.object_ in optimizer state.
    # Workaround: load the checkpoint pickle manually and apply only model weights.
    import pickle
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    worker_state = pickle.loads(checkpoint_data["worker"])
    weights = worker_state["state"]["default_policy"]["weights"]
    trainer.get_policy("default_policy").set_weights(weights)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Open a rendered environment (multiagent_player mode so we can control all 4 agents).
    # The trained policy controls team 0; team 1 stays still (action=0 for all dims).
    import soccer_twos
    import numpy as np
    from gym_unity.envs import ActionFlattener

    env = soccer_twos.make(render=True, base_port=50100)
    obs = env.reset()

    # Trainer was trained with flatten_branched=True → outputs a single int (0-26).
    # The rendered env uses default MultiDiscrete [3,3,3] → need to convert back.
    flattener = ActionFlattener(env.action_space.nvec)
    still = np.array([0, 0, 0], dtype=int)

    team0_reward = 0
    team1_reward = 0
    episode = 1

    print("Watching agent play. Close the Unity window to stop.")
    while True:
        # Team 0 uses the trained policy (players 0 and 1)
        flat0 = trainer.compute_single_action(obs[0], policy_id="default_policy")
        flat1 = trainer.compute_single_action(obs[1], policy_id="default_policy")
        action0 = flattener.lookup_action(flat0)
        action1 = flattener.lookup_action(flat1)

        obs, reward, done, info = env.step({
            0: action0,
            1: action1,
            2: still,
            3: still,
        })

        team0_reward += reward[0] + reward[1]
        team1_reward += reward[2] + reward[3]

        if max(done.values()):
            print(f"Episode {episode} | Blue: {team0_reward:.2f}  Orange: {team1_reward:.2f}")
            team0_reward = 0
            team1_reward = 0
            episode += 1
            obs = env.reset()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
