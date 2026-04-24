"""
Collect expert demonstration data for Behavioral Cloning.

Loads the best trained agent checkpoint, runs it in the SoccerTwos environment,
and saves (observation, action) pairs to expert_data.npz.

Usage:
    python collect_expert_data.py [--checkpoint PATH] [--episodes N] [--output PATH]
"""
import argparse
import os
import pickle

import gym
import numpy as np
import torch

import soccer_twos


# ── Default paths ────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results/PPO_team/"
    "PPO_Soccer_17b64_00000_0_2026-04-14_16-18-26/checkpoint_004500/checkpoint-4500",
)
DEFAULT_OUTPUT = "expert_data.npz"
DEFAULT_EPISODES = 500


def _load_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Extract policy weights from a Ray checkpoint file."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


def build_expert_model(obs_space, act_space, weights):
    """Build a FullyConnectedNetwork and load expert weights."""
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

    model_config = {
        "vf_share_layers": True,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }
    # num_outputs=9 for MultiDiscrete([3,3,3])
    model = FullyConnectedNetwork(obs_space, act_space, 9, model_config, "expert")
    model.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
    model.eval()
    return model


def expert_action(model, obs):
    """Get deterministic action from expert model (argmax over each sub-action)."""
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, _ = model({"obs_flat": obs_tensor, "obs": obs_tensor}, [], None)
        # logits shape: (1, 9) → split into 3 groups of 3
        action = []
        for i in range(3):
            sub_logits = logits[0, i * 3 : (i + 1) * 3]
            action.append(torch.argmax(sub_logits).item())
    return np.array(action)


def collect(checkpoint_path, num_episodes, output_path):
    """Collect expert demonstrations."""
    print(f"Loading expert weights from: {checkpoint_path}")
    weights = _load_weights(checkpoint_path, "default")

    # Create env to get spaces
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(336,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([3, 3, 3])

    model = build_expert_model(obs_space, act_space, weights)
    print(f"Expert model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create environment (multiagent, no rendering)
    env = soccer_twos.make(multiagent=True)

    all_obs = []
    all_actions = []
    total_steps = 0

    for ep in range(num_episodes):
        obs = env.reset()
        done = {"__all__": False}

        while not done.get("__all__", False):
            actions = {}
            for player_id in obs:
                action = expert_action(model, obs[player_id])
                actions[player_id] = action
                all_obs.append(obs[player_id])
                all_actions.append(action)
                total_steps += 1

            obs, reward, done, info = env.step(actions)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{num_episodes} — total samples: {total_steps:,}")

    env.close()

    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)

    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
    )

    print(f"\nDone! Saved {total_steps:,} samples to {output_path}")
    print(f"  observations: {observations.shape} ({observations.nbytes / 1e6:.1f} MB)")
    print(f"  actions: {actions.shape} ({actions.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect expert data for BC")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to expert checkpoint")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of episodes to collect")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output .npz file path")
    args = parser.parse_args()

    collect(args.checkpoint, args.episodes, args.output)
