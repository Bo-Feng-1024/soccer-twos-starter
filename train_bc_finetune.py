"""
PPO fine-tuning with Behavioral Cloning initialization.

Loads BC-pretrained weights into the PPO policy, then fine-tunes against
100% CEIA baseline opponent. Based on train_focused_ceia.py.

Key differences from train_focused_ceia.py:
  - No checkpoint restore — starts fresh trainer, injects BC weights
  - Lower lr (5e-5) to preserve BC initialization
  - Higher entropy_coeff (0.01) to encourage exploration from deterministic BC policy
  - lr_schedule starts from 0 (not 21M)
"""
import os
import pickle

import ray
import torch
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

BC_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bc_weights.pt",
)
# ──────────────────────────────────────────────────────────────────────────────


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Team 0 (agents 0, 1) → "default" (being trained)
    Team 1 (agents 2, 3) → always "opponent_1" (loaded with ceia weights)
    """
    if agent_id == 0 or agent_id == 1:
        return "default"
    return "opponent_1"


def _load_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Extract policy weights from a Ray checkpoint file."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class BCInitCallback(DefaultCallbacks):
    """
    On the first training iteration:
      1. Inject BC-pretrained weights into the "default" policy
      2. Load CEIA weights into all opponent slots
    """

    def __init__(self):
        super().__init__()
        self._initialized = False

    def on_train_result(self, **info):
        if not self._initialized:
            trainer = info["trainer"]

            # 1. Inject BC weights into "default" policy
            print("=== Loading BC-pretrained weights into default policy ===")
            try:
                bc_state = torch.load(BC_WEIGHTS_PATH, map_location="cpu")
                bc_weights = {k: v.numpy() for k, v in bc_state.items()}
                trainer.get_policy("default").set_weights(bc_weights)
                print("=== BC weights injected successfully ===")
            except Exception as e:
                print(f"WARNING: failed to load BC weights: {e}")

            # 2. Load CEIA weights into opponents
            print("=== Loading ceia_baseline into all opponents ===")
            try:
                ceia_weights = _load_weights(CEIA_CHECKPOINT)
                trainer.set_weights({
                    "opponent_1": ceia_weights,
                    "opponent_2": ceia_weights,
                    "opponent_3": ceia_weights,
                })
                print("=== All opponents = ceia_baseline (fixed forever) ===")
            except Exception as e:
                print(f"WARNING: failed to load ceia_baseline: {e}")

            self._initialized = True


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    tmp = create_rllib_env()
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()

    analysis = tune.run(
        "PPO",
        name="PPO_bc_finetune",
        config={
            # ── System ────────────────────────────────────────────────────
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": BCInitCallback,

            # ── Multiagent: 4 policies (compatible with existing checkpoints)
            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },

            # ── Environment ───────────────────────────────────────────────
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},

            # ── Network ──────────────────────────────────────────────────
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },

            # ── PPO hyperparameters ──────────────────────────────────────
            "entropy_coeff": 0.01,         # higher than 0.005 — BC policy is too deterministic
            "lambda": 0.95,                # GAE — reduce variance
            "grad_clip": 0.5,              # prevent catastrophic updates to BC weights
            "clip_param": 0.2,             # tighter PPO clipping
            "train_batch_size": 20000,     # large batch for stable gradients
            "sgd_minibatch_size": 2048,    # stable gradient estimates
            "num_sgd_iter": 10,            # avoid overfitting per batch
            "vf_loss_coeff": 0.5,          # reduce VF dominance on shared layers
            "lr_schedule": [               # start from 0 — this is a new training
                [0, 5e-5],                 # low lr to preserve BC initialization
                [10_000_000, 2e-5],        # mid
                [20_000_000, 1e-5],        # fine-tune
            ],

            # ── Rollout ──────────────────────────────────────────────────
            "rollout_fragment_length": 1000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 30_000_000,
            "time_total_s": 41400,         # 11.5 hours
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_trial)
    print(best_ckpt)
    print("Done training")
