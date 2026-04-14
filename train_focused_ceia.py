"""
Focused training against ceia_baseline with optimized PPO hyperparameters.

Strategy:
  Team 0: agents 0, 1 → always "default" (trained together)
  Team 1: agents 2, 3 → always "opponent_ceia" (fixed ceia_baseline weights)

Key changes from train_PPO_team.py:
  1. 100% CEIA opponent (no self-play) — concentrate all gradient signal on beating ceia
  2. PPO hyperparameter fixes — entropy, GAE lambda, grad clip, batch sizes
  3. Simplified 2-policy setup and callback
"""
import os
import pickle

import ray
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

RESTORE_CHECKPOINT = None  # Train from scratch with optimized hyperparameters
# ──────────────────────────────────────────────────────────────────────────────


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Team 0 (agents 0, 1) → "default" (being trained)
    Team 1 (agents 2, 3) → "opponent_ceia" (fixed ceia_baseline)
    """
    if agent_id == 0 or agent_id == 1:
        return "default"
    return "opponent_ceia"


def _load_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Extract policy weights from a Ray checkpoint file."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class CEIACallback(DefaultCallbacks):
    """Load ceia_baseline weights into opponent_ceia on first iteration."""

    def __init__(self):
        super().__init__()
        self._initialized = False

    def on_train_result(self, **info):
        if not self._initialized:
            trainer = info["trainer"]
            print("=== Loading ceia_baseline into opponent_ceia ===")
            try:
                weights = _load_weights(CEIA_CHECKPOINT)
                trainer.set_weights({"opponent_ceia": weights})
                print("=== opponent_ceia initialized (fixed forever) ===")
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
        name="PPO_focused_ceia",
        config={
            # ── System ────────────────────────────────────────────────────
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CEIACallback,

            # ── Multiagent: 2 policies only ───────────────────────────────
            "multiagent": {
                "policies": {
                    "default":       (None, obs_space, act_space, {}),
                    "opponent_ceia": (None, obs_space, act_space, {}),
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

            # ── PPO hyperparameters (fixed from defaults) ─────────────────
            "entropy_coeff": 0.005,         # was 0.0 — prevent premature convergence
            "lambda": 0.95,                 # was 1.0 — reduce variance with GAE
            "grad_clip": 0.5,              # was None — prevent catastrophic updates
            "clip_param": 0.2,             # was 0.3 — tighter PPO clipping (paper default)
            "train_batch_size": 20000,     # was 4000 — use more collected data
            "sgd_minibatch_size": 2048,    # was 128 — more stable gradient estimates
            "num_sgd_iter": 10,            # was 30 — less overfitting per batch
            "vf_loss_coeff": 0.5,          # was 1.0 — reduce VF dominance on shared layers
            "lr_schedule": [               # was fixed 5e-5
                [0, 3e-4],                 # start higher for faster progress
                [5_000_000, 1e-4],         # decay mid-training
                [15_000_000, 3e-5],        # fine-tune at end
            ],

            # ── Rollout ──────────────────────────────────────────────────
            "rollout_fragment_length": 1000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 30_000_000,
            "time_total_s": 82800,  # 23h (leave 1h buffer for PACE cleanup)
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        **({"restore": RESTORE_CHECKPOINT} if RESTORE_CHECKPOINT else {}),
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_trial)
    print(best_ckpt)
    print("Done training")
