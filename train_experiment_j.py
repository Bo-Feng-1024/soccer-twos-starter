"""
Experiment J: Frame stacking (4 frames) + weight injection from checkpoint-5122.

Frame stacking gives the agent temporal information (ball velocity, movement
direction) that is missing from a single 336-dim observation. Based on
Brandão 2022 which uses 8-frame stacking.

Since frame stacking changes obs dim (336 → 1344), we can't restore from
checkpoint-5122 directly. Instead we:
  1. Start from scratch (random weights)
  2. On first iteration, inject checkpoint-5122's weights into the new network
     - First layer: copy old (256,336) weights to the last 336 columns of new (256,1344)
     - All other layers: direct copy (same shape)
  3. This gives the agent an ~83% starting point, then train with self-play

Uses same Brandão-inspired hyperparameters as experiment I:
  - 70% self-play + 30% ceia
  - lr = 3e-4, num_sgd_iter = 5
"""
import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3
NUM_FRAMESTACKS = 4
OPPONENT_UPDATE_COOLDOWN = 20

CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

# Old checkpoint for weight injection (NOT restore — different obs dim)
OLD_CHECKPOINT = (
    "./ray_results/PPO_focused_ceia/"
    "PPO_Soccer_396a8_00000_0_2026-04-16_07-55-55/checkpoint_005122/checkpoint-5122"
)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0 or agent_id == 1:
        return "default"
    return np.random.choice(
        ["opponent_1", "opponent_2", "opponent_3"],
        p=[0.35, 0.35, 0.30],
    )


def _load_weights(checkpoint_path, policy_name="default"):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


def _inject_weights(trainer, old_checkpoint_path):
    """Inject old checkpoint weights into new architecture (different obs dim).

    For the first layer where shapes differ (old: 256x336, new: 256x1344),
    copy old weights to the last 336 columns (= current frame position).
    All other layers have matching shapes and are copied directly.
    """
    old_weights = _load_weights(old_checkpoint_path, "default")
    new_weights = trainer.get_weights(["default"])["default"]

    injected = {}
    matched, adapted, skipped = 0, 0, 0

    for key in new_weights:
        if key in old_weights:
            old_val = np.array(old_weights[key])
            new_val = np.array(new_weights[key])
            if old_val.shape == new_val.shape:
                injected[key] = old_weights[key]
                matched += 1
            elif len(old_val.shape) == 2 and len(new_val.shape) == 2:
                # Weight matrix with different input dim (first layer)
                # new_val shape: (out, new_in), old_val shape: (out, old_in)
                # Copy old weights to last old_in columns (= current frame)
                result = np.zeros_like(new_val)
                result[:, -old_val.shape[1]:] = old_val
                injected[key] = result
                adapted += 1
                print(f"  Adapted {key}: {old_val.shape} -> {new_val.shape}")
            else:
                injected[key] = new_weights[key]
                skipped += 1
                print(f"  Skipped {key}: shape mismatch {old_val.shape} vs {new_val.shape}")
        else:
            injected[key] = new_weights[key]
            skipped += 1

    print(f"=== Weight injection: {matched} matched, {adapted} adapted, {skipped} skipped ===")
    trainer.set_weights({"default": injected})


class FrameStackCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._weights_injected = False
        self._ceia_initialized = False
        self._last_update_iter = 0

    def on_train_result(self, **info):
        trainer = info["trainer"]

        if not self._weights_injected:
            print("=== Injecting checkpoint-5122 weights into frame-stacked model ===")
            try:
                _inject_weights(trainer, OLD_CHECKPOINT)
            except Exception as e:
                print(f"WARNING: weight injection failed: {e}")
            self._weights_injected = True

        if not self._ceia_initialized:
            print("=== Loading ceia_baseline into opponent_3 (fixed) ===")
            try:
                ceia_weights = _load_weights(CEIA_CHECKPOINT)
                trainer.set_weights({"opponent_3": ceia_weights})
                print("=== opponent_3 = ceia_baseline (fixed forever) ===")
            except Exception as e:
                print(f"WARNING: failed to load ceia_baseline: {e}")
            self._ceia_initialized = True

        current_iter = info["result"]["training_iteration"]
        default_reward = info["result"].get("policy_reward_mean", {}).get("default", -999)
        since_last = current_iter - self._last_update_iter

        if default_reward > 0.1 and since_last >= OPPONENT_UPDATE_COOLDOWN:
            print(f"---- Updating selfplay opponents (iter={current_iter}, reward={default_reward:.3f}) ----")
            trainer.set_weights({
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
            })
            self._last_update_iter = current_iter


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    # Create temp env WITH frame stacking to get correct obs_space
    tmp = create_rllib_env({"num_framestacks": NUM_FRAMESTACKS})
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()
    print(f"Observation space with {NUM_FRAMESTACKS}-frame stack: {obs_space.shape}")

    analysis = tune.run(
        "PPO",
        name="PPO_exp_j",
        config={
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": FrameStackCallback,
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
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "num_framestacks": NUM_FRAMESTACKS,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            # ── PPO hyperparameters (Brandão-inspired) ──────────────────
            "entropy_coeff": 0.005,
            "lambda": 0.95,
            "grad_clip": 0.5,
            "clip_param": 0.2,
            "train_batch_size": 20000,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 5,
            "vf_loss_coeff": 0.5,
            "lr_schedule": [
                [0, 3e-4],
                [15_000_000, 1e-4],
                [30_000_000, 5e-5],
            ],
            "rollout_fragment_length": 1000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 100_000_000,
            "time_total_s": 41400,  # 11.5h
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # NO restore — different architecture, weights injected via callback
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_trial)
    print(best_ckpt)
    print("Done training")
