"""
Experiment D: Separate policy and value networks + from checkpoint-5122 (87%).

Hypothesis: vf_share_layers=True causes VF loss to interfere with policy gradient.
PPO best practices (37 Implementation Details) recommend separate networks.

Note: Changing vf_share_layers from True to False changes the model architecture,
so we can't directly restore weights. Instead we restore and let Ray handle the
mismatch — it will keep compatible weights and reinitialize incompatible ones.
"""
import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3
OPPONENT_UPDATE_COOLDOWN = 30

CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

RESTORE_CHECKPOINT = (
    "./ray_results/PPO_exp_d/"
    "PPO_Soccer_29b63_00000_0_2026-04-20_11-54-07/checkpoint_001122/checkpoint-1122"
)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0 or agent_id == 1:
        return "default"
    return np.random.choice(
        ["opponent_1", "opponent_2", "opponent_3"],
        p=[0.15, 0.15, 0.70],  # 70% ceia
    )


def _load_weights(checkpoint_path, policy_name="default"):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class MixedOpponentCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._ceia_initialized = False
        self._last_update_iter = 0

    def on_train_result(self, **info):
        trainer = info["trainer"]
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
    tmp = create_rllib_env()
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()

    analysis = tune.run(
        "PPO",
        name="PPO_exp_d",
        config={
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": MixedOpponentCallback,
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
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            # ── Key change: separate policy and value networks ───────────
            "model": {
                "vf_share_layers": False,       # was True — separate networks
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            # PPO hyperparameters — same as best run (#10) but vf_loss_coeff=1.0
            # since separate networks don't have the VF-domination problem
            "entropy_coeff": 0.005,
            "lambda": 0.95,
            "grad_clip": 0.5,
            "clip_param": 0.2,
            "train_batch_size": 20000,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 10,
            "vf_loss_coeff": 1.0,              # was 0.5 — safe with separate networks
            "lr_schedule": [
                [0, 1e-4],
                [15_000_000, 5e-5],
                [30_000_000, 1e-5],
            ],
            "rollout_fragment_length": 1000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 100_000_000,
            "time_total_s": 165600,  # 82800 (restored) + 82800 (two more 11.5h rounds)
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        **({"restore": RESTORE_CHECKPOINT} if RESTORE_CHECKPOINT else {}),
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    print(best_trial)
    print(best_ckpt)
    print("Done training")
