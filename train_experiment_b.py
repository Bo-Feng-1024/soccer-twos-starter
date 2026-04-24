"""
Experiment B: Larger network [512,512] + near-default PPO hyperparameters.

Hypothesis: [256,256] network lacks capacity for complex 2v2 strategies.
Larger network + mostly-default hyperparameters (which Frank used successfully).
Trains from scratch since network architecture is incompatible with old checkpoints.
"""
import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3
OPPONENT_UPDATE_COOLDOWN = 20

CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)


def policy_mapping_fn(agent_id, *args, **kwargs):
    """Same mix as Frank's train_PPO_team.py: 50% ceia + 50% selfplay."""
    if agent_id == 0 or agent_id == 1:
        return "default"
    return np.random.choice(
        ["opponent_1", "opponent_2", "opponent_3"],
        p=[0.30, 0.20, 0.50],  # 50% ceia (Frank's original ratio)
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
        if default_reward > 0.3 and since_last >= OPPONENT_UPDATE_COOLDOWN:
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
        name="PPO_exp_b",
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
            # ── Larger network ───────────────────────────────────────────
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],    # was [256,256] — more capacity
                "fcnet_activation": "relu",
            },
            # ── Near-default PPO hyperparameters ─────────────────────────
            # Only grad_clip and entropy_coeff are non-default
            "entropy_coeff": 0.005,
            "grad_clip": 0.5,
            # Everything else uses Ray 1.4.0 defaults:
            # clip_param=0.3, train_batch_size=4000, sgd_minibatch_size=128,
            # num_sgd_iter=30, vf_loss_coeff=1.0, lambda=1.0, lr=5e-5
            "rollout_fragment_length": 5000,    # Frank's original value
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 30_000_000,
            "time_total_s": 41400,  # 11.5h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # NO restore — training from scratch with new network
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    print(best_trial)
    print(best_ckpt)
    print("Done training")
