"""
PPO training with dense reward shaping vs. a stationary opponent.

Key difference from example_ray_ppo_sp_still.py:
  env_config["reward_shaping"] = True
  → wraps the env with RewardShapingWrapper (see utils.py), which adds:
      - ball proximity bonus  (PROXIMITY_WEIGHT = 0.001)
      - ball progress bonus   (PROGRESS_WEIGHT = 0.002)

This is the "Agent2 – reward modification" submission.
Compare its learning curve against the baseline (example_ray_ppo_sp_still.py)
to show faster convergence or higher asymptotic reward.

Runs on both local Mac and PACE cluster without modification:
  - PACE: detects GPU + more CPUs, saves to /scratch/$USER/
  - Local: no GPU, 2 workers, saves to ./ray_results/
"""

import os
import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1

# Auto-detect environment: PACE sets the SLURM_JOB_ID variable.
ON_PACE = "SLURM_JOB_ID" in os.environ
NUM_WORKERS = 10 if ON_PACE else 2
NUM_GPUS = 1 if ON_PACE else 0
LOCAL_DIR = os.path.join("/scratch", os.environ.get("USER", ""), "soccer-twos") if ON_PACE else "./ray_results"

print(f"Running on {'PACE cluster' if ON_PACE else 'local machine'}: "
      f"{NUM_WORKERS} workers, {NUM_GPUS} GPUs, saving to {LOCAL_DIR}")


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_reward_shaped",
        config={
            # system
            "num_gpus": NUM_GPUS,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # environment
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,   # opponent stays still
                "reward_shaping": True,             # <-- the modification
            },
            # model (same as baseline for fair comparison)
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 20_000_000,
            "time_total_s": 43200 if ON_PACE else 999999,  # 12h cap on PACE
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir=LOCAL_DIR,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
